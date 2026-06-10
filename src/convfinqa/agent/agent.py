"""FinancialAgent: answers conversational questions using graph search + LLM reasoning."""

import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tabulate import tabulate

from src.config import settings
from src.convfinqa.agent.prompts import (
    RETRIEVED_FACTS_NOTE,
    SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
    build_user_prompt,
)
from src.convfinqa.dataset.models import ConvFinQARecord
from src.lib.answers.tool_calling.interfaces import ToolRegistry
from src.lib.answers.tool_calling.tool_registry import ConvFinQAToolRegistry
from src.lib.conversation.interfaces import HistoryStore
from src.lib.graph.interfaces import GraphSearcher
from src.lib.util import eval_expression, get_logger

logger = get_logger(__name__)


# Hard cap on tool calls per question before forcing the final answer.
# Generous enough for compound questions and exploratory schema probing,
# tight enough to prevent runaway loops on stuck reasoning.
MAX_TOOL_CALLS = 15

_TABLE_HEADER = "Table (pre-loaded, full):\n"

# Models that do not accept an explicit temperature parameter (reasoning models
# and gpt-5-* which only supports its default of 1). For all other models we
# pin temperature=0 so eval runs are deterministic and reproducible.
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _temp_kwargs(model: str) -> dict:
    """Return {"temperature": 0} for models that support it, {} otherwise."""
    if any(model.startswith(p) for p in _NO_TEMPERATURE_PREFIXES):
        return {}
    return {"temperature": 0}


def _to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt a stored Chat-Completions thread to Responses API ``input`` items.

    The single-pass thread only ever holds ``system``/``user``/``assistant``
    text messages, which pass through unchanged. The one exception is the Mode 3
    tool-loop *fallback*, which routes through here with a thread that may also
    contain ``tool``-role messages and assistant messages whose only payload is
    ``tool_calls`` (content ``None``). The Responses API doesn't accept that
    Chat-Completions tool shape, so we drop those entries and keep the textual
    turns -- the fallback degrades to the dialogue text, never crashes.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


class FinancialAnswer(BaseModel):
    """Schema the model fills via JSON structured outputs on every answer path
    (single-pass Modes 1/2 and the Mode 3 tool loop).

    Replaces the old free-text + ``ANSWER:`` regex protocol: the model returns a
    strict JSON object, so the answer is read from typed fields -- there is no
    text scraping or regex anywhere. ``determinable`` carries the
    "can this be answered?" decision as a boolean instead of a magic
    ``"unknown"`` string; ``answer`` is a number or a math expression that
    ``simpleeval`` evaluates (arithmetic, not regex), so the model can offload
    computation and we still store a precise value.

    Used as ``text_format`` for ``client.responses.parse`` -- the JSON Schema is
    derived from these fields and enforced server-side, so the response is
    guaranteed parseable rather than best-effort matched.
    """

    reasoning: str = Field(
        description="Step-by-step working: reference resolution, where the value "
        "was found, and any arithmetic performed."
    )
    determinable: bool = Field(
        description="True if the answer can be determined from the document, "
        "retrieved facts, or conversation history. False if it cannot -- do NOT "
        "guess; set this to false instead."
    )
    answer: str = Field(
        description="The final value when determinable: a decimal number "
        "(e.g. '0.141') or a math expression to evaluate (e.g. '11 - 8', "
        "'108 / 6197'). Express percentages as decimals (0.141, not 14.1%). "
        "Leave empty when determinable is false."
    )


def _finalize_answer(parsed: FinancialAnswer) -> str:
    """Resolve a structured answer to its final string form -- no regex, no scraping.

    ``determinable=False`` becomes the ``"unknown"`` sentinel the evaluators
    expect. Otherwise the ``answer`` field (a number or a math expression like
    ``'11 - 8'``) is evaluated with simpleeval so history stores a precise
    number; a non-numeric expression falls through unchanged.
    """
    if not parsed.determinable:
        return "unknown"
    value = parsed.answer.strip()
    try:
        return str(eval_expression(value))
    except Exception as e:
        logger.debug("simpleeval could not evaluate %r: %s", value, e)
    return value


def _to_responses_tools(chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten Chat-Completions tool schemas to the Responses API tool shape.

    Chat Completions nests the spec under ``function`` (``{"type":"function",
    "function":{name,description,parameters}}``); the Responses API takes those
    fields flat (``{"type":"function","name",...}``). ``strict=False`` keeps the
    existing parameter schemas valid without the strict-mode requirements
    (every field required + ``additionalProperties:false``).
    """
    out: list[dict[str, Any]] = []
    for t in chat_tools:
        fn = t["function"]
        out.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn["parameters"],
            "strict": False,
        })
    return out



def _build_tool_system_content(record: "ConvFinQARecord") -> str:
    """Assemble the system content for tool-loop turns: prompt + pre-loaded table."""
    return f"{TOOL_SYSTEM_PROMPT}\n\n{_TABLE_HEADER}{_format_table_grid(record.doc.table)}"


def _format_table_grid(table: dict[str, dict[str, Any]]) -> str:
    """Render a ConvFinQA table as a grid string for the model to scan.

    Input shape: ``{column: {row: value}}`` (e.g. ``{"2007": {"net income": 100, ...}, ...}``).
    Output: a `tabulate`-rendered grid with rows as table rows and columns as
    table columns, matching how financial tables are typically displayed.

    Number formatting: integer-valued floats render as integers (``12000000``,
    not ``1.2e+07``); fractional floats render at full precision (``2.61``);
    non-numeric cells pass through as-is. This avoids tabulate's default
    scientific notation on large integers, which is harder for the model
    to scan and to compare against question wording.
    """
    if not table:
        return "(empty table)"

    def _fmt(v: Any) -> str:
        # Integer-valued floats -> int form (e.g. 12000000.0 -> "12000000")
        if isinstance(v, float):
            if v.is_integer():
                return str(int(v))
            return str(v)
        # Bools, ints, strings, None pass through
        return str(v) if v is not None else ""

    columns = list(table.keys())
    # ConvFinQA tables are well-formed: all columns share the same row labels
    # in the same order. Use the first column's keys directly instead of
    # iterating all columns to build a deduplicated union.
    rows = list(table[columns[0]].keys())
    data = [
        {"metric": row, **{col: _fmt(table[col].get(row, "")) for col in columns}}
        for row in rows
    ]
    return tabulate(data, headers="keys", tablefmt="grid")


async def _rewrite_query(
    client: AsyncOpenAI, question: str, history: str, model: str
) -> str:
    """Rewrite ``question`` into a focused search query using conversation history.

    Resolves references like 'that', 'it', 'those' so graph search operates on
    an explicit query rather than a pronoun. Falls back to the raw question on
    any API error so a transient outage never breaks the retrieval path.

    Extracted from ``FinancialAgent`` because query rewriting is an independent
    concern: it needs only a client + prompt, not the agent's graph or store.
    """
    if not history:
        return question
    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"New question: {question}\n\n"
        f"Write a concise search query (10-15 words) to find the relevant facts in a financial document. "
        f"Resolve any references like 'that', 'it', 'those' using the conversation history. "
        f"Output only the search query, nothing else."
    )
    try:
        resp = await client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=50,
            **_temp_kwargs(model),
        )
        content = resp.output_text
        return content.strip() if content else question
    except Exception as e:
        logger.debug("Query rewrite failed, using raw question: %s", e)
        return question


class ContextBuilder:
    """Assembles the per-turn context block passed to the LLM.

    Extracted from ``FinancialAgent`` so context-assembly changes (truncation
    policy, new sections, different table rendering) are isolated from the
    agent's orchestration logic.

    All methods are static: the builder holds no state and needs no instance.
    """

    @staticmethod
    def build(facts: list[str], record: ConvFinQARecord) -> str:
        """Build the context block for a single question turn.

        Routes to the facts path (Mode 1) when ``facts`` is non-empty, or the
        raw-document fallback (Mode 2) when graph search returned nothing.
        """
        table_block = _TABLE_HEADER + _format_table_grid(record.doc.table)
        if facts:
            return ContextBuilder._with_facts(table_block, facts)
        logger.debug("No graph results: using full document context as Mode 2 fallback")
        return ContextBuilder._mode2_fallback(record)

    @staticmethod
    def _with_facts(table_block: str, facts: list[str]) -> str:
        """Mode 1: table grid + all retrieved facts.

        The relevance-score guidance (RETRIEVED_FACTS_NOTE) is defined in
        prompts.py with the rest of the prompt copy and injected here, so it
        only reaches the graph-RAG path that actually has retrieved facts
        (Mode 2 / raw-doc shares the system prompt but has none).
        """
        header = f"{table_block}\n\n{RETRIEVED_FACTS_NOTE}\n"
        fact_lines = "".join(f"- {fact}\n" for fact in facts)
        return (header + fact_lines).rstrip()

    @staticmethod
    def _mode2_fallback(record: ConvFinQARecord) -> str:
        """Mode 2: full pre_text + table + post_text passed verbatim."""
        table_section = f"[Table]\n{_format_table_grid(record.doc.table)}"
        post_section = f"[Text after table]\n{record.doc.post_text}"
        pre_section = f"[Text before table]\n{record.doc.pre_text}"
        return f"Financial document:\n{pre_section}\n\n{table_section}\n\n{post_section}"


@dataclass
class ToolTrace:
    """One tool invocation recorded for post-hoc analysis."""

    tool_name: str
    args: dict
    result: object


@dataclass
class ToolAnswerOutcome:
    """Result of a tool-calling answer turn.

    Carries the final answer string alongside diagnostic signals (the tool
    trace, whether the loop cap was exhausted, whether we fell back to the
    single-pass path). The evaluation runner reads these to populate
    EvaluationResult.tool_traces and loop_cap_exhausted.
    """

    answer: str
    trace: list[ToolTrace] = field(default_factory=list)
    loop_cap_exhausted: bool = False
    fell_back_to_single_pass: bool = False


class FinancialAgent:
    """Answers conversational financial questions.

    Depends on GraphStore and HistoryStore protocols (DI): concrete
    implementations (FinancialGraph, ConversationStore) are injected at
    construction, enabling easy testing with mocks.
    """

    def __init__(
        self,
        graph: GraphSearcher,
        store: HistoryStore,
        openai_client: AsyncOpenAI | None = None,
        model: str | None = None,
        query_rewriter_model: str | None = None,
    ) -> None:
        self._graph = graph
        self._store = store
        self._client = openai_client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.openai_model
        self._query_rewriter_model = query_rewriter_model or settings.query_rewriter_model

    @property
    def graph(self) -> GraphSearcher:
        """The agent's GraphSearcher dependency (retrieval only).

        Exposed so collaborators (e.g. the evaluation runner) can call
        ``is_indexed`` without reaching into private attributes. Indexing
        operations (``index_record``, ``build_indices``) are on ``GraphIndexer``
        and must be accessed through the concrete object passed to the runner.
        """
        return self._graph

    @property
    def store(self) -> HistoryStore:
        """The agent's HistoryStore dependency. Exposed for the same reason
        as ``graph``: lets callers create conversations and persist turns
        through a stable public surface.
        """
        return self._store

    @property
    def client(self) -> AsyncOpenAI:
        """The agent's OpenAI client. Exposed so collaborators (e.g. the
        LLM judge in the evaluation CLI) can share it without reaching into
        private attributes.
        """
        return self._client

    async def answer(
        self,
        conversation_id: str,
        record: ConvFinQARecord,
        question: str,
        *,
        _system_prompt: str | None = None,
    ) -> str:
        """Answer a question using graph search + conversation history.

        Uses the same canonical multi-turn pattern as Mode 3: prior turns'
        user + assistant messages are replayed in the ``messages`` array, so
        the model sees the actual conversation thread (not a Q->A summary
        string injected into the user prompt). The per-turn retrieved facts
        stay inline in the CURRENT user message because they're a fresh
        graph search per turn -- prior turns' facts would be stale.

        ``_system_prompt`` is an internal override used by the tool-loop
        fallback to persist the correct TOOL_SYSTEM_PROMPT+table instead of
        SYSTEM_PROMPT, so subsequent tool-loop turns replay the right context.
        """
        # Query rewriting still uses the compact Q->A history string -- it's
        # a cheap auxiliary LLM call that only needs reference-resolution
        # context ("it"/"that"), not the full message thread.
        history = self._store.format_history(conversation_id)
        search_query = await _rewrite_query(self._client, question, history, self._query_rewriter_model)
        facts = await self._graph.search(record.id, search_query)
        context = ContextBuilder.build(facts, record)
        return await self._run_single_pass(conversation_id, record, question, context, _system_prompt)

    async def answer_raw_doc(
        self, conversation_id: str, record: ConvFinQARecord, question: str
    ) -> str:
        """Answer using the full raw document (Mode 2) -- skips graph search entirely.

        Builds context directly from ``pre_text + table + post_text`` without
        querying the graph. Use when the record is intentionally not indexed
        (``--no-index`` path) or when graph retrieval should be bypassed.
        """
        context = ContextBuilder.build([], record)  # empty facts -> _mode2_fallback
        return await self._run_single_pass(conversation_id, record, question, context)

    async def _run_single_pass(
        self,
        conversation_id: str,
        record: ConvFinQARecord,
        question: str,
        context: str,
        _system_prompt: str | None = None,
    ) -> str:
        """Shared LLM call + message build + persist for single-pass modes (1 and 2).

        Uses the Responses API with JSON structured outputs: the model is
        constrained to the ``FinancialAnswer`` schema, so the final value
        arrives in ``output_parsed`` and ``_finalize_answer`` only evaluates the
        ``answer`` field's arithmetic (e.g. ``'11 - 8'`` -> ``'3'``). No text
        scraping or regex.
        """
        user_content = build_user_prompt(context, question)

        prior_messages = self._store.load_messages(conversation_id)
        if not prior_messages:
            system_msg = {"role": "system", "content": _system_prompt or SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": user_content}
            messages = [system_msg, user_msg]
        else:
            user_msg = {"role": "user", "content": user_content}
            messages = [*prior_messages, user_msg]

        response = await self._client.responses.parse(
            model=self._model,
            input=_to_responses_input(messages),
            text_format=FinancialAnswer,
            **_temp_kwargs(self._model),
        )

        parsed = response.output_parsed
        raw = response.output_text or ""
        # Structured path: resolve the schema's fields. If the model refused
        # (no parsed object), fail closed to "unknown" rather than scraping text.
        answer = _finalize_answer(parsed) if parsed is not None else "unknown"

        assistant_msg = {"role": "assistant", "content": raw}
        new_msgs = (
            [system_msg, user_msg, assistant_msg]
            if not prior_messages
            else [user_msg, assistant_msg]
        )
        self._store.save_messages(conversation_id, new_msgs)
        self._store.save_turn(conversation_id, record.id, question, answer)
        logger.debug("Q: %s | A: %s", question, answer)
        logger.debug("REASONING:\n%s", raw)
        return answer

    async def answer_with_tools(
        self,
        conversation_id: str,
        record: ConvFinQARecord,
        question: str,
        registry: ToolRegistry | None = None,
    ) -> ToolAnswerOutcome:
        """Answer using the Responses API function-calling tool loop.

        Hard-capped at MAX_TOOL_CALLS tool invocations. The final answer comes
        back as a ``FinancialAnswer`` structured object via ``responses.parse``
        (no ``ANSWER:`` text parsing). If the model keeps calling tools after
        the cap-warning, falls back to the single-pass path so we never
        silently fail.

        ``registry`` is injected for testability: production callers use the
        default (ConvFinQAToolRegistry bound to this record + self._graph).
        """
        if registry is None:
            registry = ConvFinQAToolRegistry(record, self._graph)

        # Full-message replay (OpenAI / Vercel AI SDK canonical pattern):
        # load every prior turn's messages -- including the persisted system
        # message that carries TOOL_SYSTEM_PROMPT + the pre-loaded table --
        # so the model sees the entire conversation, byte-for-byte, that it
        # saw last time. See OpenAI's docs on multi-turn function calling:
        # "After appending the results to your `messages`, you can send
        # them back to the model to get a final response."
        prior_messages = self._store.load_messages(conversation_id)
        messages: list[dict] = self._build_tool_messages(record, prior_messages, question)
        # `save_start` is the index from which to persist after the loop.
        # On the FIRST turn (prior_messages empty) we just built a fresh
        # system + user pair, so we save everything from index 0 (system
        # included). On subsequent turns the system + prior dialogue are
        # already in the DB; only the new user message + this loop's
        # appendages need saving.
        save_start = 0 if not prior_messages else len(prior_messages)
        trace: list[ToolTrace] = []
        # Indices in `messages` of injected control-flow items (cap-warning).
        # Tracked by position -- not by matching their text -- so they can be
        # excluded from the persisted thread without any brittle substring detection.
        control_flow_idx: set[int] = set()
        # Loop up to MAX_TOOL_CALLS+1 turns: each iteration is one LLM call.
        # We expect ≤MAX_TOOL_CALLS tool invocations + 1 final structured answer.
        loop_exhausted = False
        for _turn in range(MAX_TOOL_CALLS + 2):
            response = await self._call_with_tools(messages, registry)
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not function_calls:
                # Final answer: read the structured schema, not free text.
                parsed = response.output_parsed
                answer = _finalize_answer(parsed) if parsed is not None else "unknown"
                # Persist the new thread items for replay on the next turn:
                # everything appended this turn (the user message, the
                # function_call / function_call_output exchange) except the
                # transient control-flow nudges (tracked by index), plus the
                # final assistant message. On the first turn this includes the
                # system item at index 0 (TOOL_SYSTEM_PROMPT + pre-loaded table).
                new_items = [
                    m for i, m in enumerate(messages)
                    if i >= save_start and i not in control_flow_idx
                ]
                new_items.append({"role": "assistant", "content": response.output_text or ""})
                self._store.save_messages(conversation_id, new_items)
                self._store.save_turn(conversation_id, record.id, question, answer)
                return ToolAnswerOutcome(
                    answer=answer, trace=trace, loop_cap_exhausted=loop_exhausted
                )

            # Execute each function call; append the call + its output as
            # Responses input items so the next turn replays the full exchange.
            for fc in function_calls:
                args = self._parse_args(fc.arguments)
                result = await registry.dispatch(fc.name, args)
                trace.append(ToolTrace(fc.name, args, result))
                messages.append({
                    "type": "function_call",
                    "call_id": fc.call_id,
                    "name": fc.name,
                    "arguments": fc.arguments,
                })
                messages.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": json.dumps(result, default=str),
                })

            if len(trace) >= MAX_TOOL_CALLS and not loop_exhausted:
                # Inject a single cap-warning and give the next iteration one
                # last chance to produce the structured answer.
                messages.append({
                    "role": "system",
                    "content": (
                        "You have used your tool budget of "
                        f"{MAX_TOOL_CALLS} calls. Respond now with your final "
                        "answer in the required structured format, using the facts "
                        "you have already retrieved. If a relevant value appeared "
                        "in any prior tool result, set determinable=true and use it "
                        "rather than giving up."
                    ),
                })
                control_flow_idx.add(len(messages) - 1)
                loop_exhausted = True
                continue

            if loop_exhausted:
                # Cap-warning already sent; the model still called more tools.
                # Fall through to single-pass with the facts accumulated.
                break

        # Fallback path: re-run the single-pass agent with whatever facts the
        # tool calls retrieved.
        logger.warning(
            "Tool loop exhausted for record %s; falling back to single-pass", record.id
        )
        # Pass TOOL_SYSTEM_PROMPT+table as the system override so the message
        # thread is persisted with the correct context. Without this, answer()
        # would save SYSTEM_PROMPT, and the next answer_with_tools() turn would
        # replay the wrong system (no tools, no pre-loaded table).
        tool_system_content = _build_tool_system_content(record)
        answer = await self.answer(
            conversation_id, record, question, _system_prompt=tool_system_content
        )
        return ToolAnswerOutcome(
            answer=answer,
            trace=trace,
            loop_cap_exhausted=True,
            fell_back_to_single_pass=True,
        )

    # ---- tool-loop helpers (kept private; small, testable units) ----

    def _build_tool_messages(
        self,
        record: ConvFinQARecord,
        prior_messages: list[dict],
        question: str,
    ) -> list[dict]:
        """Construct the initial message stack for the tool loop.

        Composition mirrors the OpenAI / Vercel AI SDK pattern:
          - On the **first turn** (no prior_messages), we build a fresh
            ``system`` containing TOOL_SYSTEM_PROMPT + the record's table
            (pre-loaded as a grid), then the user question.
          - On **subsequent turns**, prior_messages already starts with
            the persisted system message from turn 1; we replay it
            verbatim and append the current user question.

        This means the system message (including the pre-loaded table) is
        persisted byte-for-byte once per conversation and replayed on
        every subsequent turn -- so the model sees exactly what it saw
        before, even if TOOL_SYSTEM_PROMPT or the table-rendering code
        change between sessions.

        Narrative pre/post text is NOT pre-loaded: it's longer and most of
        it is irrelevant per question, so the model fetches the relevant
        snippets via ``search_document`` when needed.
        """
        if not prior_messages:
            # First turn: build fresh system + user. Will be persisted by
            # the caller (`save_start = 0`) so future turns replay this
            # exact system content.
            system_content = _build_tool_system_content(record)
            return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
            ]
        # Subsequent turn: prior_messages[0] is the persisted system.
        return [*prior_messages, {"role": "user", "content": question}]

    async def _call_with_tools(self, messages: list[dict], registry: ToolRegistry):
        """One Responses API call that may emit function_calls or a final answer.

        ``text_format`` constrains the final (non-tool) turn to the
        ``FinancialAnswer`` JSON schema; ``tools`` lets the model call functions
        mid-loop. The two compose: tool turns return ``function_call`` items,
        the closing turn returns a schema-conforming message (``output_parsed``).
        """
        return await self._client.responses.parse(
            model=self._model,
            input=messages,
            tools=_to_responses_tools(registry.tools_for_openai()),
            text_format=FinancialAnswer,
            **_temp_kwargs(self._model),
        )

    @staticmethod
    def _parse_args(raw: str) -> dict:
        """Parse the LLM-emitted tool-call args (JSON). Returns {} on malformed input."""
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            logger.warning("Malformed tool args: %s", raw)
            return {}

    # ---- backward-compat shim (tests call this on the agent instance) --------

    async def _rewrite_query(self, question: str, history: str) -> str:
        return await _rewrite_query(self._client, question, history, self._query_rewriter_model)
