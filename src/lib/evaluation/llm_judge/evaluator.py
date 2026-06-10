"""LLM-as-judge: semantic answer verification via a small LLM call.

Complements the seven-path code evaluator in `metrics.py`. The code evaluator
handles numeric tolerance precisely and is free; the LLM judge handles
semantic equivalence the code evaluator can't (yes/no answers vs numeric,
unit-aware comparisons, alternative valid interpretations).

The two evaluators are *run side by side*: disagreements are interesting
signals worth logging, not failures to mask.
"""

from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.lib.evaluation.base import EvalVerdict
from src.lib.util import get_logger

logger = get_logger(__name__)


@dataclass
class JudgeVerdict:
    """The judge's decision plus a one-sentence rationale.

    `confidence` is "high" or "low": the judge declares low confidence when
    the call is genuinely close (e.g. sign ambiguity, unit interpretation).
    """

    correct: bool
    reason: str
    confidence: str = "high"


class _JudgeSchema(BaseModel):
    """JSON structured-output schema the judge model is constrained to.

    Replaces the old ``response_format={"type": "json_object"}`` + manual
    ``json.loads`` (which could surface malformed JSON or missing keys). With a
    JSON Schema enforced server-side, ``correct`` / ``reason`` / ``confidence``
    are guaranteed present and correctly typed.
    """

    correct: bool = Field(description="True iff the prediction is semantically equivalent to the gold answer.")
    reason: str = Field(description="One-sentence rationale citing the rule applied and question type.")
    confidence: Literal["high", "low"] = Field(description="'low' when the call is genuinely close or a dataset-annotation issue is suspected.")


_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for financial QA on the ConvFinQA dataset.

Given a (question, gold answer, predicted answer) triple, decide whether the prediction is semantically equivalent to the gold in the context of the question.

# STEP 1: Classify the question

First, identify the expected answer SHAPE from the question wording:

- *Raw lookup*: "what was X in YEAR?" → expect a number in the table's units (often millions/billions of dollars).
- *Raw change/difference*: "what was the change in X" / "the difference between" → expect a signed subtraction with the SAME units as the operands. NOT a percentage.
- *Percent / proportion / fraction*: "what percent" / "what fraction" / "what proportion" / "what ratio of X to Y" → expect a decimal in [0, 1] OR a percentage in [0, 100]. The two are equivalent.
- *Percent change / growth rate*: "what was the percentage change" / "by what percent did X grow" → expect a decimal (signed) representing relative change.
- *Year count*: "how many years from A to B" → expect end − start (exclusive count), not inclusive.
- *Comparison / yes-no*: "did X grow?" / "is X greater than Y?" → expect "yes"/"no" or an equivalent numeric signal.
- *Aggregation*: "total of A and B" → expect sum of the operands.

# STEP 2: Apply tolerance rules

## ACCEPT as equivalent:

1. *Numeric tolerance*: within ±1% relative, or 1e-3 absolute when values are small.
   - 60.94 ≈ 60.94000 ✓
   - 35.8 ≈ 35.80001 ✓
   - 0.10745 ≈ 0.10745008927122221 ✓ (long decimals from precise math)
   - Currency and thousands separators are cosmetic: `$1,500` ≡ `1500` ✓.

2. *Scale equivalence: accept ANY mathematically-equivalent scale*:
   - 1.5 billion ≡ 1500 million ≡ 1,500,000,000 ✓
   - 21M ≡ 21,000,000 ≡ 21 (when prior context establishes millions) ✓
   - 2200 (millions, as the table reports) ≡ 2200000000 (raw dollars) ✓: gold and pred differ only by 10^6; this IS scale equivalence, ACCEPT.
   - 503 (millions) ≡ 503000000 (raw) ✓: same pattern, ACCEPT.
   - 2200 (thousands) ≡ 2.2 (millions) ✓: gold in thousands, pred in millions, differ by 10^3. ACCEPT.
   - 2525 (thousands) ≡ 2.525 (millions) ✓: same pattern. ACCEPT.
   - **Be consistent: if a (gold, pred) pair differs by EXACTLY a power-of-1000 scale factor (10^3, 10^6, 10^9) in EITHER direction, and the relative-to-each-other magnitude is identical, ACCEPT: even if the question doesn't spell out the unit.** Financial tables routinely use implicit units (millions / thousands) that the gold inherits; the model returning a different but mathematically-equivalent scale is the same value. This applies whether the prediction is larger OR smaller than the gold.
   - REJECT only when the question EXPLICITLY names a unit and the answer is in a different unit (e.g. "in millions" + answer is 2,200,000,000,000 [trillions]: wrong unit).

3. *Percent ↔ decimal* ONLY when wording implies ratio/percentage (question type from Step 1 is *percent / proportion / fraction* or *percent change*):
   - "what percent is X of Y?" with gold=14.1 and pred=0.141 → ACCEPT (both are valid forms).
   - "what fraction" / "what proportion" → same: percent and decimal forms equivalent.
   - REJECT when question is *raw change/difference* and gold=35.8 vs pred=0.358: that's a raw value, not a ratio. Off by 100×.

4. *Both small but 100× off* on percentage-points questions:
   - gold=0.35 percent-points vs pred=0.0035 (decimal of same percentage) → ACCEPT.
   - Indicator: gold wording like "percentage point change" or values both < 1 with a 100× ratio.

5. *Sign flip* in three cases:
   (a) *Magnitude vs signed value* on direction-ambiguous wording:
       - "change in X" where X went 100→90: gold=10 (magnitude) vs pred=-10 (signed) → ACCEPT.
       - REJECT when wording is direction-explicit AND the value isn't a balance-sheet item: "what is the increase" → sign must be positive; "what is the drop" → sign must match the convention.
   (b) *Balance-sheet conventions*: financial line items like "accumulated other comprehensive losses", "deficit", "deductions", "less <item>" are conventionally stored as NEGATIVE on the balance sheet but reported as POSITIVE magnitudes in narrative. If gold and pred differ ONLY in sign and the question refers to a balance-sheet item that's typically negative (losses / deficit / deductions / "less" / "(accumulated)"), ACCEPT with confidence="low" and note the convention ambiguity in `reason`.
       - Example: gold=9402 (magnitude of "total accumulated other comprehensive losses") vs pred=-9402 (signed balance-sheet value) → ACCEPT, confidence=low.
   (c) Otherwise, sign-only disagreement on a numeric question → REJECT.

   Cascade caveat: if a sign disagreement in this turn would compound across the dialogue (later turns subtract or compare these values), the choice still matters: but ACCEPT applies *per turn*. Downstream cascade is its own evaluation.

6. *Yes/No ↔ numeric*:
   - "did X grow?" + positive change number → equivalent to "yes" ✓.
   - "is X greater than Y?" + correct comparison number ✓.

7. *Year-count convention*: "from 2008 to 2012" = 4 (end − start), not 5 (inclusive). gold=4 vs pred=5 → REJECT (inclusive count is wrong by convention here).

8. *Aggregation* answers must include ALL operands the question mentions:
   - "total of A, B, and C" requires A+B+C; missing any operand → REJECT.

## REJECT as different:

- *Wrong magnitude*: 100 vs 1000 (10× off), 35.8 vs 358 (10× off). No scale path saves these.
- *Wrong value entirely*: 35.8 vs 41.2 (different number, no tolerance path matches).
- *Inverse ratio mismatch*: NEVER accept 0.25 ≡ 4.0 or 0.0714 ≡ 14.0 just because they're mathematical inverses. Only accept genuine ratio swaps where BOTH values are in [0.5, 2.0] (canonical X/Y vs Y/X like 0.5 ≡ 2.0).
- *Numeric for yes/no when meaning doesn't match*: negative change for "did X grow?" → "yes" is wrong.
- *"unknown" / "error" predicted when a real numeric answer exists*: REJECT outright. Models should not punt on answerable questions.
- *Wrong sign on direction-explicit question*: "increase in revenue" with gold positive and pred negative → REJECT.
- *Wrong year referenced*: if the question explicitly says "in 2018" and the prediction used 2017 data, REJECT even if the math itself was right. Exception: when the question's year is clearly wrong (dataset annotation error): set confidence="low" and explain in `reason`.
- *Question/answer-type mismatch*: question asks for a count (integer) but prediction is a ratio (decimal), or vice versa → REJECT unless the question is itself ambiguous (then confidence="low").

# STEP 3: Dataset-annotation suspicion patterns

In the ConvFinQA dataset, roughly 9% of questions have annotation issues where the gold doesn't actually match the question wording. When the prediction looks *reasonable for the question* but disagrees with gold in one of these patterns, set `confidence="low"` and note the suspicion in `reason`. You should still return `correct: false` (the prediction does not equal the gold), but the low-confidence tag tells downstream analysis this is a candidate for human review, not a clear agent failure.

Suspicion patterns observed in this dataset:

1. *Question / answer-type mismatch*: question asks for a "total number of X" or "count" but gold is a *ratio* (a decimal in [0, 1]). Or question asks for a "percentage" but gold is a raw sum. The prediction matches the question's literal wording; the gold doesn't.

2. *Wrong year reference*: question explicitly says "in YEAR_A" but gold's program uses YEAR_B's values. Pattern: a later turn in the dialogue confirms the gold was from YEAR_B (e.g., calls the answer "the YEAR_B value").

3. *Question divisor mismatch*: question says "divided by 100" but gold's `turn_program` divides by 1000 (or similar). Prediction follows the question's literal divisor.

4. *Non-standard convention*: percentage-change questions usually use the *earlier* year as the base; if gold uses the later year, the prediction (using standard convention) will disagree. Often differs by ~5-10% relative.

5. *Mixed units without conversion in gold*: gold's `turn_program` subtracts or compares values in different units (e.g., 3.2 billion vs 2242 million) without a scale conversion. Prediction either matches the corrected calculation or returns "unknown".

6. *Missing data*: gold references a row/column/year that doesn't appear in the document. Prediction returns "unknown" or picks the closest available value.

7. *Mislabeled rows*: gold's values appear correct for swapped row labels (e.g., metric A's value used as metric B's).

When ANY of these match, the verdict is still based on numeric equivalence (correct: false if values differ), but `confidence="low"` and `reason` should name the pattern.

# STEP 4: Output

Return STRICT JSON (no prose, no markdown fences). When a source document was provided, include the two extra booleans:
{"gold_matches_document": true|false, "pred_matches_document": true|false, "correct": true|false, "reason": "<one-sentence rationale citing the rule applied, the question type, and any dataset-suspicion pattern>", "confidence": "high"|"low"}

When no source document was provided, omit the two `*_matches_document` fields:
{"correct": true|false, "reason": "...", "confidence": "high"|"low"}

CONFIDENCE GUIDANCE:
- "high" when at least one ACCEPT/REJECT rule clearly applies (numeric match, clear sign, unambiguous scale, definite type match).
- "low" when the call is genuinely close: ambiguous sign on a magnitude question, scale interpretation could go either way, the question wording itself is unclear, OR you suspect a dataset annotation error per the Step 3 patterns.
"""


async def judge_answer(
    question: str,
    gold: str | float,
    predicted: str,
    client: AsyncOpenAI,
    model: str = "gpt-5-mini",
) -> JudgeVerdict:
    """Ask an LLM whether predicted is semantically equivalent to gold.

    Uses the Responses API with a JSON structured-output schema (``_JudgeSchema``),
    so the verdict is guaranteed to parse. Defaults to gpt-5-mini -- a judge
    should be at least as strong as the model under evaluation, not weaker.
    Falls back to `correct=False` (with a logged reason) on API errors so a
    transient outage never silently passes a wrong answer.
    """
    user_msg = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}\n"
    )
    try:
        resp = await client.responses.parse(
            model=model,
            instructions=_JUDGE_SYSTEM_PROMPT,
            input=user_msg,
            text_format=_JudgeSchema,
            max_output_tokens=2000,
        )
        verdict = resp.output_parsed
        if verdict is None:
            raise ValueError("judge returned no parsed verdict")
        return JudgeVerdict(
            correct=verdict.correct,
            reason=verdict.reason,
            confidence=verdict.confidence,
        )
    except Exception as e:
        logger.warning("LLM judge failed for question %r: %s", question[:60], e)
        return JudgeVerdict(
            correct=False,
            reason=f"judge_error: {e}",
            confidence="low",
        )


class LLMJudgeEvaluator:
    """LLM-as-judge wrapped as an Evaluator.

    Holds the OpenAI client + model and exposes the same async `evaluate`
    signature as `CodeEvaluator`, so the runner can iterate a uniform list.
    """

    name = "judge"

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def evaluate(
        self, question: str, gold: str | float, predicted: str
    ) -> EvalVerdict:
        """Delegate to `judge_answer` and adapt to the common `EvalVerdict` shape."""
        v = await judge_answer(question, gold, predicted, self._client, self._model)
        return EvalVerdict(correct=v.correct, reason=v.reason, confidence=v.confidence)
