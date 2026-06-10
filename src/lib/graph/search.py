"""_GraphSearcherImpl: all read operations over the Graphiti graph."""

import re

from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from src.lib.graph.utils import record_id_to_group_id
from src.lib.util import get_logger

logger = get_logger(__name__)


class _GraphSearcherImpl:
    """Handles all read operations: search, narrative, table, existence checks.

    SRP: changes to query config, result ranking, or BM25 params live here
    only, without touching indexing or schema logic.
    """

    _CELL_FACT_RE = re.compile(
        r"In column \[(?P<col>.+?)\], the row \[(?P<row>.+?)\] has value (?P<val>.+)$"
    )


    def __init__(self, graphiti, indexed_ids: set[str]) -> None:
        self._graphiti = graphiti
        self._indexed_ids = indexed_ids

    async def is_indexed(self, record_id: str) -> bool:
        """Check if a document is already indexed; persistent across process restarts."""
        group_id = record_id_to_group_id(record_id)
        if group_id in self._indexed_ids:
            return True
        result, _, _ = await self._graphiti.driver.execute_query(
            "MATCH (n:Episodic {group_id: $gid}) RETURN count(n) AS cnt",
            gid=group_id,
        )
        if result and result[0].get("cnt", 0) > 0:
            self._indexed_ids.add(group_id)
            return True
        return False

    async def _run_search(self, record_id: str, question: str, config: SearchConfig):
        group_id = record_id_to_group_id(record_id)
        clean_question = question.replace("&", " ").replace("/", " ")
        return await self._graphiti.search_with_config(
            query=clean_question,
            group_ids=[group_id],
            config=config,
        )

    async def search(self, record_id: str, question: str) -> list[str]:
        """Return the top retrieved facts, highest reranker score first.

        We (1) attach each fact's reranker score and (2) sort the combined
        table+narrative set best-first, so the model isn't biased by a
        table-always-first ordering. We deliberately do NOT trim or threshold:
        the score is surfaced on each line and the model decides what to use.

        Each line is prefixed with ``[relevance <score>]`` so the model can
        weight a strong match over a weak one instead of guessing from text
        similarity alone.
        """
        tagged = await self._search_tagged(record_id, question)
        combined = tagged["table_facts"] + tagged["narrative_chunks"]
        # Sort best-first so the model isn't biased by a table-always-first
        # ordering. We deliberately do NOT trim or threshold: the reranker
        # score is surfaced on each line (below) and the model decides what to
        # use. A hard cap risks dropping the narrative fact that holds the
        # answer; a fixed score floor is unreliable because RRF scores are not
        # calibrated across queries or across the edge/episode rerankers.
        combined.sort(key=lambda r: r["score"] if r.get("score") is not None else -1.0, reverse=True)
        lines: list[str] = []
        for r in combined:
            score = r.get("score")
            lines.append(f"[relevance {score:.3f}] {r['fact']}" if score is not None else r["fact"])
        return lines

    async def _search_tagged(self, record_id: str, question: str) -> dict[str, list[dict]]:
        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={"limit": 50})
        results = await self._run_search(record_id, question, config)
        edge_scores = getattr(results, "edge_reranker_scores", None) or []
        table_facts: list[dict] = []
        for i, edge in enumerate(results.edges):
            if edge.fact:
                score = edge_scores[i] if i < len(edge_scores) else None
                table_facts.append({"fact": edge.fact, "score": score})
        episode_scores = getattr(results, "episode_reranker_scores", None) or []
        narrative_chunks: list[dict] = []
        seen_chunks: set[str] = set()
        for i, episode in enumerate(results.episodes):
            content = (episode.content or "").strip()
            if content and content not in seen_chunks:
                score = episode_scores[i] if i < len(episode_scores) else None
                narrative_chunks.append({"fact": content, "score": score})
                seen_chunks.add(content)
        return {"table_facts": table_facts, "narrative_chunks": narrative_chunks}

    async def search_narrative(self, record_id: str, question: str) -> list[dict]:
        config = SearchConfig(
            episode_config=EpisodeSearchConfig(
                search_methods=[EpisodeSearchMethod.bm25],
                reranker=EpisodeReranker.rrf,
            ),
            limit=50,
        )
        results = await self._run_search(record_id, question, config)
        chunks: list[dict] = []
        seen: set[str] = set()
        for episode in results.episodes:
            content = (episode.content or "").strip()
            if not content or content in seen:
                continue
            seen.add(content)
            name = getattr(episode, "name", "") or ""
            source = "pre" if "::pre_text::" in name else "post" if "::post_text::" in name else "unknown"
            chunks.append({"source": source, "snippet": content})
        return chunks

    async def search_table(self, record_id: str, question: str) -> list[dict]:
        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                reranker=EdgeReranker.rrf,
            ),
            limit=20,
        )
        results = await self._run_search(record_id, question, config)
        cells: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for edge in results.edges:
            m = self._CELL_FACT_RE.match(edge.fact or "")
            if not m:
                continue
            col, row, val = m.group("col"), m.group("row"), m.group("val")
            if (col, row) in seen:
                continue
            seen.add((col, row))
            try:
                parsed: float | str = float(val)
            except ValueError:
                parsed = val
            cells.append({"column": col, "row": row, "value": parsed})
        return cells
