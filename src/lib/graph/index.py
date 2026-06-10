"""_GraphIndexerImpl: all write operations over the Graphiti graph."""

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from chonkie import SentenceChunker
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode

from src.config import settings
from src.lib.graph.interfaces import IndexableRecord
from src.lib.graph.utils import _parse_year, record_id_to_group_id
from src.lib.util import get_logger

logger = get_logger(__name__)

MIN_CHUNK_CHARS = 20
_CHUNKER = SentenceChunker(chunk_size=400, chunk_overlap=0)


class _GraphIndexerImpl:
    """Handles all write operations: schema setup, triplet/episode ingestion, bulk indexing.

    SRP: changes to chunking strategy, episode format, triplet schema, or
    retry/checkpoint logic live here only, without touching search logic.
    """

    def __init__(self, graphiti, indexed_ids: set[str]) -> None:
        self._graphiti = graphiti
        self._indexed_ids = indexed_ids

    async def build_indices(self) -> None:
        """Initialise Graphiti schema and indices safe to call multiple times."""
        try:
            await self._graphiti.build_indices_and_constraints(delete_existing=False)
        except Exception as e:
            fingerprint = getattr(e, "code", None) or str(e)
            if "EquivalentSchemaRuleAlreadyExists" in fingerprint:
                logger.debug("Indices already exist, skipping: %s", e)
            else:
                raise

    async def index_record(self, record: IndexableRecord) -> None:
        """Index a single record using the hybrid approach. Skips if already indexed."""
        group_id = record_id_to_group_id(record.id)
        if group_id in self._indexed_ids:
            logger.debug("Skipping already-indexed record (cached): %s", record.id)
            return
        # Idempotency across processes: the in-memory set is empty in a fresh
        # process (and index_bulk never consults the searcher's is_indexed), so
        # also check the graph itself before inserting. Without this, re-running
        # indexing re-inserts a group's nodes/edges and duplicates them.
        if await self._group_in_graph(group_id):
            self._indexed_ids.add(group_id)
            logger.debug("Skipping already-indexed record (present in graph): %s", record.id)
            return

        now = datetime.now(UTC)
        table_size = sum(len(m) for m in record.doc.table.values())
        logger.info("  [triplets] %s: %d cells", record.id, table_size)
        t1 = datetime.now(UTC).timestamp()
        await self._index_table_triplets(record, group_id, now)
        logger.info("  [triplets] done (%.1fs)", datetime.now(UTC).timestamp() - t1)

        episodes: list[RawEpisode] = []
        for section, text in [("pre_text", record.doc.pre_text), ("post_text", record.doc.post_text)]:
            if not text.strip():
                continue
            chunks = _CHUNKER(text.strip())
            for i, chunk in enumerate(chunks):
                content = chunk.text.strip()
                if len(content) < MIN_CHUNK_CHARS:
                    continue
                episodes.append(RawEpisode(
                    name=f"{record.id}::{section}::{i}",
                    content=content,
                    source=EpisodeType.text,
                    source_description=f"{section} chunk {i} for {record.id}",
                    reference_time=now,
                ))
        if episodes:
            logger.info("  [episodes] %s: %d chunks (embedding + LLM extraction)", record.id, len(episodes))
            t2 = datetime.now(UTC).timestamp()
            await self._graphiti.add_episode_bulk(episodes, group_id=group_id)
            logger.info("  [episodes] done (%.1fs)", datetime.now(UTC).timestamp() - t2)

        self._indexed_ids.add(group_id)
        logger.info("Indexed record: %s", record.id)

    async def _group_in_graph(self, group_id: str) -> bool:
        """True if this group_id already has Entity nodes in the graph.

        Every indexed record creates table-triplet Entity nodes, so this is a
        reliable "already indexed" signal and makes index_record idempotent for
        all callers (single and bulk), not just within one process.
        """
        result, _, _ = await self._graphiti.driver.execute_query(
            "MATCH (n:Entity {group_id: $gid}) RETURN count(n) AS cnt", gid=group_id
        )
        return bool(result and result[0].get("cnt", 0) > 0)

    async def _index_table_triplets(
        self, record: IndexableRecord, group_id: str, reference_time: datetime
    ) -> None:
        """Insert each table cell as a direct Graphiti triplet (no LLM extraction)."""

        def stable_uuid(name: str) -> str:
            h = hashlib.md5(f"{group_id}::{name}".encode()).hexdigest()
            return str(uuid.UUID(h))

        triplets = []
        for column, metrics in record.doc.table.items():
            if not column.strip():
                logger.debug("Skipping empty column key in %s", record.id)
                continue
            year = _parse_year(column)
            if year is None:
                logger.debug("Could not parse year from '%s' in %s", column, record.id)
            valid_at = datetime(year, 1, 1, tzinfo=UTC) if year else reference_time
            year_node = EntityNode(uuid=stable_uuid(column.strip()), name=column.strip(), group_id=group_id, summary=f"Column header: {column.strip()}")
            for metric, value in metrics.items():
                if not metric.strip():
                    continue
                metric_node = EntityNode(uuid=stable_uuid(metric.strip()), name=metric.strip(), group_id=group_id, summary=f"Row label: {metric.strip()}")
                edge = EntityEdge(
                    # Deterministic edge UUID (like the node UUIDs above) so
                    # re-indexing the same cell upserts instead of inserting a
                    # duplicate edge. One cell == one (row, column) pair.
                    uuid=stable_uuid(f"{metric.strip()}|{column.strip()}|HAS_VALUE_IN_COLUMN"),
                    group_id=group_id,
                    source_node_uuid=metric_node.uuid, target_node_uuid=year_node.uuid,
                    name="HAS_VALUE_IN_COLUMN",
                    fact=f"In column [{column}], the row [{metric}] has value {value}",
                    created_at=reference_time, valid_at=valid_at,
                )
                triplets.append((metric_node, edge, year_node))

        for m, e, y in triplets:
            await self._graphiti.add_triplet(m, e, y)

    async def index_bulk(
        self,
        records: list[IndexableRecord],
        concurrency: int = 1,
        checkpoint_path: str | None = None,
    ) -> None:
        """Index records concurrently with checkpoint resume support."""
        semaphore = asyncio.Semaphore(concurrency)
        ckpt = Path(checkpoint_path or settings.index_checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if ckpt.exists():
            try:
                completed: set[str] = set(json.loads(ckpt.read_text()).get("completed", []))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Checkpoint %s is corrupted; starting fresh", ckpt)
                completed = set()
        else:
            completed = set()

        pending = [r for r in records if r.id not in completed]
        initial_done = len(completed)
        total_all = initial_done + len(pending)
        logger.info("=" * 50)
        logger.info("index_bulk starting: %d total | %d done | %d pending", total_all, initial_done, len(pending))
        logger.info("=" * 50)
        done_count = initial_done
        quota_exhausted = False
        lock = asyncio.Lock()

        async def _bounded(record: IndexableRecord) -> None:
            nonlocal quota_exhausted, done_count
            if quota_exhausted:
                return
            async with semaphore:
                if quota_exhausted:  # pragma: no cover
                    return
                for attempt in range(5):
                    if quota_exhausted:
                        return
                    try:
                        await self.index_record(record)
                        async with lock:
                            completed.add(record.id)
                            done_count += 1
                            ckpt.write_text(json.dumps({"completed": list(completed)}))
                            logger.info("[%d/%d] ✓ %s", done_count, total_all, record.id)
                        break
                    except Exception as e:
                        err = str(e)
                        if "insufficient_quota" in err:
                            quota_exhausted = True
                            logger.error("OpenAI quota exhausted stopping. Resume when credits are added.")
                            return
                        if "rate_limit_exceeded" in err or "Rate limit" in err:
                            wait = 60 * (attempt + 1)
                            logger.warning("Rate limit hit for %s waiting %ds before retry %d", record.id, wait, attempt + 1)
                            await asyncio.sleep(wait)
                        elif attempt < 4:
                            logger.warning("Retry %d for %s: %s", attempt + 1, record.id, e)
                        else:
                            logger.error("Gave up on %s after 5 attempts: %s", record.id, e)

        await asyncio.gather(*[_bounded(r) for r in pending])
