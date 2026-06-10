"""FinancialGraph: facade over _GraphSearcherImpl and _GraphIndexerImpl."""

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient as _BaseOpenAIClient
from graphiti_core.search.search_config import SearchConfig

from src.config import settings
from src.lib.graph.index import _GraphIndexerImpl
from src.lib.graph.interfaces import IndexableRecord
from src.lib.graph.search import _GraphSearcherImpl
from src.lib.util import get_logger

logger = get_logger(__name__)


class OpenAIClient(_BaseOpenAIClient):
    """OpenAI client with increased retries for malformed JSON responses.

    Graphiti's default MAX_RETRIES=2. For financial documents with long
    pre_text, the LLM occasionally returns truncated JSON. On retry, it
    receives the error context and corrects itself 5 attempts is enough.
    """

    MAX_RETRIES: int = 5


class _FinancialGraphiti(Graphiti):
    """Graphiti subclass that exposes ``_search`` as a stable public method.

    Graphiti's ``_search`` is a private API with no deprecation guarantees.
    Subclassing isolates that coupling here: if the method is renamed or
    removed in a future Graphiti release, it fails loudly in one place rather
    than silently across multiple call sites.

    Same pattern as ``OpenAIClient`` above -- one override, inherit everything.
    """

    async def search_with_config(
        self,
        query: str,
        group_ids: list[str],
        config: SearchConfig,
    ):
        """Public wrapper around ``_search`` with a stable name."""
        return await self._search(query=query, group_ids=group_ids, config=config)


class FinancialGraph:
    """Thin facade over ``_GraphSearcherImpl`` and ``_GraphIndexerImpl``.

    Creates the shared Graphiti client and ``_indexed_ids`` set, then wires
    both impl classes together. Each change reason now lives in its own module:
    search config / ranking  ->  lib/graph/search.py
    indexing / chunking / schema  ->  lib/graph/index.py
    """

    def __init__(self, graphiti: Graphiti | None = None) -> None:
        if graphiti is None:
            driver = Neo4jDriver(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
            )
            llm_client = OpenAIClient(
                config=LLMConfig(
                    api_key=settings.openai_api_key,
                    model=settings.graphiti_model,
                    max_tokens=32768,
                )
            )
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=settings.openai_api_key,
                    embedding_model=settings.openai_embedding_model,
                )
            )
            graphiti = _FinancialGraphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder)
        self._graphiti = graphiti
        indexed_ids: set[str] = set()
        self._searcher = _GraphSearcherImpl(self._graphiti, indexed_ids)
        self._indexer = _GraphIndexerImpl(self._graphiti, indexed_ids)

    # ---- GraphSearcher delegation ----------------------------------------

    async def is_indexed(self, record_id: str) -> bool:
        """Delegate to _GraphSearcherImpl."""
        return await self._searcher.is_indexed(record_id)

    async def search(self, record_id: str, question: str) -> list[str]:
        """Delegate to _GraphSearcherImpl."""
        return await self._searcher.search(record_id, question)

    async def search_narrative(self, record_id: str, question: str) -> list[dict]:
        """Delegate to _GraphSearcherImpl."""
        return await self._searcher.search_narrative(record_id, question)

    async def search_table(self, record_id: str, question: str) -> list[dict]:
        """Delegate to _GraphSearcherImpl."""
        return await self._searcher.search_table(record_id, question)

    # ---- GraphIndexer delegation ----------------------------------------

    async def build_indices(self) -> None:
        """Delegate to _GraphIndexerImpl."""
        await self._indexer.build_indices()

    async def index_record(self, record: IndexableRecord) -> None:
        """Delegate to _GraphIndexerImpl."""
        await self._indexer.index_record(record)

    async def index_bulk(
        self,
        records: list[IndexableRecord],
        concurrency: int = 1,
        checkpoint_path: str | None = None,
    ) -> None:
        """Delegate to _GraphIndexerImpl."""
        await self._indexer.index_bulk(records, concurrency=concurrency, checkpoint_path=checkpoint_path)

    # ---- Infrastructure --------------------------------------------------

    async def is_reachable(self) -> bool:
        """Return True if the Neo4j backend is reachable (used by CLI health check)."""
        try:
            await self._graphiti.driver.execute_query("RETURN 1")
            return True
        except Exception as e:
            logger.debug("Neo4j health check failed: %s", e)
            return False

    async def close(self) -> None:
        """Close the Neo4j connection."""
        await self._graphiti.close()
