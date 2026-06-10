"""Graph protocols: GraphSearcher, GraphIndexer, IndexableRecord."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexableDocument(Protocol):
    """Minimal document structure required for graph indexing."""

    pre_text: str
    post_text: str
    table: dict


@runtime_checkable
class IndexableRecord(Protocol):
    """Minimal record interface required by GraphIndexer.

    ``ConvFinQARecord`` satisfies this at runtime. Keeping it minimal means
    lib/graph/ has no dependency on convfinqa/ domain types.
    """

    id: str
    doc: IndexableDocument


@runtime_checkable
class GraphSearcher(Protocol):
    """Read-only graph interface: search and existence checks.

    The only graph operations ``FinancialAgent`` ever needs. Keeping this
    separate means the agent can be tested with a lightweight stub that has
    no indexing methods at all.
    """

    async def search(self, record_id: str, question: str) -> list[str]:
        """Search for relevant facts scoped to a document (flat list)."""
        ...

    async def search_narrative(self, record_id: str, question: str) -> list[dict]:
        """Search narrative content only (pre_text + post_text), tagged by source."""
        ...

    async def search_table(self, record_id: str, question: str) -> list[dict]:
        """Search table cells only."""
        ...

    async def is_indexed(self, record_id: str) -> bool:
        """Return True if the given record is already indexed in the graph."""
        ...


@runtime_checkable
class GraphIndexer(Protocol):
    """Write-only graph interface: schema setup and document ingestion."""

    async def build_indices(self) -> None:
        """Initialise the graph schema and indices."""
        ...

    async def index_record(self, record: IndexableRecord) -> None:
        """Index a single document record."""
        ...

    async def index_bulk(
        self,
        records: list[IndexableRecord],
        concurrency: int = 1,
        checkpoint_path: str | None = None,
    ) -> None:
        """Index multiple records."""
        ...
