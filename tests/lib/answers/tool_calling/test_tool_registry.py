"""Unit tests for ConvFinQAToolRegistry dispatch and serialisation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.lib.answers.tool_calling.tool_registry import ConvFinQAToolRegistry
from src.lib.answers.tool_calling.tools import ALL_TOOL_SCHEMAS

from tests.conftest import make_record


def _record():
    return make_record(
        rid="Single_TEST/2007/page_1.pdf-1",
        table={"2007": {"net income": 100.0}, "2008": {"net income": 120.0}},
    )


@pytest.fixture
def registry() -> ConvFinQAToolRegistry:
    mock_graph = MagicMock()
    mock_graph.search_table = AsyncMock(return_value=[
        {"column": "2007", "row": "net income", "value": 100.0},
    ])
    mock_graph.search_narrative = AsyncMock(return_value=[
        {"source": "pre", "snippet": "intro"},
        {"source": "post", "snippet": "footnote"},
    ])
    return ConvFinQAToolRegistry(_record(), mock_graph)


def test_tools_for_openai_returns_three_schemas(registry) -> None:
    schemas = registry.tools_for_openai()
    assert len(schemas) == 3
    names = {s["function"]["name"] for s in schemas}
    assert names == {"search_document", "compute", "compare"}


def test_tool_schemas_match_canonical_export() -> None:
    """Registry must expose the same canonical schemas defined in tools.py."""
    mock_graph = MagicMock()
    registry = ConvFinQAToolRegistry(_record(), mock_graph)
    assert registry.tools_for_openai() == ALL_TOOL_SCHEMAS


# Dispatch -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_search_document_returns_both_result_sets(registry) -> None:
    result = await registry.dispatch("search_document", {"query": "net income 2007"})
    assert "table_cells" in result
    assert "narrative_snippets" in result
    assert result["table_cells"] == [{"column": "2007", "row": "net income", "value": 100.0}]


@pytest.mark.asyncio
async def test_dispatch_compute_success(registry) -> None:
    result = await registry.dispatch("compute", {"expression": "20 / 100"})
    assert result == 0.2


@pytest.mark.asyncio
async def test_dispatch_compute_invalid_returns_structured_error(registry) -> None:
    result = await registry.dispatch("compute", {"expression": "revenue / total"})
    assert isinstance(result, dict)
    assert result["error"] == "ValueError"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_structured_error(registry) -> None:
    result = await registry.dispatch("does_not_exist", {})
    assert isinstance(result, dict)
    assert result["error"] == "UnknownTool"


@pytest.mark.asyncio
async def test_dispatch_missing_arg_returns_structured_error(registry) -> None:
    result = await registry.dispatch("search_document", {})  # missing 'query'
    assert isinstance(result, dict)
    assert result["error"] == "MissingArg"


@pytest.mark.asyncio
async def test_dispatch_compare_yes(registry) -> None:
    result = await registry.dispatch("compare", {"expr_a": "120", "expr_b": "100"})
    assert result == "yes"


@pytest.mark.asyncio
async def test_dispatch_compare_no(registry) -> None:
    result = await registry.dispatch("compare", {"expr_a": "80", "expr_b": "100"})
    assert result == "no"


@pytest.mark.asyncio
async def test_dispatch_compare_equal_returns_no(registry) -> None:
    result = await registry.dispatch("compare", {"expr_a": "100", "expr_b": "100"})
    assert result == "no"


@pytest.mark.asyncio
async def test_dispatch_compare_invalid_expression_returns_structured_error(registry) -> None:
    result = await registry.dispatch("compare", {"expr_a": "revenue", "expr_b": "100"})
    assert isinstance(result, dict)
    assert result["error"] == "ValueError"
