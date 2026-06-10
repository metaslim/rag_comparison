"""Unit tests for tool functions in src.lib.answers.tool_calling.tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.lib.answers.tool_calling.tools import (
    compare,
    compute,
    search_document,
)

# compute --------------------------------------------------------------------

def test_compute_basic_arithmetic() -> None:
    assert compute("23.4 - 18.7") == pytest.approx(4.7)
    assert compute("20 / 100") == pytest.approx(0.20)
    assert compute("(120 - 100) / 100") == pytest.approx(0.20)


def test_compute_returns_float() -> None:
    assert isinstance(compute("1 + 1"), float)


def test_compute_invalid_expression_raises() -> None:
    with pytest.raises(ValueError, match="Invalid expression"):
        compute("revenue / total")


def test_compute_non_numeric_result_raises() -> None:
    """simpleeval supports string concat e.g. 'a' + 'b'; we reject non-numeric."""
    with pytest.raises(ValueError):
        compute("'a' + 'b'")


# search_document -------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_document_returns_both_result_sets() -> None:
    """search_document runs table and narrative searches in parallel and returns both."""
    mock_graph = MagicMock()
    mock_graph.search_table = AsyncMock(return_value=[
        {"column": "2007", "row": "net income", "value": 100.0},
    ])
    mock_graph.search_narrative = AsyncMock(return_value=[
        {"source": "pre", "snippet": "intro paragraph"},
        {"source": "post", "snippet": "footnote about senior notes"},
    ])
    result = await search_document("net income 2007", mock_graph, "rec-1")
    assert "table_cells" in result
    assert "narrative_snippets" in result
    assert result["table_cells"] == [{"column": "2007", "row": "net income", "value": 100.0}]
    assert len(result["narrative_snippets"]) == 2
    mock_graph.search_table.assert_awaited_once_with("rec-1", "net income 2007")
    mock_graph.search_narrative.assert_awaited_once_with("rec-1", "net income 2007")


@pytest.mark.asyncio
async def test_search_document_uses_same_query_for_both() -> None:
    """Both underlying searches receive the exact same query string."""
    mock_graph = MagicMock()
    mock_graph.search_table = AsyncMock(return_value=[])
    mock_graph.search_narrative = AsyncMock(return_value=[])
    await search_document("senior notes 2015", mock_graph, "rec-2")
    mock_graph.search_table.assert_awaited_once_with("rec-2", "senior notes 2015")
    mock_graph.search_narrative.assert_awaited_once_with("rec-2", "senior notes 2015")


# compare --------------------------------------------------------------------

def test_compare_returns_yes_when_first_greater() -> None:
    assert compare("100", "90") == "yes"


def test_compare_returns_no_when_first_smaller() -> None:
    assert compare("80", "90") == "no"


def test_compare_evaluates_expressions() -> None:
    assert compare("60.94 - 25.14", "25.14") == "yes"  # 35.8 > 25.14


def test_compare_returns_no_when_equal() -> None:
    assert compare("50", "50") == "no"


def test_compare_raises_on_invalid_expression() -> None:
    import pytest
    with pytest.raises(ValueError):
        compare("not_a_number", "50")
