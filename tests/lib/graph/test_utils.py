"""Unit tests for convfinqa.graph.utils: record_id_to_group_id and _parse_year."""

import pytest

from src.lib.graph.utils import _parse_year, record_id_to_group_id


def test_record_id_to_group_id_strips_suffix() -> None:
    assert record_id_to_group_id("Single_JKHY/2009/page_28.pdf-3") == "Single_JKHY-2009-page_28-pdf"
    assert record_id_to_group_id("Single_JKHY/2009/page_28.pdf-4") == "Single_JKHY-2009-page_28-pdf"
    assert record_id_to_group_id("Single_JKHY/2009/page_28.pdf-1") == "Single_JKHY-2009-page_28-pdf"


def test_record_id_to_group_id_no_suffix_preserved() -> None:
    assert record_id_to_group_id("Double_ETR/2011/page_250.pdf") == "Double_ETR-2011-page_250-pdf"


def test_record_id_to_group_id_higher_suffixes() -> None:
    assert record_id_to_group_id("Single_UNP/2014/page_35.pdf-5") == "Single_UNP-2014-page_35-pdf"
    assert record_id_to_group_id("Single_UNP/2014/page_35.pdf-6") == "Single_UNP-2014-page_35-pdf"


@pytest.mark.parametrize("column,expected", [
    ("2009", 2009),
    ("2008", 2008),
    ("Year ended June 30, 2009", 2009),
    ("year ended december 31 2008 ( unaudited )", 2008),
    ("december 312016", 2016),
    ("$ 2014", None),
    ("$ 1977", None),
    ("1357", None),
    ("total", None),
    ("", None),
    ("12/31/04", None),
])
def test_parse_year(column: str, expected: int | None) -> None:
    assert _parse_year(column) == expected
