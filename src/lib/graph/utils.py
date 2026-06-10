"""Graph utility functions for ConvFinQA document handling."""

import re


def record_id_to_group_id(record_id: str) -> str:
    """Map a record ID to a document-level Graphiti group_id.

    Strips the trailing question number suffix (e.g. '-1', '-3') so that multiple
    records from the same PDF page share one group_id and are indexed only once.
    E.g. 'Single_JKHY/2009/page_28.pdf-3' → 'Single_JKHY-2009-page_28-pdf'
    Graphiti validates group_id against [a-zA-Z0-9_-].
    """
    doc_id = re.sub(r"-\d+$", "", record_id)
    return re.sub(r"[^a-zA-Z0-9_-]", "-", doc_id)


def _parse_year(column: str) -> int | None:
    """Extract a 4-digit year (1970-2039) from a table column key.

    Needed for triplet valid_at timestamps: column keys like 'Year ended June 30, 2009'
    must be parsed to set the correct temporal context on each edge.
    Returns None for non-year columns (e.g. 'total', '$ 2014').
    """
    col = column.strip()
    if col.startswith("$"):
        return None
    m = re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2}(\d{4})", col, re.I)
    if m:
        yr = int(m.group(1))
        return yr if 1970 <= yr <= 2039 else None
    m = re.search(r"(?<!\d)(19[7-9]\d|20[0-3]\d)(?!\d)", col)
    return int(m.group()) if m else None
