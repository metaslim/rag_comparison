"""Answer strategies: uniform execution paths for FinancialAgent.

Structure:
  base/               -- AnswerResult dataclass + AnswerStrategy protocol
  single_pass_graph/  -- SinglePassGraphStrategy (Mode 1: graph-RAG)
  raw_doc/            -- RawDocStrategy           (Mode 2: raw document, no graph)
  tool_calling/       -- ToolCallingStrategy      (Mode 3: function-calling loop)

Adding a new mode: new subfolder + new class. No edits to existing code (OCP).
lib/ has no runtime dependency on convfinqa/; agent objects are passed as Any.
"""

from src.lib.answers.base import AnswerResult, AnswerStrategy
from src.lib.answers.raw_doc import RawDocStrategy
from src.lib.answers.single_pass_graph import SinglePassGraphStrategy
from src.lib.answers.tool_calling import ToolCallingStrategy

__all__ = [
    "AnswerResult",
    "AnswerStrategy",
    "RawDocStrategy",
    "SinglePassGraphStrategy",
    "ToolCallingStrategy",
]
