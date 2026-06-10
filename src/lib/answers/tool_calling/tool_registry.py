"""ConvFinQAToolRegistry: binds tool functions to a record + graph for dispatch."""

import asyncio
from collections.abc import Callable
from typing import Any

from src.lib.answers.base.strategy import IndexedRecord
from src.lib.answers.tool_calling.tools import (
    ALL_TOOL_SCHEMAS,
    compare,
    compute,
    search_document,
)
from src.lib.graph.interfaces import GraphSearcher


class ConvFinQAToolRegistry:
    """Dispatcher binding the three tools to one record + one graph.

    Open/Closed: extend the tool surface by adding to ``_handlers``; the
    ``dispatch`` method itself is closed for modification.
    """

    def __init__(self, record: IndexedRecord, graph: GraphSearcher) -> None:
        self._record = record
        self._graph = graph
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "search_document": lambda a: search_document(
                a["query"], self._graph, self._record.id
            ),
            "compute": lambda a: compute(a["expression"]),
            "compare": lambda a: compare(a["expr_a"], a["expr_b"]),
        }

    def tools_for_openai(self) -> list[dict[str, Any]]:
        """Return tool schemas in OpenAI function-calling format."""
        return ALL_TOOL_SCHEMAS

    async def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a tool by name. Returns a JSON-serialisable result.

        Errors the model can recover from (bad expression, unknown tool,
        missing arg) are returned as ``{"error": ..., "message": ...}`` so the
        loop continues. Other exceptions propagate to the agent.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": "UnknownTool", "message": f"No such tool: {name!r}"}
        try:
            result = handler(args)
            if asyncio.iscoroutine(result):
                result = await result
        except KeyError as e:
            return {"error": "MissingArg", "message": f"Missing argument: {e}"}
        except ValueError as e:
            return {"error": "ValueError", "message": str(e)}
        return result
