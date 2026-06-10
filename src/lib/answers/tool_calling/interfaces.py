"""Tool-calling protocols: ToolRegistry."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolRegistry(Protocol):
    """Interface for the agent's tool-calling layer.

    Bundles JSON-schema tool definitions and dispatches calls by name.
    Colocated with ``ConvFinQAToolRegistry`` (the concrete implementation).
    """

    def tools_for_openai(self) -> list[dict[str, Any]]:
        """Return the tool schemas in OpenAI function-calling format."""
        ...

    async def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a tool by name with JSON-decoded args; return the result."""
        ...
