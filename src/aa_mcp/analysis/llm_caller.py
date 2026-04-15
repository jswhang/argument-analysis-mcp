"""LLMCaller protocol and MCP adapter factory.

In standalone (non-MCP) use, supply any async callable with the signature::

    async def my_llm(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        ...

For the MCP server, use ``mcp_session_caller(session)`` to wrap the active
ServerSession into the same interface.
"""

from __future__ import annotations

from typing import Awaitable, Callable

# The sole interface between SamplingHelper and the underlying LLM.
# (prompt, system_prompt, max_tokens, temperature) -> response text
LLMCaller = Callable[[str, str, int, float], Awaitable[str]]


def mcp_session_caller(session) -> LLMCaller:
    """Wrap an MCP ServerSession as an LLMCaller."""
    from mcp.types import SamplingMessage, TextContent

    async def _call(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        result = await session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
            max_tokens=max_tokens,
            system_prompt=system,
        )
        if hasattr(result.content, "text"):
            return result.content.text
        return str(result.content)

    return _call


def has_mcp_sampling(session) -> bool:
    """Return True if the MCP ServerSession supports sampling/createMessage."""
    try:
        caps = session.client_params.capabilities
        return caps is not None and caps.sampling is not None
    except AttributeError:
        return False
