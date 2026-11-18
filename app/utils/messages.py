"""Utilities for extracting and inspecting LangChain messages."""

from typing import Any


def extract_text(content: Any) -> str:
    """Extract readable text from common LangChain content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(part for part in parts if part)
    return ""


def extract_last_ai_text(messages: list[Any]) -> str:
    """Return the last non-empty assistant message from an agent run."""
    response_text = ""
    for message in messages:
        if getattr(message, "type", "") == "ai":
            text_content = extract_text(getattr(message, "content", None))
            if text_content.strip():
                response_text = text_content.strip()
    return response_text


def print_agent_debug(agent_name: str, messages: list[Any]) -> None:
    """Print a compact trace useful when diagnosing live agent responses."""
    print(f"\n========== {agent_name.upper()} DEBUG TRACE ==========")
    print("Messages returned:", len(messages))
    for index, message in enumerate(messages, start=1):
        message_type = getattr(message, "type", "unknown")
        content = getattr(message, "content", None)
        metadata = getattr(message, "response_metadata", {}) or {}
        tool_calls = getattr(message, "tool_calls", []) or []
        print(f"\nMessage {index}")
        print("Type:", message_type)
        print("Content:", repr(content)[:500])
        print("Tool calls:", tool_calls)
        print("Finish reason:", metadata.get("finish_reason"))
