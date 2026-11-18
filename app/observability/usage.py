"""Usage normalization, history, and reporting helpers."""

from typing import Any

from app.state import WorkflowState


def aggregate_usage(messages: list[Any]) -> dict[str, Any]:
    """Aggregate token usage reported by LangChain/OpenRouter responses."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_was_reported = False

    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            response_metadata = getattr(message, "response_metadata", {}) or {}
            usage = response_metadata.get("token_usage", {}) or {}
        if usage:
            usage_was_reported = True
        for key in totals:
            value = usage.get(key, 0)
            if isinstance(value, (int, float)):
                totals[key] += int(value)

    return {
        "source": "openrouter_actual",
        "tokens_measured": usage_was_reported,
        **totals,
        # Billed USD cost is not inferred from token counts.
        "api_cost_usd": None,
    }


def mock_usage(agent_name: str) -> dict[str, Any]:
    """Return explicitly labelled usage for a simulated execution."""
    return {
        "source": "mock",
        "agent": agent_name,
        "tokens_measured": False,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "api_cost_usd": 0.0,
    }


def append_usage_history(
    state: WorkflowState,
    agent_name: str,
    usage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append one invocation while preserving prior revision records."""
    history = list(state.get("usage_history", []))
    history.append({"agent": agent_name, **usage})
    return history


def summarize_usage_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize classifier and specialist usage without inventing cost."""
    classifier_records = [
        item for item in history if item.get("agent") == "model_router_classifier"
    ]
    specialist_records = [
        item for item in history
        if item.get("agent") in {"researcher", "developer", "reviewer"}
    ]
    actual_records = [item for item in history if item.get("source") == "openrouter_actual"]
    mock_records = [item for item in history if item.get("source") == "mock"]
    measured_records = [item for item in actual_records if item.get("tokens_measured")]

    return {
        "agent_invocations_total": len(specialist_records),
        "openrouter_agent_invocations": sum(
            item.get("source") == "openrouter_actual" for item in specialist_records
        ),
        "mock_agent_invocations": sum(
            item.get("source") == "mock" for item in specialist_records
        ),
        "classifier_invocations_total": len(classifier_records),
        "openrouter_classifier_invocations": sum(
            item.get("source") == "openrouter_actual" for item in classifier_records
        ),
        "mock_classifier_invocations": sum(
            item.get("source") == "mock" for item in classifier_records
        ),
        "openrouter_calls_total": len(actual_records),
        "mock_calls_total": len(mock_records),
        "measured_input_tokens": sum(
            item.get("input_tokens", 0) or 0 for item in measured_records
        ),
        "measured_output_tokens": sum(
            item.get("output_tokens", 0) or 0 for item in measured_records
        ),
        "measured_total_tokens": sum(
            item.get("total_tokens", 0) or 0 for item in measured_records
        ),
        "api_cost_usd": 0.0 if not actual_records else None,
        "cost_note": (
            "No LLM API calls were made in mock mode."
            if not actual_records
            else "Token usage is recorded; billed USD cost is not available "
                 "from this usage metadata."
        ),
    }
