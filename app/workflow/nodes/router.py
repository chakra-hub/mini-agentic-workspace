"""LangGraph node responsible for project-level model routing."""

from typing import Any

from app.config import EXECUTION_MODE, MODEL_REGISTRY
from app.state import WorkflowState

from app.model_router.classifier import classify_and_route
from app.model_router.policy import (
    choose_agent_model_tiers,
    mock_classify_task,
)


def normalize_router_usage(raw_usage: dict[str, Any]) -> dict[str, Any]:
    """Normalize classifier usage into the application usage schema."""

    input_tokens = (
        raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)) or 0
    )
    output_tokens = (
        raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)) or 0
    )
    total_tokens = (
        raw_usage.get("total_tokens", input_tokens + output_tokens) or 0
    )

    return {
        "source": "openrouter_actual",
        "agent": "model_router_classifier",
        "tokens_measured": bool(raw_usage),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "api_cost_usd": None,
    }


def smart_router_node(state: WorkflowState) -> dict[str, Any]:
    """Classify the incoming task and choose models for each specialist."""

    task = state.get("task", "").strip()

    if not task:
        raise ValueError("Task cannot be empty before model routing.")

    print("\n[LangGraph] Executing Smart Model Router")

    if EXECUTION_MODE == "mock":
        routing = mock_classify_task(task)

        router_usage = {
            "source": "mock",
            "agent": "model_router_classifier",
            "tokens_measured": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "api_cost_usd": 0.0,
        }

        routing_source = "mock_rule_based"

    else:
        routing = classify_and_route(task)

        router_usage = normalize_router_usage(
            routing.get("classifier_usage", {})
        )

        routing_source = "openrouter_classifier"

    # Apply the agent-level model selection policy.
    selected_tiers = choose_agent_model_tiers(routing["complexity"])

    agent_models = {
        agent_name: {
            "tier": tier,
            "model": MODEL_REGISTRY[tier],
        }
        for agent_name, tier in selected_tiers.items()
    }

    routing_summary = {
        "source": routing_source,
        "complexity": routing["complexity"],
        "complexity_score": routing.get("complexity_score"),
        "confidence": routing.get("confidence"),
        "task_type": routing.get("task_type"),
        "requires_decomposition": routing.get("requires_decomposition"),
        "requires_tools": routing.get("requires_tools"),
        "requires_research": routing.get("requires_research"),
        "rationale": routing.get("rationale", ""),
        "classifier_model": routing.get("classifier_model"),
        "selected_model": routing.get("selected_model"),
        "agent_models": agent_models,
    }

    print("[Smart Router] Project complexity:", routing_summary["complexity"])
    print("[Smart Router] Classification source:", routing_source)

    for agent_name, selection in agent_models.items():
        print(
            f"[Smart Router] {agent_name}: "
            f"{selection['tier']} -> {selection['model']}"
        )

    history = list(state.get("usage_history", []))
    history.append({
        "agent": "model_router_classifier",
        **router_usage,
    })

    return {
        "model_routing": routing_summary,
        "agent_models": agent_models,
        "router_usage": router_usage,
        "usage_history": history,
    }
