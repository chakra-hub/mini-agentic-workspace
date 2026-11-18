"""Model routing policies and deterministic mock classification."""

from typing import Any

from app.config import MODEL_REGISTRY


def apply_routing_policy(analysis: Any) -> str:
    """Select a complexity tier from the classifier's analysis."""

    score = analysis.complexity_score

    if (
        score >= 70
        or analysis.requires_decomposition
        or analysis.requires_tools
    ):
        tier = "complex"

    elif score >= 35 or analysis.requires_research:
        tier = "medium"

    else:
        tier = "simple"

    # Escalate uncertain classifications to avoid underpowered models.
    if analysis.confidence < 0.65:
        if tier == "simple":
            tier = "medium"
        elif tier == "medium":
            tier = "complex"

    return tier


def choose_agent_model_tiers(complexity: str) -> dict[str, str]:
    """Apply role-aware model budgets after project classification."""

    routing_policy = {
        "simple": {
            "researcher": "simple",
            "developer": "simple",
            "reviewer": "simple",
        },
        "medium": {
            "researcher": "medium",
            "developer": "medium",
            "reviewer": "medium",
        },
        "complex": {
            "researcher": "medium",
            "developer": "complex",
            "reviewer": "medium",
        },
    }

    if complexity not in routing_policy:
        raise ValueError(f"Unsupported routing complexity: {complexity}")

    return routing_policy[complexity].copy()


def mock_classify_task(task: str) -> dict[str, Any]:
    """Simulate classification without making an LLM API request."""

    if not task or not task.strip():
        raise ValueError("Task cannot be empty.")

    normalized = task.lower()

    complex_terms = (
        "multi-agent",
        "multi agent",
        "system architecture",
        "distributed",
        "decompose",
        "orchestration",
        "enterprise architecture",
        "production-grade",
        "production grade",
    )

    medium_terms = (
        "implement",
        "implementation",
        "write code",
        "write a function",
        "create a function",
        "debug",
        "pytest",
        "unit test",
        "compare",
        "trade-off",
        "refactor",
        "integrate",
    )

    if any(term in normalized for term in complex_terms):
        score = 85
        tier = "complex"
        task_type = "architecture_or_multi_stage_engineering"
        rationale = "The task contains architecture, orchestration, or decomposition signals."

    elif any(term in normalized for term in medium_terms):
        score = 52
        tier = "medium"
        task_type = "technical_implementation"
        rationale = "The task requires moderate technical reasoning or implementation."

    else:
        score = 20
        tier = "simple"
        task_type = "explanation"
        rationale = "The task appears to be a straightforward single-step request."

    return {
        "task": task,
        "complexity": tier,
        "complexity_score": score,
        "confidence": 1.0,
        "task_type": task_type,
        "requires_decomposition": tier == "complex",
        "requires_tools": False,
        "requires_research": any(
            term in normalized
            for term in ("research", "compare", "evaluate alternatives")
        ),
        "rationale": rationale,
        "classifier_model": None,
        "selected_model": MODEL_REGISTRY[tier],
        "source": "mock_rule_based",
    }
