import pytest

from app.model_router import (
    mock_classify_task,
    choose_agent_model_tiers,
    apply_routing_policy,
    RouterAnalysis,
)


@pytest.mark.parametrize(
    "task, expected",
    [
        ("Explain Python decorators with a simple example.", "simple"),
        ("Implement a Python function with pytest tests.", "medium"),
        (
            "Design a production-grade multi-agent RAG architecture.",
            "complex",
        ),
    ],
)
def test_mock_classification(task, expected):
    result = mock_classify_task(task)
    assert result["complexity"] == expected


def test_simple_task_uses_simple_models():
    result = choose_agent_model_tiers("simple")

    assert result == {
        "researcher": "simple",
        "developer": "simple",
        "reviewer": "simple",
    }


def test_complex_task_uses_expensive_model_only_for_developer():
    result = choose_agent_model_tiers("complex")

    assert result == {
        "researcher": "medium",
        "developer": "complex",
        "reviewer": "medium",
    }


def test_invalid_complexity_raises_error():
    with pytest.raises(ValueError):
        choose_agent_model_tiers("unknown")


def test_empty_task_raises_error():
    with pytest.raises(ValueError):
        mock_classify_task("")


def test_low_confidence_escalates_simple_to_medium():
    analysis = RouterAnalysis(
        complexity_score=20,
        task_type="explanation",
        requires_decomposition=False,
        requires_tools=False,
        requires_research=False,
        confidence=0.40,
        rationale="Uncertain classification",
    )

    assert apply_routing_policy(analysis) == "medium"


def test_tools_requirement_promotes_task_to_complex():
    analysis = RouterAnalysis(
        complexity_score=20,
        task_type="technical_task",
        requires_decomposition=False,
        requires_tools=True,
        requires_research=False,
        confidence=0.95,
        rationale="External tools are necessary",
    )

    assert apply_routing_policy(analysis) == "complex"
