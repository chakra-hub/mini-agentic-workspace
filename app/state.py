"""Shared LangGraph workflow state."""

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    # Original user request.
    task: str
    research_task: str

    # AIDLC planning.
    plan: str

    # Smart model routing.
    model_routing: dict[str, Any]
    agent_models: dict[str, dict[str, str]]
    router_usage: dict[str, Any]

    # Agent outputs.
    research_report: str
    implementation: str
    review_report: str

    # Latest agent usage.
    researcher_usage: dict[str, Any]
    developer_usage: dict[str, Any]
    reviewer_usage: dict[str, Any]

    # Historical usage, including revision calls.
    usage_history: list[dict[str, Any]]

    # Review and revision management.
    review_status: str
    review_feedback: str
    revision_count: int
    max_revisions: int
