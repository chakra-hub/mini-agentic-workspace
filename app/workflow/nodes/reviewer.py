"""Reviewer specialist node and structured review decision parsing."""

import re
from typing import Any

from app.config import EXECUTION_MODE
from app.state import WorkflowState
from app.agents.factory import get_specialist_agent
from app.skills.loader import load_skill_files
from app.demo.mock_outputs import reviewer_output
from app.observability.usage import (
    aggregate_usage, append_usage_history, mock_usage,
)
from app.utils.messages import extract_last_ai_text, print_agent_debug


def reviewer_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Invoking Reviewer Deep Agent")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)
    selection = state["agent_models"]["reviewer"]

    if EXECUTION_MODE == "mock":
        print("[MOCK] Reviewer Deep Agent invocation skipped.")
        decision = reviewer_output(revision_count, max_revisions)
        usage = {
            **mock_usage("reviewer"),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
        return {
            **decision,
            "reviewer_usage": usage,
            "usage_history": append_usage_history(state, "reviewer", usage),
        }

    reviewer_task = f"""
ORIGINAL REQUIREMENTS:
{state['task']}

RESEARCH REPORT:
{state.get('research_report', '')}

DEVELOPER IMPLEMENTATION:
{state.get('implementation', '')}

Review the implementation against the original requirements and acceptance
criteria. Inspect correctness, edge cases, validation, error handling,
security, maintainability, and test quality.

Provide actionable findings with severity and recommended fixes. Do not approve
merely because the response contains code. If actionable correctness, security,
or requirement gaps remain, request changes.

At the very end, include exactly one decision marker on its own line:
REVIEW_STATUS: APPROVED
or
REVIEW_STATUS: CHANGES_REQUESTED
""".strip()

    agent = get_specialist_agent("reviewer", selection["model"])
    result = agent.invoke({
        "messages": [{"role": "user", "content": reviewer_task}],
        "files": load_skill_files(),
    })
    messages = result.get("messages", [])
    print_agent_debug("reviewer", messages)
    report = extract_last_ai_text(messages)
    usage = {
        **aggregate_usage(messages),
        "selected_tier": selection["tier"],
        "selected_model": selection["model"],
    }
    print("\nReviewer usage:", usage)

    if not report:
        raise RuntimeError("Reviewer returned no textual response.")

    markers = re.findall(
        r"^\s*REVIEW_STATUS:\s*(APPROVED|CHANGES_REQUESTED)\s*$",
        report,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    decision = markers[-1].upper() if markers else None
    if decision == "APPROVED":
        status, feedback = "approved", ""
    else:
        # Fail closed: a missing or malformed decision cannot approve the task.
        status, feedback = "changes_requested", report

    print("\n[LangGraph] Reviewer completed:", status)
    return {
        "review_report": report,
        "reviewer_usage": usage,
        "review_status": status,
        "review_feedback": feedback,
        "usage_history": append_usage_history(state, "reviewer", usage),
    }
