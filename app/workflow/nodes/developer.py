"""Developer specialist node, including bounded review revisions."""

from typing import Any

from app.config import EXECUTION_MODE
from app.state import WorkflowState
from app.agents.factory import get_specialist_agent
from app.skills.loader import load_skill_files
from app.demo.mock_outputs import developer_output
from app.observability.usage import (
    aggregate_usage, append_usage_history, mock_usage,
)
from app.utils.messages import extract_last_ai_text, print_agent_debug


def developer_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Invoking Developer Deep Agent")
    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")
    next_revision_count = revision_count + (1 if review_feedback else 0)
    selection = state["agent_models"]["developer"]

    if EXECUTION_MODE == "mock":
        print("[MOCK] Developer Deep Agent invocation skipped.")
        implementation = developer_output(review_feedback)
        usage = {
            **mock_usage("developer"),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
    else:
        developer_task = f"""
ORIGINAL REQUIREMENTS:
{state['task']}

RESEARCH REPORT:
{state.get('research_report', '')}

PREVIOUS IMPLEMENTATION:
{state.get('implementation') or 'No previous implementation. This is the initial implementation.'}

PREVIOUS REVIEW FEEDBACK:
{review_feedback or 'No previous review feedback. This is the initial implementation.'}

Implement the requested functionality. Include Python code, type hints, input
validation, error handling, pytest tests, and a concise explanation.

If review feedback is provided, revise the previous implementation to address
all actionable findings while preserving functionality that already works.
Return the revised implementation and tests, not merely a revision summary.

Keep the implementation focused and concise. Do not repeat the research report.
""".strip()
        agent = get_specialist_agent("developer", selection["model"])
        result = agent.invoke({
            "messages": [{"role": "user", "content": developer_task}],
            "files": load_skill_files(),
        })
        messages = result.get("messages", [])
        print_agent_debug("developer", messages)
        implementation = extract_last_ai_text(messages)
        usage = {
            **aggregate_usage(messages),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
        print("\nDeveloper usage:", usage)
        if not implementation:
            raise RuntimeError("Developer returned no textual implementation.")

    print("\n[LangGraph] Developer completed")
    return {
        "implementation": implementation,
        "developer_usage": usage,
        "revision_count": next_revision_count,
        "usage_history": append_usage_history(state, "developer", usage),
    }
