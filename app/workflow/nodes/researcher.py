"""Researcher specialist node."""

from typing import Any

from app.config import EXECUTION_MODE
from app.state import WorkflowState
from app.agents.factory import get_specialist_agent
from app.skills.loader import load_skill_files
from app.demo.mock_outputs import research_output
from app.observability.usage import (
    aggregate_usage, append_usage_history, mock_usage,
)
from app.utils.messages import extract_last_ai_text, print_agent_debug


def researcher_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Invoking Researcher Deep Agent")
    selection = state["agent_models"]["researcher"]

    if EXECUTION_MODE == "mock":
        print("[MOCK] Researcher Deep Agent invocation skipped.")
        report = research_output(state.get("task", ""))
        usage = {
            **mock_usage("researcher"),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
    else:
        research_task = state.get("research_task") or state["task"]
        agent = get_specialist_agent("researcher", selection["model"])
        result = agent.invoke({
            "messages": [{"role": "user", "content": research_task}],
            "files": load_skill_files(),
        })
        messages = result.get("messages", [])
        print_agent_debug("researcher", messages)
        report = extract_last_ai_text(messages)
        usage = {
            **aggregate_usage(messages),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
        print("\nResearcher usage:", usage)
        if not report:
            raise RuntimeError("Researcher returned no textual response.")

    print("\n[LangGraph] Researcher completed")
    return {
        "research_report": report,
        "researcher_usage": usage,
        "usage_history": append_usage_history(state, "researcher", usage),
    }
