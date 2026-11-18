"""Build and compile the parent AIDLC LangGraph workflow."""

from app.state import WorkflowState
from app.workflow.nodes.router import smart_router_node
from app.workflow.nodes.planner import planner_node
from app.workflow.nodes.researcher import researcher_node
from app.workflow.nodes.developer import developer_node
from app.workflow.nodes.reviewer import reviewer_node
from langgraph.graph import StateGraph, START, END


def route_after_review(state: WorkflowState) -> str:
    """Route approved work to END or send feedback for a bounded revision."""
    review_status = state.get("review_status", "changes_requested")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)

    if review_status == "approved":
        print("[AIDLC] Review approved. Workflow completed.")
        return "end"
    if revision_count < max_revisions:
        print("[AIDLC] Changes requested. Returning to Developer.")
        return "developer"
    print("[AIDLC] Revision limit reached. Ending with unresolved findings.")
    return "end"


def build_workflow():
    """Create the AIDLC graph. Calling this function performs no LLM calls."""
    builder = StateGraph(WorkflowState)
    builder.add_node("smart_router", smart_router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("developer", developer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "smart_router")
    builder.add_edge("smart_router", "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "developer")
    builder.add_edge("developer", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"developer": "developer", "end": END},
    )
    return builder.compile()
