"""Command-line entry point for the AIDLC demonstration."""

from app.config import EXECUTION_MODE
from app.skills.loader import load_skill_files
from app.workflow.graph import build_workflow
from app.observability.usage import summarize_usage_history


def main() -> None:
    task = """
Design a production-grade multi-agent RAG platform.

Requirements:
1. Multi-tenant document ingestion and authorization.
2. Hybrid retrieval using BM25 and vector search.
3. Cross-encoder reranking.
4. Multi-agent orchestration using LangGraph.
5. FastAPI backend with streaming responses.
6. Evaluation, tracing and observability.
7. Deployment architecture and scalability.
8. Python implementation and unit tests.
""".strip()

    print(f"Execution mode: {EXECUTION_MODE.upper()}")
    skill_files = load_skill_files()
    print("\n========== AVAILABLE SKILLS ==========")
    for skill_path in skill_files:
        print(f" - {skill_path}")

    workflow = build_workflow()
    print("\n========== LANGGRAPH STRUCTURE ==========\n")
    print(workflow.get_graph().draw_mermaid())
    print("\n========== EXECUTION ==========\n")

    result = workflow.invoke({
        "task": task,
        "research_task": "",
        "revision_count": 0,
        "max_revisions": 1,
        "usage_history": [],
    })

    print("\n========== MODEL ROUTING DECISION ==========\n")
    print(result.get("model_routing", {}))
    print("\n========== PER-AGENT MODEL SELECTION ==========\n")
    for name, selection in result.get("agent_models", {}).items():
        print(f"{name}: {selection['tier']} -> {selection['model']}")

    sections = [
        ("EXECUTION PLAN", "plan"),
        ("RESEARCH REPORT", "research_report"),
        ("RESEARCHER TOKEN USAGE", "researcher_usage"),
        ("DEVELOPER IMPLEMENTATION", "implementation"),
        ("DEVELOPER TOKEN USAGE (LATEST INVOCATION)", "developer_usage"),
        ("REVIEW REPORT", "review_report"),
        ("REVIEWER TOKEN USAGE (LATEST INVOCATION)", "reviewer_usage"),
        ("FINAL REVIEW FEEDBACK", "review_feedback"),
    ]
    for title, key in sections:
        print(f"\n========== {title} ==========\n")
        print(result.get(key, ""))

    print("\n========== FINAL REVIEW STATUS ==========\n")
    print(result.get("review_status", "unknown"))
    print("Revision count:", result.get("revision_count", 0))
    print("Maximum revisions:", result.get("max_revisions", 1))

    print("\n========== AGENT USAGE HISTORY ==========\n")
    for item in result.get("usage_history", []):
        print(item)
    print("\n========== USAGE SUMMARY ==========\n")
    print(summarize_usage_history(result.get("usage_history", [])))


if __name__ == "__main__":
    main()
