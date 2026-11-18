"""Deterministic AIDLC planning node."""

from typing import Any

from app.state import WorkflowState


def planner_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Executing Planner")
    task = state.get("task", "").strip()
    if not task:
        raise ValueError("Task cannot be empty.")

    routing = state.get("model_routing", {})
    models = state.get("agent_models", {})
    plan = f"""
PROJECT EXECUTION PLAN

Original requirement:
{task}

1. RESEARCH PHASE
   Agent: Researcher
   - Identify core technical concepts.
   - Evaluate relevant approaches.
   - Document trade-offs and limitations.

2. DEVELOPMENT PHASE
   Agent: Developer
   - Translate requirements into implementation.
   - Follow software development practices.
   - Implement error handling and tests.

3. REVIEW PHASE
   Agent: Reviewer
   - Validate implementation against requirements.
   - Check correctness, security, and maintainability.
   - Evaluate test coverage.
   - Report actionable findings.

ACCEPTANCE CRITERIA
- Requirements are addressed.
- Implementation includes type hints.
- Input validation and error handling are present.
- Relevant unit tests are provided.
- Review findings and final review status are documented.

MODEL ROUTING
- Classification source: {routing.get('source', 'not available')}
- Project complexity: {routing.get('complexity', 'not available')}
- Researcher model: {models.get('researcher', {}).get('model', 'not available')}
- Developer model: {models.get('developer', {}).get('model', 'not available')}
- Reviewer model: {models.get('reviewer', {}).get('model', 'not available')}
- At most {state.get('max_revisions', 1)} developer revision(s) are allowed.
""".strip()
    print("[Planner] Execution plan created.")
    return {"plan": plan}
