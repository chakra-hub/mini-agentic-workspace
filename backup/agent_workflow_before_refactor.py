"""AIDLC-style multi-agent workflow using LangGraph and Deep Agents.

Execution modes:
    EXECUTION_MODE=mock  -> deterministic offline demonstration; no LLM calls
    EXECUTION_MODE=live  -> actual Deep Agent inference through OpenRouter

The mock outputs are explicitly labelled as simulated. They must not be
presented as outputs produced by an LLM.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepagents import (
    create_deep_agent,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. Environment configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "mock").strip().lower()

if EXECUTION_MODE not in {"mock", "live"}:
    raise ValueError("EXECUTION_MODE must be either 'mock' or 'live'.")

API_KEY = os.getenv("OPENROUTER_API_KEY")
CLASSIFIER_MODEL = os.getenv("OPENROUTER_CLASSIFIER_MODEL")
MODEL_REGISTRY = {
    "simple": os.getenv("OPENROUTER_SIMPLE_MODEL"),
    "medium": os.getenv("OPENROUTER_MEDIUM_MODEL"),
    "complex": os.getenv("OPENROUTER_COMPLEX_MODEL"),
}

# Mock mode must run without credentials or configured model IDs. Placeholders
# are labels only and are never passed to an LLM in mock execution.
if EXECUTION_MODE == "live":
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing from .env.")
    if not CLASSIFIER_MODEL:
        raise ValueError("OPENROUTER_CLASSIFIER_MODEL is missing from .env.")
    missing_models = [tier for tier, model in MODEL_REGISTRY.items() if not model]
    if missing_models:
        raise ValueError(
            "Missing model IDs in .env for tiers: " + ", ".join(missing_models)
        )
else:
    API_KEY = API_KEY or "mock-mode-no-network-key"
    MODEL_REGISTRY = {
        tier: model or f"mock-{tier}-model-not-used"
        for tier, model in MODEL_REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# 2. Deep Agents harness configuration
# ---------------------------------------------------------------------------

# The outer LangGraph controls specialist routing. Disable automatic
# general-purpose subagents and code execution to keep orchestration explicit.
register_harness_profile(
    "openai",
    HarnessProfile(
        excluded_tools=frozenset({"execute"}),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)


# ---------------------------------------------------------------------------
# 3. Specialist agent definitions and model factory
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPTS = {
    "researcher": """
You are a technical research specialist.

Your responsibility is to investigate the assigned technical topic and
produce a structured, accurate report.

Guidelines:
1. Explain the underlying technical concepts.
2. Include formulas and numerical examples when useful.
3. Explain advantages, limitations, and trade-offs.
4. Focus on production engineering considerations.
5. Avoid unnecessary repetition.

Use the research skill at /skills/research/SKILL.md when relevant.

Do not delegate work to other agents.
Do not execute shell commands or code.
Do not modify project files.
Return your research report directly.
""",
    "developer": """
You are a senior software development specialist.

You receive:
- The original user requirements.
- A technical research report.
- Previous implementation and reviewer feedback, when revising.

Your responsibilities:
1. Understand and satisfy the implementation requirements.
2. Use the research report as technical context.
3. Produce clean, maintainable Python code.
4. Include type hints and appropriate input validation.
5. Include relevant error handling.
6. Provide pytest unit tests where relevant.
7. Explain important implementation decisions.

Use the software-development skill at
/skills/software-development/SKILL.md when relevant.

When reviewer feedback is provided, revise the previous implementation to
address the findings while preserving functionality that already works.

Do not delegate work.
Do not execute shell commands or code.
Do not modify project files.
Return the implementation directly in your response.
""",
    "reviewer": """
You are a senior software code reviewer.

You receive:
- Original user requirements.
- Technical research report.
- Developer implementation.

Your responsibilities:
1. Review the implementation against the requirements.
2. Identify logical errors and edge cases.
3. Check input validation and exception handling.
4. Identify security vulnerabilities.
5. Evaluate performance and maintainability.
6. Review the quality and coverage of unit tests.
7. Identify missing test scenarios.

Use the code-review skill at /skills/code-review/SKILL.md when relevant.

Output format:
- Findings
- Severity (Critical, High, Medium, Low)
- Explanation
- Recommended fixes
- Overall assessment

At the very end of your response, include exactly one decision marker:
REVIEW_STATUS: APPROVED
or
REVIEW_STATUS: CHANGES_REQUESTED

Approve only when the implementation adequately satisfies the requirements and
has no unresolved critical or high-severity issues. Request changes when there
are actionable correctness, security, or requirement gaps. If changes are
requested, include actionable feedback for the developer.

Do not delegate work.
Do not execute shell commands or code.
Do not modify project files.
Do not rewrite the entire implementation.
""",
}

AGENT_MAX_TOKENS = {
    "researcher": 4096,
    "developer": 4096,
    "reviewer": 3072,
}


@lru_cache(maxsize=12)
def get_specialist_agent(agent_name: str, model_name: str):
    """Build and cache a specialist Deep Agent for the selected model ID."""

    if agent_name not in AGENT_SYSTEM_PROMPTS:
        raise ValueError(f"Unknown specialist agent: {agent_name}")
    if not model_name:
        raise ValueError(f"No model configured for specialist: {agent_name}")

    llm = ChatOpenAI(
        model=model_name,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=AGENT_MAX_TOKENS[agent_name],
    )

    return create_deep_agent(
        model=llm,
        backend=StateBackend(),
        skills=["/skills/"],
        name=agent_name,
        system_prompt=AGENT_SYSTEM_PROMPTS[agent_name],
    )


# ---------------------------------------------------------------------------
# 4. Load project skills into StateBackend-compatible file data
# ---------------------------------------------------------------------------

def load_skill_files() -> dict[str, Any]:
    """Load every skills/*/SKILL.md file into virtual backend paths."""

    skills_dir = BASE_DIR / "skills"
    skill_files: dict[str, Any] = {}

    if not skills_dir.exists():
        raise FileNotFoundError(f"Skills directory not found: {skills_dir}")

    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        virtual_path = "/" + skill_path.relative_to(BASE_DIR).as_posix()
        content = skill_path.read_text(encoding="utf-8")
        skill_files[virtual_path] = create_file_data(content)

    if not skill_files:
        raise FileNotFoundError(
            f"No SKILL.md files found inside {skills_dir}."
        )

    return skill_files


# ---------------------------------------------------------------------------
# 5. Specialist Deep Agents are created on demand
# ---------------------------------------------------------------------------

# get_specialist_agent(agent_name, model_name) above creates the selected
# specialist lazily and caches it. This allows the smart router to choose the
# model before a Deep Agent is instantiated.
print("[Setup] Specialist Deep Agent factory configured.")


# ---------------------------------------------------------------------------
# 6. Define the parent LangGraph state
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict, total=False):
    task: str
    research_task: str
    plan: str

    model_routing: dict[str, Any]
    agent_models: dict[str, dict[str, str]]
    router_usage: dict[str, Any]

    research_report: str
    implementation: str
    review_report: str

    researcher_usage: dict[str, Any]
    developer_usage: dict[str, Any]
    reviewer_usage: dict[str, Any]

    # One record per specialist-node invocation, including revisions.
    usage_history: list[dict[str, Any]]

    review_status: str
    review_feedback: str
    revision_count: int
    max_revisions: int


# ---------------------------------------------------------------------------
# 7. Helper functions
# ---------------------------------------------------------------------------

def extract_text(content: Any) -> str:
    """Extract readable text from common LangChain message content shapes."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))

        return "\n".join(part for part in parts if part)

    return ""


def aggregate_usage(messages: list[Any]) -> dict[str, Any]:
    """Aggregate token usage reported by LangChain/OpenRouter responses."""

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    usage_was_reported = False

    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}

        # Compatibility fallback for responses exposing token_usage metadata.
        if not usage:
            response_metadata = getattr(message, "response_metadata", {}) or {}
            usage = response_metadata.get("token_usage", {}) or {}

        if usage:
            usage_was_reported = True

        for key in totals:
            value = usage.get(key, 0)
            if isinstance(value, (int, float)):
                totals[key] += int(value)

    return {
        "source": "openrouter_actual",
        "tokens_measured": usage_was_reported,
        **totals,
        # The standard ChatOpenAI usage metadata does not reliably provide
        # the billed dollar cost. Do not invent a price estimate here.
        "api_cost_usd": None,
    }


def mock_usage(agent_name: str) -> dict[str, Any]:
    """Return explicitly labelled usage for a simulated node execution."""

    return {
        "source": "mock",
        "agent": agent_name,
        "tokens_measured": False,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "api_cost_usd": 0.0,
    }


def append_usage_history(
    state: WorkflowState,
    agent_name: str,
    usage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append this node invocation without losing earlier revision records."""

    history = list(state.get("usage_history", []))
    history.append({"agent": agent_name, **usage})
    return history


def print_agent_debug(agent_name: str, messages: list[Any]) -> None:
    """Print a compact trace useful when diagnosing live agent responses."""

    print(f"\n========== {agent_name.upper()} DEBUG TRACE ==========")
    print("Messages returned:", len(messages))

    for index, message in enumerate(messages, start=1):
        message_type = getattr(message, "type", "unknown")
        content = getattr(message, "content", None)
        metadata = getattr(message, "response_metadata", {}) or {}
        tool_calls = getattr(message, "tool_calls", []) or []

        print(f"\nMessage {index}")
        print("Type:", message_type)
        print("Content:", repr(content)[:500])
        print("Tool calls:", tool_calls)
        print("Finish reason:", metadata.get("finish_reason"))


def extract_last_ai_text(messages: list[Any]) -> str:
    """Return the last non-empty assistant message from an agent run."""

    response_text = ""

    for message in messages:
        if getattr(message, "type", "") == "ai":
            text_content = extract_text(getattr(message, "content", None))
            if text_content.strip():
                response_text = text_content.strip()

    return response_text


def summarize_usage_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize classifier and specialist usage without inventing cost."""

    classifier_records = [
        item for item in history if item.get("agent") == "model_router_classifier"
    ]
    specialist_records = [
        item for item in history
        if item.get("agent") in {"researcher", "developer", "reviewer"}
    ]
    actual_records = [item for item in history if item.get("source") == "openrouter_actual"]
    mock_records = [item for item in history if item.get("source") == "mock"]
    measured_records = [item for item in actual_records if item.get("tokens_measured")]

    return {
        "agent_invocations_total": len(specialist_records),
        "openrouter_agent_invocations": sum(
            item.get("source") == "openrouter_actual" for item in specialist_records
        ),
        "mock_agent_invocations": sum(
            item.get("source") == "mock" for item in specialist_records
        ),
        "classifier_invocations_total": len(classifier_records),
        "openrouter_classifier_invocations": sum(
            item.get("source") == "openrouter_actual" for item in classifier_records
        ),
        "mock_classifier_invocations": sum(
            item.get("source") == "mock" for item in classifier_records
        ),
        "openrouter_calls_total": len(actual_records),
        "mock_calls_total": len(mock_records),
        "measured_input_tokens": sum(
            item.get("input_tokens", 0) or 0 for item in measured_records
        ),
        "measured_output_tokens": sum(
            item.get("output_tokens", 0) or 0 for item in measured_records
        ),
        "measured_total_tokens": sum(
            item.get("total_tokens", 0) or 0 for item in measured_records
        ),
        "api_cost_usd": 0.0 if not actual_records else None,
        "cost_note": (
            "No LLM API calls were made in mock mode."
            if not actual_records
            else "Token usage is recorded; billed USD cost is not available "
                 "from this usage metadata."
        ),
    }


# ---------------------------------------------------------------------------
# 8. Smart model routing helpers
# ---------------------------------------------------------------------------

def mock_classify_task(task: str) -> dict[str, Any]:
    """Deterministic routing simulation; this does not call an LLM."""

    normalized = task.lower()
    complex_terms = (
        "multi-agent", "multi agent", "system architecture", "distributed",
        "decompose", "orchestration", "enterprise architecture",
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
        "complexity": tier,
        "complexity_score": score,
        "confidence": 1.0,
        "task_type": task_type,
        "requires_decomposition": tier == "complex",
        "requires_tools": False,
        "requires_research": any(
            term in normalized for term in ("research", "rag", "compare", "architecture")
        ),
        "rationale": rationale,
        "selected_model": MODEL_REGISTRY[tier],
        "source": "mock_rule_based",
    }


def choose_agent_model_tiers(complexity: str) -> dict[str, str]:
    """Apply role-aware model budgets after one project-level classification."""

    if complexity == "simple":
        return {
            "researcher": "simple",
            "developer": "simple",
            "reviewer": "simple",
        }

    if complexity == "medium":
        return {
            "researcher": "medium",
            "developer": "medium",
            "reviewer": "medium",
        }

    if complexity == "complex":
        # Keep research and review on the medium tier; reserve the strongest
        # model for implementation, where multi-step code synthesis is needed.
        return {
            "researcher": "medium",
            "developer": "complex",
            "reviewer": "medium",
        }

    raise ValueError(f"Unsupported routing complexity: {complexity}")


def normalize_router_usage(raw_usage: dict[str, Any]) -> dict[str, Any]:
    """Normalize classifier token metadata into the shared usage schema."""

    input_tokens = raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)) or 0
    output_tokens = raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)) or 0
    total_tokens = raw_usage.get("total_tokens", input_tokens + output_tokens) or 0
    measured = bool(raw_usage)

    return {
        "source": "openrouter_actual",
        "agent": "model_router_classifier",
        "tokens_measured": measured,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "api_cost_usd": None,
    }


def smart_router_node(state: WorkflowState) -> dict[str, Any]:
    """Classify the incoming project once and choose a model for each role."""

    task = state.get("task", "").strip()
    if not task:
        raise ValueError("Task cannot be empty before model routing.")

    print("\n[LangGraph] Executing Smart Model Router")

    if EXECUTION_MODE == "mock":
        routing = mock_classify_task(task)
        router_usage = mock_usage("model_router_classifier")
        routing_source = "mock_rule_based"
    else:
        # Import only in live mode. model_router.py validates its API settings
        # during import and makes no request until classify_and_route is called.
        from model_router import classify_and_route

        routing = classify_and_route(task)
        router_usage = normalize_router_usage(routing.get("classifier_usage", {}))
        routing_source = "openrouter_classifier"

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

    return {
        "model_routing": routing_summary,
        "agent_models": agent_models,
        "router_usage": router_usage,
        "usage_history": append_usage_history(
            state, "model_router_classifier", router_usage
        ),
    }


# ---------------------------------------------------------------------------
# 8. Planner node — deterministic, no LLM call
# ---------------------------------------------------------------------------

def planner_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Executing Planner")

    task = state.get("task", "").strip()
    if not task:
        raise ValueError("Task cannot be empty.")

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
- Classification source: {state.get('model_routing', {}).get('source', 'not available')}
- Project complexity: {state.get('model_routing', {}).get('complexity', 'not available')}
- Researcher model: {state.get('agent_models', {}).get('researcher', {}).get('model', 'not available')}
- Developer model: {state.get('agent_models', {}).get('developer', {}).get('model', 'not available')}
- Reviewer model: {state.get('agent_models', {}).get('reviewer', {}).get('model', 'not available')}
- At most {state.get('max_revisions', 1)} developer revision(s) are allowed.
""".strip()

    print("[Planner] Execution plan created.")
    return {"plan": plan}


# ---------------------------------------------------------------------------
# 9. Researcher node
# ---------------------------------------------------------------------------

def researcher_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Invoking Researcher Deep Agent")

    if EXECUTION_MODE == "mock":
        print("[MOCK] Researcher Deep Agent invocation skipped.")

        research_report = """
[SIMULATED RESEARCH OUTPUT — NOT GENERATED BY AN LLM]

# Reciprocal Rank Fusion in Production RAG

## 1. Why rank fusion is needed

Hybrid retrieval combines dense vector search with lexical retrieval such as
BM25. Dense retrieval captures semantic similarity, while BM25 performs well
for exact keywords, identifiers, and technical terms. Their relevance scores
have different scales, so directly combining raw scores can be unreliable.
RRF combines document rankings rather than raw relevance scores.

## 2. RRF formula

RRF(d) = sum over retrieval systems of 1 / (k + rank(d))

Where:
- d is a document.
- k is a smoothing constant, commonly 60.
- rank(d) is the 1-based position of the document in a list.

A document ranked highly by multiple retrieval systems receives a higher
combined score.

## 3. Advantages

- Does not require relevance scores to be calibrated.
- Combines heterogeneous retrieval systems.
- Simple and computationally inexpensive.
- Rewards documents appearing consistently across rankings.

## 4. Limitations

- Does not directly use the original relevance scores.
- Depends on the quality of the individual retrieval lists.
- The k parameter influences score distribution.
- Additional reranking may be needed for higher precision.

## 5. Production considerations

Use authorization filters before retrieval, deduplicate document identifiers,
tune retrieval depth, and evaluate recall@K, precision@K, and answer faithfulness.
""".strip()

        selection = state["agent_models"]["researcher"]
        usage = {
            **mock_usage("researcher"),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
        return {
            "research_report": research_report,
            "researcher_usage": usage,
            "usage_history": append_usage_history(state, "researcher", usage),
        }

    research_task = state.get("research_task") or state["task"]

    selection = state["agent_models"]["researcher"]
    researcher_agent = get_specialist_agent("researcher", selection["model"])
    agent_result = researcher_agent.invoke(
        {
            "messages": [{"role": "user", "content": research_task}],
            "files": load_skill_files(),
        }
    )

    messages = agent_result.get("messages", [])
    print_agent_debug("researcher", messages)

    research_report = extract_last_ai_text(messages)
    usage = {
        **aggregate_usage(messages),
        "selected_tier": selection["tier"],
        "selected_model": selection["model"],
    }
    print("\nResearcher usage:", usage)

    if not research_report:
        raise RuntimeError(
            "Researcher returned no textual response. Inspect the debug trace."
        )

    print("\n[LangGraph] Researcher completed")
    return {
        "research_report": research_report,
        "researcher_usage": usage,
        "usage_history": append_usage_history(state, "researcher", usage),
    }


# ---------------------------------------------------------------------------
# 10. Developer node
# ---------------------------------------------------------------------------

def developer_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Invoking Developer Deep Agent")

    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")
    next_revision_count = revision_count + (1 if review_feedback else 0)

    if EXECUTION_MODE == "mock":
        print("[MOCK] Developer Deep Agent invocation skipped.")

        implementation = """
[SIMULATED IMPLEMENTATION OUTPUT — NOT GENERATED BY AN LLM]

## Python implementation

```python
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    # Combine ranked document lists using RRF.

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    if isinstance(rankings, (str, bytes)):
        raise TypeError("rankings must be a sequence of ranked sequences")

    scores: dict[str, float] = {}

    for ranking in rankings:
        if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
            raise TypeError("Each ranking must be a sequence of document IDs")

        seen: set[str] = set()

        for rank, doc_id in enumerate(ranking, start=1):
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Document IDs must be non-empty strings")

            if doc_id in seen:
                continue

            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
```

## Pytest tests

```python
import pytest

from rrf import reciprocal_rank_fusion


def test_single_ranking():
    result = reciprocal_rank_fusion([["a", "b"]])
    assert result[0] == ("a", 1 / 61)
    assert result[1] == ("b", 1 / 62)


def test_multiple_rankings():
    result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert result[0][0] == "b"


def test_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_invalid_outer_input():
    with pytest.raises(TypeError):
        reciprocal_rank_fusion("not-a-ranking")


def test_invalid_inner_input():
    with pytest.raises(TypeError):
        reciprocal_rank_fusion(["not-a-ranking"])


def test_duplicate_document_is_counted_once_per_ranking():
    result = reciprocal_rank_fusion([["a", "a", "b"]])
    assert result[0][1] == 1 / 61
```

## Implementation decisions

- Uses 1-based document ranks.
- Deduplicates documents within each individual ranking.
- Supports multiple retrieval result lists.
- Uses deterministic sorting for equal scores.
- Validates the smoothing parameter and ranking inputs.

""".strip()

        if review_feedback:
            implementation += """

## Simulated revision summary

This is a simulated revision for workflow demonstration, not a regenerated
implementation. The mock revision records the feedback passed from Reviewer:
- Validate the outer rankings collection.
- Validate individual ranking types.
- Add edge-case tests.
- Document duplicate-document handling.
"""

        selection = state["agent_models"]["developer"]
        usage = {
            **mock_usage("developer"),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
        return {
            "implementation": implementation,
            "developer_usage": usage,
            "revision_count": next_revision_count,
            "usage_history": append_usage_history(state, "developer", usage),
        }

    developer_task = f"""
ORIGINAL REQUIREMENTS:
{state['task']}

RESEARCH REPORT:
{state.get('research_report', '')}

PREVIOUS IMPLEMENTATION:
{state.get('implementation') or 'No previous implementation. This is the initial implementation.'}

PREVIOUS REVIEW FEEDBACK:
{review_feedback or 'No previous review feedback. This is the initial implementation.'}

Implement the requested functionality.

Include:
- Python implementation.
- Type hints.
- Input validation.
- Error handling.
- Pytest unit tests.
- A concise explanation of important implementation decisions.

If review feedback is provided, revise the previous implementation to address
all actionable findings. Preserve functionality that already works. Do not
merely append a revision summary; return the revised implementation and tests.

Keep the implementation focused and concise. Do not repeat the research report.
""".strip()

    selection = state["agent_models"]["developer"]
    developer_agent = get_specialist_agent("developer", selection["model"])
    agent_result = developer_agent.invoke(
        {
            "messages": [{"role": "user", "content": developer_task}],
            "files": load_skill_files(),
        }
    )

    messages = agent_result.get("messages", [])
    print_agent_debug("developer", messages)

    implementation = extract_last_ai_text(messages)
    usage = {
        **aggregate_usage(messages),
        "selected_tier": selection["tier"],
        "selected_model": selection["model"],
    }
    print("\nDeveloper usage:", usage)

    if not implementation:
        raise RuntimeError(
            "Developer returned no textual implementation. Inspect the debug trace."
        )

    print("\n[LangGraph] Developer completed")
    return {
        "implementation": implementation,
        "developer_usage": usage,
        "revision_count": next_revision_count,
        "usage_history": append_usage_history(state, "developer", usage),
    }


# ---------------------------------------------------------------------------
# 11. Reviewer node
# ---------------------------------------------------------------------------

def reviewer_node(state: WorkflowState) -> dict[str, Any]:
    print("\n[LangGraph] Invoking Reviewer Deep Agent")

    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)

    if EXECUTION_MODE == "mock":
        print("[MOCK] Reviewer Deep Agent invocation skipped.")

        if revision_count < max_revisions:
            review_status = "changes_requested"
            review_feedback = """
Please address these findings:

1. Validate the outer rankings collection.
2. Validate individual ranking types.
3. Add edge-case tests.
4. Document duplicate-document handling.
""".strip()
            review_report = """
[SIMULATED REVIEW OUTPUT — NOT GENERATED BY AN LLM]

# Code Review — Initial Review

## Findings

1. Medium — Input validation:
   Validate the outer rankings collection and each ranking's type.

2. Medium — Test coverage:
   Add tests for empty inputs, invalid ranking structures, equal-score
   tie-breaking, and non-string document IDs.

3. Low — Maintainability:
   Document the duplicate-document policy and its effect on rank positions.

## Recommended fixes

- Add explicit validation for ranking inputs.
- Expand unit tests for edge cases.
- Document duplicate handling.

REVIEW_STATUS: CHANGES_REQUESTED
""".strip()
        else:
            review_status = "approved"
            review_feedback = ""
            review_report = """
[SIMULATED REVIEW OUTPUT — NOT GENERATED BY AN LLM]

# Code Review — Revision Demonstration

The mock workflow has reached its scripted approval branch after one
simulated developer revision.

IMPORTANT: This mock approval demonstrates conditional LangGraph routing.
It does not execute pytest, inspect regenerated code, or establish that the
implementation is production-ready.

REVIEW_STATUS: APPROVED
""".strip()

        selection = state["agent_models"]["reviewer"]
        usage = {
            **mock_usage("reviewer"),
            "selected_tier": selection["tier"],
            "selected_model": selection["model"],
        }
        return {
            "review_report": review_report,
            "reviewer_usage": usage,
            "review_status": review_status,
            "review_feedback": review_feedback,
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

Provide actionable findings with severity and recommended fixes.
Do not approve merely because the response contains code. If actionable
correctness, security, or requirement gaps remain, request changes.

At the very end, include exactly one decision marker on its own line:
REVIEW_STATUS: APPROVED
or
REVIEW_STATUS: CHANGES_REQUESTED
""".strip()

    selection = state["agent_models"]["reviewer"]
    reviewer_agent = get_specialist_agent("reviewer", selection["model"])
    agent_result = reviewer_agent.invoke(
        {
            "messages": [{"role": "user", "content": reviewer_task}],
            "files": load_skill_files(),
        }
    )

    messages = agent_result.get("messages", [])
    print_agent_debug("reviewer", messages)

    review_report = extract_last_ai_text(messages)
    usage = {
        **aggregate_usage(messages),
        "selected_tier": selection["tier"],
        "selected_model": selection["model"],
    }
    print("\nReviewer usage:", usage)

    if not review_report:
        raise RuntimeError(
            "Reviewer returned no textual response. Inspect the debug trace."
        )

    # Parse the last explicit decision marker. If the marker is missing or
    # malformed, fail closed and request changes rather than auto-approving.
    decision_markers = re.findall(
        r"^\s*REVIEW_STATUS:\s*(APPROVED|CHANGES_REQUESTED)\s*$",
        review_report,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    decision = decision_markers[-1].upper() if decision_markers else None

    if decision == "APPROVED":
        review_status = "approved"
        review_feedback = ""
    else:
        review_status = "changes_requested"
        review_feedback = review_report

    print("\n[LangGraph] Reviewer completed:", review_status)
    return {
        "review_report": review_report,
        "reviewer_usage": usage,
        "review_status": review_status,
        "review_feedback": review_feedback,
        "usage_history": append_usage_history(state, "reviewer", usage),
    }


# ---------------------------------------------------------------------------
# 12. Conditional routing after review
# ---------------------------------------------------------------------------

def route_after_review(state: WorkflowState) -> str:
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


# ---------------------------------------------------------------------------
# 13. Build the parent LangGraph
# ---------------------------------------------------------------------------

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
    {
        "developer": "developer",
        "end": END,
    },
)

research_workflow = builder.compile()


# ---------------------------------------------------------------------------
# 14. Run the demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = task = """
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
"""

    research_task =""

    print(f"Execution mode: {EXECUTION_MODE.upper()}")

    skill_files = load_skill_files()
    print("\n========== AVAILABLE SKILLS ==========")
    for skill_path in skill_files:
        print(f" - {skill_path}")

    print("\n========== LANGGRAPH STRUCTURE ==========\n")
    print(research_workflow.get_graph().draw_mermaid())

    print("\n========== EXECUTION ==========\n")

    result = research_workflow.invoke(
        {
            "task": task,
            "research_task": research_task,
            "revision_count": 0,
            "max_revisions": 1,
            "usage_history": [],
        }
    )

    print("\n========== MODEL ROUTING DECISION ==========\n")
    print(result.get("model_routing", {}))

    print("\n========== PER-AGENT MODEL SELECTION ==========\n")
    for agent_name, selection in result.get("agent_models", {}).items():
        print(f"{agent_name}: {selection['tier']} -> {selection['model']}")

    print("\n========== EXECUTION PLAN ==========\n")
    print(result.get("plan", "No plan returned."))

    print("\n========== RESEARCH REPORT ==========\n")
    print(result.get("research_report", "No research report returned."))

    print("\n========== RESEARCHER TOKEN USAGE ==========\n")
    print(result.get("researcher_usage", {}))

    print("\n========== DEVELOPER IMPLEMENTATION ==========\n")
    print(result.get("implementation", "No implementation returned."))

    print("\n========== DEVELOPER TOKEN USAGE (LATEST INVOCATION) ==========\n")
    print(result.get("developer_usage", {}))

    print("\n========== REVIEW REPORT ==========\n")
    print(result.get("review_report", "No review report returned."))

    print("\n========== REVIEWER TOKEN USAGE (LATEST INVOCATION) ==========\n")
    print(result.get("reviewer_usage", {}))

    print("\n========== FINAL REVIEW STATUS ==========\n")
    print(result.get("review_status", "unknown"))
    print("Revision count:", result.get("revision_count", 0))
    print("Maximum revisions:", result.get("max_revisions", 1))

    print("\n========== FINAL REVIEW FEEDBACK ==========\n")
    print(result.get("review_feedback", ""))

    print("\n========== AGENT USAGE HISTORY ==========\n")
    for item in result.get("usage_history", []):
        print(item)

    print("\n========== USAGE SUMMARY ==========\n")
    print(summarize_usage_history(result.get("usage_history", [])))
