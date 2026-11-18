"""Live OpenRouter complexity classifier."""

import json
import re
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import (
    API_KEY,
    CLASSIFIER_MODEL,
    EXECUTION_MODE,
    MODEL_REGISTRY,
    OPENROUTER_BASE_URL,
)

from app.model_router.policy import apply_routing_policy


class RouterAnalysis(BaseModel):
    """Validated classifier response."""

    complexity_score: int = Field(ge=0, le=100)
    task_type: str
    requires_decomposition: bool
    requires_tools: bool
    requires_research: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str


SYSTEM_PROMPT = """
You are an intelligent task complexity classifier
for an Agentic AI system.

Classify the actual work required, not the length of the user's message.
Do not answer or execute the task.

COMPLEXITY SCORING:

0-34: SIMPLE
- Definitions and straightforward explanations.
- Basic conceptual questions.
- Small text transformations.
- Single-step tasks without external tools.

35-69: MEDIUM
- Writing small or moderate code snippets.
- Debugging self-contained code.
- Technical comparisons with trade-offs.
- Detailed explanations involving multiple concepts.
- Moderate reasoning without major delegation.

70-100: COMPLEX
- System or software architecture design.
- Multi-stage implementation.
- Multi-agent workflows.
- Tasks requiring decomposition into independent subtasks.
- Research, planning and synthesis across multiple stages.
- Tasks requiring multiple tools or specialist agents.

IMPORTANT RULES:

1. Do not classify every coding task as complex.
2. A simple explanation remains simple even with technical terminology.
3. Multiple deliverables may indicate complexity.
4. requires_tools means tools are genuinely needed.
5. requires_decomposition means independent subtasks are required.
6. Report your confidence honestly.
7. Classify the latest task provided.

Return ONLY one valid JSON object with these fields:

{
  "complexity_score": 0,
  "task_type": "explanation",
  "requires_decomposition": false,
  "requires_tools": false,
  "requires_research": false,
  "confidence": 0.95,
  "rationale": "Brief explanation of the classification"
}
"""


def extract_text(content: Any) -> str:
    """Extract readable text from a model response."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(item.text)

        return "\n".join(parts)

    return str(content)


def parse_analysis(content: Any) -> RouterAnalysis:
    """Extract JSON and validate its fields."""

    text = extract_text(content).strip()
    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError(
            f"Classifier did not return valid JSON: {text}"
        )

    data = json.loads(match.group(0))
    return RouterAnalysis.model_validate(data)


@lru_cache(maxsize=1)
def get_classifier() -> ChatOpenAI:
    """Create and cache the live classifier client."""

    if EXECUTION_MODE != "live":
        raise RuntimeError(
            "Live classifier requested while EXECUTION_MODE is mock."
        )

    if not API_KEY or not CLASSIFIER_MODEL:
        raise ValueError("OpenRouter classifier configuration is missing.")

    return ChatOpenAI(
        model=CLASSIFIER_MODEL,
        api_key=API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_tokens=700,
    )


def classify_and_route(task: str) -> dict[str, Any]:
    """Classify a task using OpenRouter and select its model tier."""

    if not task or not task.strip():
        raise ValueError("Task cannot be empty.")

    classifier = get_classifier()

    response = classifier.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Classify this task:\n\n{task}"),
    ])

    analysis = parse_analysis(response.content)
    tier = apply_routing_policy(analysis)

    return {
        "task": task,
        "complexity": tier,
        "complexity_score": analysis.complexity_score,
        "confidence": analysis.confidence,
        "task_type": analysis.task_type,
        "requires_decomposition": analysis.requires_decomposition,
        "requires_tools": analysis.requires_tools,
        "requires_research": analysis.requires_research,
        "rationale": analysis.rationale,
        "classifier_model": CLASSIFIER_MODEL,
        "selected_model": MODEL_REGISTRY[tier],
        "classifier_usage": response.usage_metadata or {},
    }


def execute_routed_task(task: str) -> dict[str, Any]:
    """Classify a task, then execute it with the selected model."""

    routing = classify_and_route(task)

    execution_llm = ChatOpenAI(
        model=routing["selected_model"],
        api_key=API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_tokens=2048,
    )

    response = execution_llm.invoke(task)

    return {
        **routing,
        "response": extract_text(response.content),
        "execution_usage": response.usage_metadata or {},
        "execution_finish_reason": (
            response.response_metadata.get("finish_reason")
        ),
    }
