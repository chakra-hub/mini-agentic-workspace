
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field


# Load environment variables from this project's .env file.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")

CLASSIFIER_MODEL = os.getenv(
    "OPENROUTER_CLASSIFIER_MODEL"
)

MODEL_REGISTRY = {
    "simple": os.getenv("OPENROUTER_SIMPLE_MODEL"),
    "medium": os.getenv("OPENROUTER_MEDIUM_MODEL"),
    "complex": os.getenv("OPENROUTER_COMPLEX_MODEL"),
}

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing.")

if not CLASSIFIER_MODEL:
    raise ValueError("OPENROUTER_CLASSIFIER_MODEL is missing.")

if any(not model for model in MODEL_REGISTRY.values()):
    raise ValueError("One or more routing models are missing from .env.")


# Define the expected output of the classification model.
class RouterAnalysis(BaseModel):
    complexity_score: int = Field(ge=0, le=100)

    task_type: str

    requires_decomposition: bool

    requires_tools: bool

    requires_research: bool

    confidence: float = Field(ge=0, le=1)

    rationale: str


# Initialize the classifier model.
classifier = ChatOpenAI(
    model=CLASSIFIER_MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=700,
)


SYSTEM_PROMPT = """
You are an intelligent task complexity classifier
for an Agentic AI system.

Your job is to classify the complexity of a user's task.
Do not answer or execute the task.

Evaluate the actual work required, not just the length
of the user's message.

COMPLEXITY SCORING:

0-34: SIMPLE
- Definitions and straightforward explanations.
- Basic conceptual questions.
- Small text transformations.
- Single-step tasks with no external tools required.

35-69: MEDIUM
- Writing small or moderate code snippets.
- Debugging a self-contained code example.
- Technical comparisons with trade-offs.
- Detailed explanations involving multiple concepts.
- Tasks requiring moderate reasoning but no major delegation.

70-100: COMPLEX
- System or software architecture design.
- Multi-stage implementation tasks.
- Multi-agent workflows.
- Tasks requiring decomposition into independent subtasks.
- Research, planning and synthesis across multiple stages.
- Tasks requiring multiple tools or specialist agents.

IMPORTANT RULES:

1. Do not classify every coding task as complex.
2. A simple explanation remains simple even if it contains
   technical terminology.
3. A task with multiple deliverables may be complex.
4. requires_tools means external tools, files, retrieval
   or execution are genuinely needed, not merely helpful.
5. requires_decomposition means the task should be broken
   into independently manageable subtasks.
6. Be conservative when uncertain and report your confidence.
7. Base your decision on the latest task provided.

Return ONLY one valid JSON object with these fields:

{
  "complexity_score": 0,
  "task_type": "explanation",
  "requires_decomposition": false,
  "requires_tools": false,
  "requires_research": false,
  "confidence": 0.95,
  "rationale": "A brief explanation of the classification"
}
"""


# Handle either a normal string or a list of content blocks.
def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


# Extract and validate the JSON returned by the model.
def parse_analysis(content: Any) -> RouterAnalysis:
    text = extract_text(content).strip()

    # Allow the model to wrap JSON in a Markdown code block.
    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError(
            f"Classifier did not return valid JSON: {text}"
        )

    data = json.loads(match.group(0))

    return RouterAnalysis.model_validate(data)


# Promote a route when the classifier is not sufficiently confident.
def apply_routing_policy(analysis: RouterAnalysis) -> str:
    score = analysis.complexity_score

    if (
        score >= 70
        or analysis.requires_decomposition
        or analysis.requires_tools
    ):
        tier = "complex"

    elif score >= 35 or analysis.requires_research:
        tier = "medium"

    else:
        tier = "simple"

    # Avoid sending uncertain tasks down the cheapest path.
    if analysis.confidence < 0.65:
        if tier == "simple":
            tier = "medium"

        elif tier == "medium":
            tier = "complex"

    return tier


# Main entry point for the model router.
def classify_and_route(task: str) -> dict:
    if not task or not task.strip():
        raise ValueError("Task cannot be empty.")

    # Ask the classifier to analyse the user request.
    response = classifier.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Classify this task:\n\n{task}"
            ),
        ]
    )

    # Validate the model's classification.
    analysis = parse_analysis(response.content)

    # Apply our routing policy.
    tier = apply_routing_policy(analysis)

    # Select the corresponding model.
    selected_model = MODEL_REGISTRY[tier]

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
        "selected_model": selected_model,
        "classifier_usage": response.usage_metadata or {},
    }

def execute_routed_task(task: str) -> dict:
    # First, determine the task complexity and selected model.
    routing = classify_and_route(task)

    # Initialize the selected execution model.
    execution_llm = ChatOpenAI(
        model=routing["selected_model"],
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=2048,
    )

    # Execute the original task using the selected model.
    response = execution_llm.invoke(task)

    # Return both the classification and execution details.
    return {
        **routing,
        "response": extract_text(response.content),
        "execution_usage": response.usage_metadata or {},
        "execution_finish_reason": (
            response.response_metadata.get("finish_reason")
        ),
    }


# Run a small routing demonstration.
if __name__ == "__main__":

    task = (
        "Design an enterprise multi-agent RAG architecture "
        "with document ingestion, hybrid retrieval, "
        "authorization, evaluation and observability."
    )

    result = execute_routed_task(task)

    print("\nSelected complexity:", result["complexity"])
    print("Selected model:", result["selected_model"])

    print("\nClassification usage:")
    print(result["classifier_usage"])

    print("\nModel response:")
    print(result["response"])

    print("\nExecution usage:")
    print(result["execution_usage"])
