
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


MODEL_REGISTRY = {
    "simple": os.getenv("SIMPLE_MODEL"),
    "medium": os.getenv("MEDIUM_MODEL"),
    "complex": os.getenv("COMPLEX_MODEL"),
}


def select_model(complexity: str) -> str:
    """Select a model based on task complexity."""

    model = MODEL_REGISTRY.get(complexity)

    if not model:
        raise ValueError(
            f"No model configured for: {complexity}"
        )

    return model


def get_llm(model_name: str) -> ChatOpenAI:
    """Create an LLM client using GroqCloud."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1"
        ),
        max_tokens=1500,
        temperature=0.2,
    )
