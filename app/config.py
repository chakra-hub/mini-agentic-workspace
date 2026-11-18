"""Application configuration and environment settings."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Project root is one level above the app package.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the project's environment configuration.
load_dotenv(BASE_DIR / ".env", override=True)


# Execution mode: mock or live.
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "mock").strip().lower()

if EXECUTION_MODE not in {"mock", "live"}:
    raise ValueError("EXECUTION_MODE must be either 'mock' or 'live'.")


# OpenRouter configuration.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

API_KEY = os.getenv("OPENROUTER_API_KEY")

CLASSIFIER_MODEL = os.getenv("OPENROUTER_CLASSIFIER_MODEL")

MODEL_REGISTRY = {
    "simple": os.getenv("OPENROUTER_SIMPLE_MODEL"),
    "medium": os.getenv("OPENROUTER_MEDIUM_MODEL"),
    "complex": os.getenv("OPENROUTER_COMPLEX_MODEL"),
}


# Validate configuration only when live execution is requested.
if EXECUTION_MODE == "live":

    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing from .env.")

    if not CLASSIFIER_MODEL:
        raise ValueError("OPENROUTER_CLASSIFIER_MODEL is missing from .env.")

    missing_models = [
        tier
        for tier, model in MODEL_REGISTRY.items()
        if not model
    ]

    if missing_models:
        raise ValueError(
            "Missing OpenRouter model IDs for: "
            + ", ".join(missing_models)
        )

else:
    # Placeholders are never sent to an LLM in mock mode.
    API_KEY = API_KEY or "mock-mode-no-network-key"

    MODEL_REGISTRY = {
        tier: model or f"mock-{tier}-model-not-used"
        for tier, model in MODEL_REGISTRY.items()
    }
