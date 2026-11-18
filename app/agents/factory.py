"""Factory for lazily created, model-specific Deep Agents."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from deepagents import (
    create_deep_agent,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from deepagents.backends import StateBackend

from app.config import API_KEY, EXECUTION_MODE, OPENROUTER_BASE_URL
from app.agents.prompts import AGENT_MAX_TOKENS, AGENT_SYSTEM_PROMPTS


# The parent LangGraph controls specialist orchestration. Disable automatic
# general-purpose subagents and code execution to keep routing explicit.
register_harness_profile(
    "openai",
    HarnessProfile(
        excluded_tools=frozenset({"execute"}),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)


@lru_cache(maxsize=12)
def get_specialist_agent(agent_name: str, model_name: str):
    """Build and cache a specialist Deep Agent for the selected model ID."""

    if EXECUTION_MODE != "live":
        raise RuntimeError(
            "Deep Agent inference is disabled in mock mode. "
            "Set EXECUTION_MODE=live to invoke a model."
        )

    if agent_name not in AGENT_SYSTEM_PROMPTS:
        raise ValueError(f"Unknown specialist agent: {agent_name}")

    if not model_name:
        raise ValueError(f"No model configured for specialist: {agent_name}")

    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing.")

    llm = ChatOpenAI(
        model=model_name,
        api_key=API_KEY,
        base_url=OPENROUTER_BASE_URL,
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
