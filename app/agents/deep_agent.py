
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data

from app.model_router.router import (
    get_llm,
    select_model,
)

from app.sandbox.mock_executor import mock_execute


def build_deep_agent():

    # Configure models for different responsibilities.
    planner_model = get_llm(select_model("complex"))
    research_model = get_llm(select_model("simple"))
    developer_model = get_llm(select_model("complex"))
    reviewer_model = get_llm(select_model("simple"))

    # Define specialized subagents.
    subagents = [

        {
            "name": "researcher",
            "description": (
                "Use this agent for technical research, "
                "technology comparisons and architecture analysis."
            ),
            "model": research_model,
            "system_prompt": (
                "You are a technical research specialist. "
                "Provide structured findings, trade-offs "
                "and practical recommendations."
            ),
        },

        {
            "name": "developer",
            "description": (
                "Use this agent to design and generate "
                "Python code and application implementations."
            ),
            "model": developer_model,
            "tools": [mock_execute],
            "system_prompt": (
                "You are a Python software developer. "
                "Generate clean, maintainable code. "
                "When execution is requested, use mock_execute. "
                "Never claim that code was actually executed."
            ),
        },

        {
            "name": "reviewer",
            "description": (
                "Use this agent to review code for bugs, "
                "security issues and maintainability."
            ),
            "model": reviewer_model,
            "system_prompt": (
                "You are a senior software engineer. "
                "Review code critically and provide "
                "actionable recommendations."
            ),
        },
    ]

    # Use state-backed files rather than exposing the host filesystem.
    backend = StateBackend()

    # Create the main Deep Agent.
    agent = create_deep_agent(
        model=planner_model,
        backend=backend,
        subagents=subagents,
        skills=["/skills/"],
        system_prompt="""
        You are the supervisor of a multi-agent AI workspace.

        Your responsibilities:

        1. Understand the user's task.
        2. Create a clear execution plan.
        3. Use relevant skills when necessary.
        4. Delegate research tasks to the researcher.
        5. Delegate coding tasks to the developer.
        6. Delegate code review to the reviewer.
        7. Combine the results.
        8. Return a structured final response.

        Do not delegate simple questions unnecessarily.

        Never claim that code was executed unless
        an execution result explicitly confirms it.
        """,
    )

    return agent


# Build the reusable agent instance.
deep_agent = build_deep_agent()


# Prepare skill files for the StateBackend.
def load_skill_files():
    skills = [
        "research",
        "software-development",
        "code-review",
    ]

    files = {}

    for skill in skills:
        skill_path = (
            f"skills/{skill}/SKILL.md"
        )

        with open(skill_path, encoding="utf-8") as file:
            content = file.read()

        virtual_path = f"/skills/{skill}/SKILL.md"

        files[virtual_path] = create_file_data(content)

    return files


def execute_deep_agent(task: str):
    """
    Execute a complex task using Deep Agents.
    """

    result = deep_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ],
            "files": load_skill_files(),
        }
    )

    # Extract the final assistant message.
    messages = result["messages"]

    for message in reversed(messages):
        if message.type == "ai" and message.content:
            return {
                "response": message.content,
                "status": "completed",
            }

    return {
        "response": "Agent completed without a text response.",
        "status": "completed",
    }
