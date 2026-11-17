
from langchain_core.tools import tool


@tool
def mock_execute(code: str) -> str:
    """
    Simulate code execution without executing it.

    This tool is for demonstrating an isolated execution
    workflow in the POC.
    """

    if not code.strip():
        return "Execution rejected: empty code."

    return (
        "MOCK SANDBOX RESULT\n"
        "-------------------\n"
        "Execution environment: Isolated mock workspace\n"
        "Code received: Yes\n"
        "Actual execution: No\n"
        "Status: Simulated successfully\n"
        "Output: Code was accepted by the mock executor. "
        "No Python code was executed."
    )
