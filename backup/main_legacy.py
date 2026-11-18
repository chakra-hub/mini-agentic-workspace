
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph.workflow import agent_workflow


app = FastAPI(
    title="Mini Agentic Workspace",
    description="LangGraph + Deep Agents + Groq POC",
    version="0.3.0",
)


class TaskRequest(BaseModel):
    task: str = Field(min_length=3, max_length=10000)


@app.get("/")
def root():
    return {"message": "Mini Agentic Workspace is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/execute")
def execute_task(request: TaskRequest):

    run_id = str(uuid4())

    initial_state = {
        "run_id": run_id,
        "task": request.task,
        "complexity": None,
        "classification_reason": None,
        "selected_model": None,
        "execution_path": None,
        "response": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": None,
        "status": "started",
    }

    try:
        result = agent_workflow.invoke(initial_state)

        return {
            "run_id": run_id,
            "task": request.task,
            "complexity": result["complexity"],
            "reason": result["classification_reason"],
            "execution_path": result["execution_path"],
            "selected_model": result["selected_model"],
            "response": result["response"],
            "usage": {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"],
                "cost_usd": result["cost_usd"],
            },
            "status": result["status"],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
