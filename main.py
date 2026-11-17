
from fastapi import FastAPI

app = FastAPI(
    title="Mini Agentic Workspace",
    description="A lightweight multi-agent AI orchestration POC",
    version="0.1.0"
)


@app.get("/")
async def root():
    return {
        "message": "Mini Agentic Workspace is running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }
