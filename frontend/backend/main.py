import uuid
import asyncio
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store
tasks: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    question: str
    history: List[Any] = []

class TaskResponse(BaseModel):
    taskId: str
    message: str

async def process_task(task_id: str, question: str):
    """Simulate long-running task (LLM/OCR)"""
    tasks[task_id]["status"] = "processing"

    # 1. Timeline: Analysis
    tasks[task_id]["timeline"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "message": "Analyzing intent...",
        "detail": "Keywords identified: text enhancement, agent"
    })
    await asyncio.sleep(1)

    # 2. Timeline: Retrieval
    tasks[task_id]["timeline"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "message": "Searching the knowledge base...",
        "detail": "Found 3 relevant documents, relevance is low"
    })
    await asyncio.sleep(1.5)

    # 3. Timeline: Judgment
    tasks[task_id]["timeline"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "message": "Agent judgment: insufficient resources",
        "detail": "Decided to trigger the crawler module"
    })
    await asyncio.sleep(1.5)

    # 4. Timeline: Crawling
    tasks[task_id]["timeline"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "message": "Searching the web and downloading the latest papers...",
        "detail": "Target: arXiv:2401.xxxxx.pdf"
    })
    await asyncio.sleep(2)

    # 5. Timeline: OCR
    tasks[task_id]["timeline"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "message": "Extracting text via OCR...",
        "detail": "Parsing PDF page 1/12"
    })
    await asyncio.sleep(2)

    # 6. Timeline: Generation
    tasks[task_id]["timeline"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "message": "Generating the final answer...",
        "detail": "Synthesizing all information"
    })
    await asyncio.sleep(1.5)

    # Complete
    tasks[task_id]["status"] = "completed"
    tasks[task_id]["result"] = {
        "answer": f"This is the backend's answer to the question **\"{question}\"** after running through the full pipeline.\n\n### Agent execution report\n\nThe agent autonomously determined that the local knowledge base was insufficient to answer your question, so it triggered a web search, downloaded and read the latest arXiv paper.\n\n**Key findings:**\n1. RAG combined with an Agent enables self-correction.\n2. The crawler and OCR modules act as the system's \"eyes\".\n\n*(Note: the backend is currently in demo mode and the interface pipeline is wired up; connect a real LLM API key to generate real content.)*",
        "sources": ["arXiv:2401.xxxxx.pdf (newly crawled)", "Local knowledge base doc A"],
        "confidence": 0.98
    }

@app.post("/api/v1/chat/submit")
async def submit_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "question": request.question,
        "createdAt": time.time(),
        "timeline": [],
        "result": None,
        "error": None
    }

    # Run processing in background
    background_tasks.add_task(process_task, task_id, request.question)

    return {
        "success": True,
        "data": {
            "taskId": task_id,
            "message": "Task submitted, processing..."
        }
    }

@app.get("/api/v1/task/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "success": True,
        "data": {
            "status": task["status"],
            "timeline": task["timeline"],
            "result": task["result"],
            "error": task["error"]
        }
    }

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
