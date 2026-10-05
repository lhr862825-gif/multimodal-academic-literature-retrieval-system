from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
from typing import List, Optional, Any, Dict
from datetime import datetime
from main_controller import OCRRAGController
from rag_system import setup_rag_system

app = FastAPI(title="RAG Backend API")

# Allow cross-origin requests (for easier frontend integration testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task database: in production, consider replacing it with Redis + Celery
tasks_db = {}

# Globally initialize the Controller
controller = OCRRAGController("./pdf")
print("Initializing the RAG environment at system startup...")
try:
    controller.setup_pdf_processor()
    controller.setup_arxiv_crawler()
    controller.rag_system = setup_rag_system()
    if controller.rag_system:
        print("System initialization complete!")
    else:
        print("RAG system initialization failed, falling back to Mock mode")
except Exception as e:
    print(f"System initialization error: {e}, falling back to Mock mode")
    controller.rag_system = None


# Define the request data structures used by the frontend
class QuestionRequest(BaseModel):
    question: str

class ChatQuestionRequest(BaseModel):
    message: str
    conversationId: Optional[str] = None
    attachments: Optional[List[Any]] = None

class ChatRetrieveRequest(BaseModel):
    questionId: str
    query: str

class ChatGenerateRequest(BaseModel):
    questionId: str
    query: str
    retrievedDocs: List[Dict[str, Any]]
    conversationHistory: Optional[List[Any]] = None
    chainLength: Optional[int] = None


def run_rag_task(task_id: str, question: str):
    """Background function that runs the main RAG flow."""
    tasks_db[task_id]["status"] = "processing"

    # Closure callback: collects the controller's intermediate status and attaches a timestamp
    def timeline_cb(message: str):
        tasks_db[task_id]["timeline"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message
        })

    try:
        # Call the refactored method, passing in the callback
        answer, docs = controller.ask_question_with_fallback(
            question=question,
            show_docs=True,
            progress_callback=timeline_cb
        )

        # Normalize the returned document metadata to avoid passing non-JSON-serializable objects
        formatted_docs = []
        for d in docs:
            formatted_docs.append({
                "content": d.page_content,
                "source": d.metadata.get("source", "Unknown file")
            })

        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["result"] = {
            "answer": answer,
            "sources": formatted_docs
        }

    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        timeline_cb(f"Task execution failed with an error: {str(e)}")


# Endpoint 1: submit a question (Create Task)
@app.post("/api/task/create")
async def create_task(req: QuestionRequest, bg_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    # Initialize the task status
    tasks_db[task_id] = {
        "id": task_id,
        "status": "pending",  # pending | processing | completed | failed
        "timeline": [],
        "result": None
    }

    # Push the task to the background queue so it does not block the current request
    bg_tasks.add_task(run_rag_task, task_id, req.question)

    return {"task_id": task_id, "message": "Task created successfully"}


# Endpoint 2: get task status and details (Get Task Status)
@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]


# --- New: frontend-adapted API endpoints (Chat Flow) ---

@app.post("/api/v1/chat/question")
async def chat_question(req: ChatQuestionRequest):
    """Receive the user's question."""
    question_id = f"q_{uuid.uuid4()}"
    return {
        "success": True,
        "data": {
            "questionId": question_id,
            "conversationId": req.conversationId or f"conv_{uuid.uuid4()}",
            "message": req.message,
            "timestamp": datetime.now().isoformat(),
            "attachments": req.attachments or []
        },
        "message": "Question received"
    }

@app.post("/api/v1/chat/retrieve")
async def chat_retrieve(req: ChatRetrieveRequest):
    """Perform document retrieval."""
    if controller.rag_system and controller.rag_system.retriever:
        try:
            # Use the real retriever
            docs = controller.rag_system.retriever.invoke(req.query)
            results = []
            for i, doc in enumerate(docs):
                results.append({
                    "id": f"doc_{i}",
                    "content": doc.page_content,
                    "similarity": 0.0, # The real retriever may not return a score; default to 0
                    "source": doc.metadata.get("source", "Unknown source")
                })
            return {
                "success": True,
                "data": {
                    "questionId": req.questionId,
                    "results": results,
                    "totalResults": len(results),
                    "retrievalTime": 0.5 # Simulated time
                },
                "message": "Retrieval complete"
            }
        except Exception as e:
            print(f"Retrieval failed: {e}")
            # Fallback to mock
            pass

    # Mock data
    return {
        "success": True,
        "data": {
            "questionId": req.questionId,
            "results": [
                {"id": "mock_1", "content": "This is mock data: the backend RAG system is not ready.", "source": "System notice"},
                {"id": "mock_2", "content": "Please check whether the LLM service is running, or whether the PDF folder is empty.", "source": "System notice"}
            ],
            "totalResults": 2,
            "retrievalTime": 0.1
        },
        "message": "Retrieval complete (Mock)"
    }

@app.post("/api/v1/chat/generate")
async def chat_generate(req: ChatGenerateRequest):
    """Generate an answer."""
    if controller.rag_system and controller.rag_system.query_processor:
        try:
            # Build the context
            context = "\n\n".join([doc.get("content", "") for doc in req.retrievedDocs])

            # Use the real generator
            answer = controller.rag_system.query_processor.generate_answer(context, req.query)

            return {
                "success": True,
                "data": {
                    "questionId": req.questionId,
                    "answer": answer,
                    "generationTime": 1.0,
                    "confidence": 0.9,
                    "sources": [doc.get("source", "Unknown") for doc in req.retrievedDocs],
                    "thinking": []
                },
                "message": "Generation complete"
            }
        except Exception as e:
            print(f"Generation failed: {e}")
            pass

    # Mock data
    return {
        "success": True,
        "data": {
            "questionId": req.questionId,
            "answer": "This is a mock answer: the backend RAG system is not ready, so a real answer cannot be generated.\n\nPlease ensure:\n1. The LLM service is running at http://localhost:8000 (or modify the rag_system.py configuration)\n2. The PDF folder has been created and contains documents\n3. Dependencies are installed",
            "generationTime": 0.1,
            "confidence": 0.0,
            "sources": [],
            "thinking": []
        },
        "message": "Generation complete (Mock)"
    }

if __name__ == "__main__":
    import os
    import uvicorn
    # Port layout: vLLM (LLM) = 8000, RAG backend = 8001, frontend = 3000
    # The default port can be overridden with the RAG_BACKEND_PORT environment variable
    port = int(os.getenv("RAG_BACKEND_PORT", "8001"))
    # Start command: uvicorn api_server2:app --host 0.0.0.0 --port 8001
    uvicorn.run("api_server2:app", host="0.0.0.0", port=port, reload=True)
