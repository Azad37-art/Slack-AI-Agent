# main.py
# ─────────────────────────────────────────────────────────────
# FastAPI server + Slack Socket Mode bot — single entry point.
#
# When you run:  python main.py
#   Thread 1 → FastAPI on http://localhost:8000  (for API testing)
#   Thread 2 → Slack bot via WebSocket           (no ngrok needed)
#
# Both threads share the same agent — run_agent() is stateless
# so it is safe to call from both threads simultaneously.
#
# FastAPI Endpoints:
#   POST /chat          — send a message, get a response
#   POST /confirm       — confirm a pending action
#   GET  /data          — view all CSV data
#   GET  /health        — server health check
#   POST /rebuild-index — manually rebuild the RAG index
# ─────────────────────────────────────────────────────────────

import uuid
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from agent.graph_agent import run_agent
from agent.rag_pipeline import get_qa_chain, rebuild_index
from tools.csv_tools import get_all_rows, get_columns

# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart CSV Agent API",
    description="AI agent that answers questions and performs actions on CSV data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Session store (FastAPI side)
# Key   = session_id string (from the client)
# Value = pending_action dict (waiting for confirmation)
# ─────────────────────────────────────────────────────────────
session_store: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ConfirmRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    needs_confirmation: bool = False
    status: str = "ok"


# ─────────────────────────────────────────────────────────────
# Startup — build RAG index before accepting requests
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Smart CSV Agent...")
    print("🔧 Building RAG pipeline (30–60s on first run)...")
    get_qa_chain()
    print("✅ RAG pipeline ready.")

    # ── Start Slack bot in a background thread ────────────────
    # We import here (not at top) so that if SLACK tokens are
    # missing the FastAPI server still starts for local testing.
    try:
        from slack_bot import start_slack_bot
        slack_thread = threading.Thread(
            target=start_slack_bot,
            daemon=True,   # thread dies automatically when main process exits
            name="SlackBot",
        )
        slack_thread.start()
        print("✅ Slack bot started in background thread.")
    except Exception as e:
        print(f"⚠️  Slack bot did not start: {e}")
        print("    API is still available at http://localhost:8000")


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1: /chat
# ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send any message — question or command.

    Examples:
      "What is the email of Ahmad?"
      "Change Ali's email to ali@new.com"
      "Delete the record for Bilal"
      "Add new customer: John Doe, john@test.com, Male, Lahore, 500, 2024-01-01"

    Returns needs_confirmation=true if multiple records matched.
    Use /confirm to resolve.
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = run_agent(user_message=request.message)

        if result.get("needs_confirmation") and result.get("pending_action"):
            session_store[session_id] = result["pending_action"]
            return ChatResponse(
                session_id=session_id,
                answer=result["answer"],
                needs_confirmation=True,
                status="waiting_confirmation",
            )

        return ChatResponse(
            session_id=session_id,
            answer=result["answer"],
            needs_confirmation=False,
            status="ok",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2: /confirm
# ─────────────────────────────────────────────────────────────
@app.post("/confirm", response_model=ChatResponse)
async def confirm_action(request: ConfirmRequest):
    """
    Reply to a confirmation prompt.
    Pass the same session_id returned by /chat.
    Message can be: "Row 3", "3", "Ali Khan", "yes"
    """
    pending_action = session_store.get(request.session_id)
    if not pending_action:
        raise HTTPException(
            status_code=404,
            detail=f"No pending action for session '{request.session_id}'. Start a new /chat.",
        )

    try:
        result = run_agent(
            user_message=request.message,
            pending_action=pending_action,
        )

        if not result.get("needs_confirmation"):
            session_store.pop(request.session_id, None)

        return ChatResponse(
            session_id=request.session_id,
            answer=result["answer"],
            needs_confirmation=result.get("needs_confirmation", False),
            status="ok" if not result.get("needs_confirmation") else "waiting_confirmation",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Confirm error: {str(e)}")


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3: /data
# ─────────────────────────────────────────────────────────────
@app.get("/data")
async def get_data():
    """Return all rows currently in the CSV."""
    result = get_all_rows()
    return {
        "columns":    get_columns()["columns"],
        "total_rows": result["total_rows"],
        "data":       result["data"],
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 4: /rebuild-index
# ─────────────────────────────────────────────────────────────
@app.post("/rebuild-index")
async def trigger_rebuild():
    """Manually rebuild the RAG index (e.g. after editing CSV externally)."""
    rebuild_index()
    return {"status": "ok", "message": "RAG index rebuilt."}


# ─────────────────────────────────────────────────────────────
# ENDPOINT 5: /health
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "active_sessions": len(session_store),
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # reload=False is required when running Slack bot in a thread.
    # uvicorn --reload spawns child processes which breaks threading.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)