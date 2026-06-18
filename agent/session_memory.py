# agent/session_memory.py

from datetime import datetime

# Local memory only. It resets when app restarts.
SESSION_MEMORY = {}


def get_session_id(user_id: str, channel_id: str) -> str:
    """
    One memory per Slack user per channel.
    """
    return f"{channel_id}:{user_id}"


def get_memory(session_id: str) -> dict:
    """
    Return memory for this Slack session.
    """
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = {
            "history": [],
            "last_person": None,
            "last_row_id": None,
            "last_email": None,
            "last_action": None,
            "pending_action": None,
            "updated_at": None,
        }

    return SESSION_MEMORY[session_id]


def update_memory(session_id: str, user_message: str, agent_answer: str, result: dict = None):
    """
    Save last conversation turn.
    """
    memory = get_memory(session_id)

    memory["history"].append({
        "user": user_message,
        "assistant": agent_answer,
        "time": datetime.utcnow().isoformat(),
    })

    # Keep only last 6 turns
    memory["history"] = memory["history"][-6:]

    if result:
        if result.get("pending_action") is not None:
            memory["pending_action"] = result.get("pending_action")

        if result.get("needs_confirmation") is False:
            memory["pending_action"] = None

    memory["updated_at"] = datetime.utcnow().isoformat()


def set_last_entity(session_id: str, row_id=None, person=None, email=None):
    """
    Save last person/row/email discussed.
    """
    memory = get_memory(session_id)

    if row_id is not None:
        memory["last_row_id"] = row_id

    if person is not None:
        memory["last_person"] = person

    if email is not None:
        memory["last_email"] = email

    memory["updated_at"] = datetime.utcnow().isoformat()


def memory_to_text(memory: dict) -> str:
    """
    Convert memory into text for the LLM prompt.
    """
    if not memory:
        return "No previous memory."

    lines = []

    if memory.get("last_person"):
        lines.append(f"Last discussed person: {memory['last_person']}")

    if memory.get("last_row_id"):
        lines.append(f"Last discussed Row ID: {memory['last_row_id']}")

    if memory.get("last_email"):
        lines.append(f"Last discussed email: {memory['last_email']}")

    if memory.get("history"):
        lines.append("Recent conversation:")
        for turn in memory["history"][-3:]:
            lines.append(f"User: {turn.get('user')}")
            lines.append(f"Assistant: {turn.get('assistant')}")

    return "\n".join(lines) if lines else "No previous memory."


def clear_memory(session_id: str):
    SESSION_MEMORY.pop(session_id, None)