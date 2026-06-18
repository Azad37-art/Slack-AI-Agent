# slack_bot.py
# ─────────────────────────────────────────────────────────────
# Slack Socket Mode bot — no ngrok, no public URL needed.
#
# How Socket Mode works:
#   Your app opens a WebSocket connection TO Slack's servers.
#   Slack pushes events (messages) through that connection.
#   You respond back through the same socket.
#   Nothing needs to be open/exposed on your machine.
#
# Flow:
#   User sends message in Slack
#       ↓
#   Slack pushes event to your WebSocket
#       ↓
#   handle_message() receives it
#       ↓
#   Calls run_agent() (same agent as FastAPI)
#       ↓
#   If needs_confirmation → stores pending_action in session dict
#                           keyed by Slack user_id + channel
#       ↓
#   Sends reply back to Slack
# ─────────────────────────────────────────────────────────────

import os
import asyncio
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent.graph_agent import run_agent
from agent.session_memory import (
    get_session_id,
    get_memory,
    memory_to_text,
    update_memory,
    clear_memory,
)

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")   # must start with xapp-

if not SLACK_BOT_TOKEN:
    raise ValueError("SLACK_BOT_TOKEN is missing from your .env file")
if not SLACK_APP_TOKEN:
    raise ValueError("SLACK_APP_TOKEN is missing from your .env file")
if not SLACK_APP_TOKEN.startswith("xapp-"):
    raise ValueError("SLACK_APP_TOKEN must start with 'xapp-' — make sure you copied the App-Level Token")


# ─────────────────────────────────────────────────────────────
# Slack App instance
# ─────────────────────────────────────────────────────────────
app = App(token=SLACK_BOT_TOKEN)


# ─────────────────────────────────────────────────────────────
# Session store
# Keyed by  "{user_id}:{channel_id}"
# Value  =  the pending_action dict (waiting for user to confirm)
#
# When the agent finds 2+ matching records it stores the pending
# action here. The user's next message in the same channel
# is treated as their confirmation reply.
# ─────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}

def _session_key(user_id: str, channel: str) -> str:
    return f"{user_id}:{channel}"


# ─────────────────────────────────────────────────────────────
# Main message handler
# Triggered on every message the bot can see:
#   - Direct messages (DMs) to the bot
#   - Messages in channels where the bot is added (only when @mentioned)
# ─────────────────────────────────────────────────────────────
@app.event("message")
def handle_message(event, say, client):
    """
    Handle incoming Slack messages and reply with the agent's answer.

    'say'    → sends a message back to the same channel/DM
    'client' → full Slack Web API client (for advanced formatting)
    'event'  → the raw Slack event dict
    """
    # ── Ignore messages from bots (including ourselves) ──────
    # Without this the bot would respond to its own replies
    # and create an infinite loop.
    if event.get("bot_id"):
        return
    if event.get("subtype") == "bot_message":
        return

    user_id  = event.get("user", "")
    channel  = event.get("channel", "")
    text     = event.get("text", "").strip()
    thread   = event.get("thread_ts") or event.get("ts")  # reply in same thread

    if not text:
        return

    # ── Remove @mention from text if present ─────────────────
    # When user types "@BotName change Ali's email..." Slack sends
    # "<@U12345> change Ali's email..." — we strip the mention part.
    import re
    text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
    if not text:
        say(text="Yes? How can I help you?", thread_ts=thread)
        return

    # ── Show typing indicator ─────────────────────────────────
    # This is optional but makes the bot feel responsive
    try:
        client.reactions_add(channel=channel, name="thinking_face", timestamp=event["ts"])
    except Exception:
        pass  # if it fails (e.g. no permission), just continue

    # ── Check if there is a pending confirmation for this user ─
    key            = _session_key(user_id, channel)
    pending_action = _sessions.get(key)

    session_id = get_session_id(user_id, channel)
    memory = get_memory(session_id)
    memory_context = memory_to_text(memory)

    try:
        result = run_agent(
            user_message=text,
            pending_action=pending_action,
            memory_context=memory_context,
            session_id=session_id,
        )

        answer = result.get("answer", "Sorry, something went wrong.")
        needs_confirmation = result.get("needs_confirmation", False)
        new_pending = result.get("pending_action")

        # Save short-term memory after every answer
        update_memory(
            session_id=session_id,
            user_message=text,
            agent_answer=answer,
            result=result,
        )

        # Store or clear pending action session
        if needs_confirmation and new_pending:
            _sessions[key] = new_pending
        else:
            _sessions.pop(key, None)

        say(text=answer, thread_ts=thread)

    except Exception as e:
        say(text=f"❌ Something went wrong: {str(e)}", thread_ts=thread)

    finally:
        # Remove the thinking emoji
        try:
            client.reactions_remove(channel=channel, name="thinking_face", timestamp=event["ts"])
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Also handle app_mention events
# This fires when someone @mentions the bot in a channel.
# We re-use the same handle_message logic.
# ─────────────────────────────────────────────────────────────
@app.event("app_mention")
def handle_mention(event, say, client):
    """
    Fires when someone @mentions the bot in a public channel.
    Delegates to the same logic as direct messages.
    """
    handle_message(event, say, client)


# ─────────────────────────────────────────────────────────────
# Start function — called from main.py
# ─────────────────────────────────────────────────────────────
def start_slack_bot():
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.connect()
    print("Bolt app is running!")
