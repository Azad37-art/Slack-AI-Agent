# agent/graph_agent.py
# ─────────────────────────────────────────────────────────────
# LangGraph agent — intent routing, multi-intent handling,
# name matching, confirmation flow.
#
# KEY CHANGE — Multi-intent support:
#   When user sends something like:
#     "what is the email of Susy and change it to susy@new.com"
#   The old classifier forced a single intent and got confused.
#
#   Now the classifier returns THREE possible intents:
#     "question" → only a question
#     "action"   → only a command
#     "both"     → question + action in same message
#
#   For "both", a new node (multi_intent_handler) splits the
#   message into its question part and action part, runs the
#   RAG answer first, then runs the action — and combines both
#   results into one clean reply.
# ─────────────────────────────────────────────────────────────

import json
import re
from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from config import GROQ_API_KEY, GROQ_MODEL
from tools.csv_tools import (
    find_similar_rows,
    update_field,
    add_row,
    delete_row,
    get_columns,
    load_csv,
)
from agent.rag_pipeline import get_qa_chain


EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"


def normalize_slack_mailto_text(text: str) -> str:
    """
    Convert Slack mailto formatting into normal email text.

    Examples:
    <mailto:test@gmail.com|test@gmail.com> -> test@gmail.com
    mailto:test@gmail.com|test@gmail.com   -> test@gmail.com
    """
    if not text:
        return ""

    text = str(text)

    # Slack with angle brackets
    text = re.sub(r"<mailto:([^|>]+)\|[^>]+>", r"\1", text)

    # Slack without angle brackets
    text = re.sub(r"mailto:([^|\s>]+)\|[^\s>]+", r"\1", text)

    # Slack simple mailto
    text = re.sub(r"<mailto:([^>]+)>", r"\1", text)

    return text


def unique_keep_order(items):
    seen = set()
    result = []

    for item in items:
        item = item.strip()

        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)

    return result


def detect_email_update_command(user_message: str) -> dict | None:
    """
    Detect direct email replacement commands.

    Example:
    please change this email mailto:old@gmail.com|old@gmail.com into mailto:new@gmail.com|new@gmail.com
    """
    text = normalize_slack_mailto_text(user_message)
    emails = unique_keep_order(re.findall(EMAIL_RE, text))

    if len(emails) >= 2:
        lower = text.lower()

        action_words = [
            "change",
            "update",
            "edit",
            "replace",
            "set",
            "modify",
        ]

        if any(word in lower for word in action_words):
            return {
                "operation": "update",
                "target_name": emails[0],
                "field": "email",
                "value": emails[-1],
            }

    return None
# ─────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_message: str
    intent: Optional[str]
    action_plan: Optional[dict]
    similar_rows: Optional[list]
    needs_confirmation: Optional[bool]
    confirmation_message: Optional[str]
    pending_action: Optional[dict]
    final_answer: Optional[str]
    question_part: Optional[str]
    action_part: Optional[str]
    memory_context: Optional[str]
    session_id: Optional[str]
    error: Optional[str]


def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0)


# ─────────────────────────────────────────────────────────────
# NODE 1 — Intent Classifier
# Now detects THREE intents: question / action / both
# ─────────────────────────────────────────────────────────────
@traceable(name="Intent Classifier", run_type="chain")
def intent_classifier(state: AgentState) -> AgentState:
    user_message = state.get("user_message", "")
    memory_context = state.get("memory_context") or "No previous memory."

    direct_action = detect_email_update_command(user_message)

    if direct_action:
        return {
            **state,
            "intent": "action",
            "action_plan": direct_action,
        }

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an intent classifier for a spreadsheet assistant.

Classify the user message as exactly one of:
- "question" : user ONLY wants to read, find, show, or ask information
- "action"   : user ONLY wants to add, update, change, edit, remove, or delete data
- "both"     : user wants to BOTH ask a question AND perform an action in the same message

Conversation memory:
{memory_context}

Use memory to understand:
- him
- her
- his
- that person
- same person
- that email

Examples of "both":
- what is the email of Ali and change it to ali@new.com
- show me Susy's details and update her city to Karachi
- what is Brook's phone number? also delete his record
- tell me the price for order 5 and change it to 99

Return ONLY valid JSON.

JSON format:
{{
  "intent": "question"
}}

or

{{
  "intent": "action"
}}

or

{{
  "intent": "both"
}}
"""
        ),
        ("human", "{user_message}"),
    ])

    chain = prompt | llm | StrOutputParser()

    raw = chain.invoke({
        "user_message": user_message,
        "memory_context": memory_context,
    })

    try:
        result = extract_json(raw)
        intent = result.get("intent", "question")

        if intent not in ("question", "action", "both"):
            intent = "question"

    except Exception:
        intent = "question"

    return {
        **state,
        "intent": intent,
    }


# ─────────────────────────────────────────────────────────────
# NODE 2 — Multi-intent handler
# Splits the message, answers the question, then runs the action.
# Combines both results into one reply.
# ─────────────────────────────────────────────────────────────
@traceable(name="Multi Intent Handler", run_type="chain")
def multi_intent_handler(state: AgentState) -> AgentState:
    user_message = state.get("user_message", "")
    memory_context = state.get("memory_context") or "No previous memory."

    df = load_csv()
    columns = list(df.columns)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You split a user message into two parts:
1. question_part
2. action_part

Conversation memory:
{memory_context}

Available CSV columns:
{columns}

The user may ask a question and request an update/delete/add in one message.

Examples:

User:
what is the email of Ali and change his city to Lahore

Return:
{{
  "question_part": "what is the email of Ali",
  "action_part": "change Ali city to Lahore"
}}

User:
show me the price of tmccrone9@rediff.com and change email to new@gmail.com

Return:
{{
  "question_part": "show me the price of tmccrone9@rediff.com",
  "action_part": "change tmccrone9@rediff.com email to new@gmail.com"
}}

Return ONLY valid JSON.
"""
        ),
        ("human", "{user_message}"),
    ])

    chain = prompt | llm | StrOutputParser()

    raw = chain.invoke({
        "user_message": user_message,
        "memory_context": memory_context,
        "columns": columns,
    })

    parsed = extract_json(raw)

    question_part = parsed.get("question_part", "").strip()
    action_part = parsed.get("action_part", "").strip()

    if not question_part and not action_part:
        return {
            **state,
            "final_answer": "I could not split the question and action clearly.",
        }

    question_answer = ""

    if question_part:
        qa_chain = get_qa_chain()

        query = f"""
Conversation memory:
{memory_context}

Current question:
{question_part}
"""

        question_answer = qa_chain.invoke(query)

    if action_part:
        action_state = action_planner({
            **state,
            "user_message": action_part,
            "memory_context": memory_context,
        })

        action_state = similarity_checker(action_state)

        if action_state.get("needs_confirmation"):
            final = question_answer + "\n\n" + action_state.get("final_answer", "")
            return {
                **action_state,
                "final_answer": final.strip(),
            }

        action_state = execute_action(action_state)

        final = question_answer + "\n\n" + action_state.get("final_answer", "")

        return {
            **action_state,
            "final_answer": final.strip(),
        }

    return {
        **state,
        "final_answer": question_answer,
    }


# ─────────────────────────────────────────────────────────────
# NODE 2a — RAG Answer (question only)
# ─────────────────────────────────────────────────────────────
def resolve_followup_question(user_message: str, memory_context: str) -> str:
    """
    Replace pronouns using short-term memory.
    """
    if not memory_context or memory_context == "No previous memory.":
        return user_message

    last_person = None
    last_email = None

    for line in memory_context.splitlines():
        if line.lower().startswith("last discussed person:"):
            last_person = line.split(":", 1)[1].strip()

        if line.lower().startswith("last discussed email:"):
            last_email = line.split(":", 1)[1].strip()

    resolved = user_message

    if last_person:
        resolved = re.sub(
            r"\b(him|her|his|same person|that person)\b",
            last_person,
            resolved,
            flags=re.IGNORECASE,
        )

    if last_email:
        resolved = re.sub(
            r"\b(that email|this email|same email)\b",
            last_email,
            resolved,
            flags=re.IGNORECASE,
        )

    return resolved

#------------------------------------------
def resolve_followup_question(user_message: str, memory_context: str) -> str:
    """
    Replace pronouns using short-term memory.
    """
    if not memory_context or memory_context == "No previous memory.":
        return user_message

    last_person = None
    last_email = None

    for line in memory_context.splitlines():
        if line.lower().startswith("last discussed person:"):
            last_person = line.split(":", 1)[1].strip()

        if line.lower().startswith("last discussed email:"):
            last_email = line.split(":", 1)[1].strip()

    resolved = user_message

    if last_person:
        resolved = re.sub(
            r"\b(him|her|his|same person|that person)\b",
            last_person,
            resolved,
            flags=re.IGNORECASE,
        )

    if last_email:
        resolved = re.sub(
            r"\b(that email|this email|same email)\b",
            last_email,
            resolved,
            flags=re.IGNORECASE,
        )

    return resolved


@traceable(name="RAG Answer", run_type="chain")
def rag_answer(state: AgentState) -> AgentState:
    user_message = state.get("user_message", "")
    memory_context = state.get("memory_context") or "No previous memory."

    try:
        qa_chain = get_qa_chain()

        resolved_question = resolve_followup_question(user_message, memory_context)

        query = f"""
Conversation memory:
{memory_context}

Current user question:
{resolved_question}
"""

        answer = qa_chain.invoke(query)

    except Exception as e:
        answer = f"❌ Error while answering from RAG: {str(e)}"

    return {
        **state,
        "final_answer": answer,
    }

# ─────────────────────────────────────────────────────────────
# NODE 2b — Action Planner (action only)
# ─────────────────────────────────────────────────────────────
@traceable(name="Action Planner", run_type="chain")
def action_planner(state: AgentState) -> AgentState:
    user_message = state.get("user_message", "")
    memory_context = state.get("memory_context") or "No previous memory."

    direct_action = detect_email_update_command(user_message)

    if direct_action:
        return {
            **state,
            "action_plan": direct_action,
        }

    df = load_csv()
    columns = list(df.columns)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You convert the user's message into a CSV/Google Sheet action plan.

Conversation memory:
{memory_context}

Use memory when user says:
- his email
- her city
- same person
- that row
- that email

Available CSV columns:
{columns}

Supported operations:

1. Update existing row:
{{
  "operation": "update",
  "target_name": "person name, email, Row number, or remembered person",
  "field": "column name",
  "value": "new value"
}}

2. Delete existing row:
{{
  "operation": "delete",
  "target_name": "person name, email, Row number, or remembered person"
}}

3. Add new row:
{{
  "operation": "add",
  "row_data": {{
    "first_name": "...",
    "last_name": "...",
    "email": "...",
    "gender": "...",
    "city": "...",
    "price": "...",
    "order_date": "..."
  }}
}}

Rules:
- If user says "change old@email.com to new@email.com", use:
  target_name = old@email.com
  field = email
  value = new@email.com

- If user says "remove email", "clear email", or "delete email", use:
  operation = update
  field = email
  value = ""

- Do not create fake column names.
- Use only available columns.
- Return ONLY valid JSON.
"""
        ),
        ("human", "{user_message}"),
    ])

    chain = prompt | llm | StrOutputParser()

    raw = chain.invoke({
        "user_message": user_message,
        "memory_context": memory_context,
        "columns": columns,
    })

    action_plan = extract_json(raw)

    if not action_plan or "operation" not in action_plan:
        return {
            **state,
            "final_answer": "I could not understand the action. Please say clearly what you want to update, delete, or add.",
            "pending_action": None,
            "needs_confirmation": False,
        }

    return {
        **state,
        "action_plan": action_plan,
    }

# ─────────────────────────────────────────────────────────────
# NODE 3 — Similarity Checker
# ─────────────────────────────────────────────────────────────
@traceable(name="Similarity Checker", run_type="chain")
def similarity_checker(state: AgentState) -> AgentState:
    action_plan = state.get("action_plan") or {}
    operation = action_plan.get("operation", "unclear")

    if operation == "add":
        return {**state, "needs_confirmation": False, "pending_action": action_plan}

    if operation == "unclear":
        return {
            **state,
            "needs_confirmation": False,
            "final_answer": action_plan.get("message", "I couldn't understand that. Could you rephrase?"),
        }

    target_name = action_plan.get("target_name", "").strip()
    if not target_name:
        return {
            **state,
            "needs_confirmation": False,
            "final_answer": "I could not identify who you want me to act on. Please mention a name.",
        }

    df      = load_csv()
    matched = find_similar_rows(df, target_name)
    rows    = matched.to_dict(orient="records")

    if len(rows) == 0:
        return {
            **state,
            "needs_confirmation": False,
            "final_answer": f"I couldn't find anyone matching '{target_name}'. Please check the spelling.",
        }

    if len(rows) == 1:
        row          = rows[0]
        updated_plan = {**action_plan, "confirmed_row_id": row.get("Row"), "confirmed_row": row}
        return {**state, "needs_confirmation": False, "similar_rows": rows, "pending_action": updated_plan}

    # Multiple matches — always confirm
    lines = []
    for r in rows:
        lines.append(
            f"  Row {r.get('Row','?')}: {r.get('first_name','')} {r.get('last_name','')} "
            f"| email: {r.get('email','N/A')} | city: {r.get('city','N/A')} | gender: {r.get('gender','N/A')}"
        )
    confirm_msg = (
        f"I found {len(rows)} records matching '{target_name}':\n\n"
        + "\n".join(lines)
        + "\n\nWhich one did you mean? Reply with the Row number (e.g. 'Row 3') or full name."
    )
    return {
        **state,
        "needs_confirmation":   True,
        "similar_rows":         rows,
        "pending_action":       action_plan,
        "confirmation_message": confirm_msg,
        "final_answer":         confirm_msg,
    }


# ─────────────────────────────────────────────────────────────
# NODE 4 — Execute Action
# ─────────────────────────────────────────────────────────────
@traceable(name="Execute CSV Action", run_type="tool")
def execute_action(state: AgentState) -> AgentState:
    action = state.get("pending_action") or {}
    operation = action.get("operation")

    try:
        if operation in ("update", "delete"):
            row_id = action.get("confirmed_row_id")

            if row_id is None:
                return {
                    **state,
                    "final_answer": "❌ Please reply with the Row number first, for example: Row 3",
                }

        if operation == "update":
            field = action.get("field")
            value = action.get("value", "")

            if not field:
                result = {
                    "success": False,
                    "error": "No field/column was identified for update.",
                }
            else:
                result = update_field(
                    action.get("confirmed_row_id"),
                    field,
                    value,
                )

        elif operation == "delete":
            result = delete_row(action.get("confirmed_row_id"))

        elif operation == "add":
            result = add_row(action.get("row_data", {}))

        else:
            result = {
                "success": False,
                "error": f"Unknown operation: {operation}",
            }

        answer = (
            f"✅ Done! {result.get('message', 'Action completed.')}"
            if result.get("success")
            else f"❌ Action failed: {result.get('error', 'Unknown error')}"
        )

    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    return {**state, "final_answer": answer}

# ─────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────
def route_after_intent(state: AgentState) -> Literal["rag_answer", "action_planner", "multi_intent_handler"]:
    intent = state.get("intent")
    if intent == "both":
        return "multi_intent_handler"
    if intent == "action":
        return "action_planner"
    return "rag_answer"


def route_after_similarity(state: AgentState) -> Literal["execute_action", "end"]:
    if state.get("needs_confirmation"):
        return "end"

    if state.get("final_answer"):
        return "end"

    pending_action = state.get("pending_action") or {}

    if pending_action.get("operation") in ("update", "delete", "add"):
        return "execute_action"

    return "end"


# ─────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────
def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier",    intent_classifier)
    graph.add_node("rag_answer",           rag_answer)
    graph.add_node("action_planner",       action_planner)
    graph.add_node("multi_intent_handler", multi_intent_handler)
    graph.add_node("similarity_checker",   similarity_checker)
    graph.add_node("execute_action",       execute_action)

    graph.set_entry_point("intent_classifier")

    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "rag_answer":           "rag_answer",
            "action_planner":       "action_planner",
            "multi_intent_handler": "multi_intent_handler",
        },
    )

    # Single-intent paths
    graph.add_edge("rag_answer",     END)
    graph.add_edge("action_planner", "similarity_checker")

    graph.add_conditional_edges(
        "similarity_checker",
        route_after_similarity,
        {"execute_action": "execute_action", "end": END},
    )
    graph.add_edge("execute_action", END)

    # Multi-intent path — handler does everything internally, goes straight to END
    graph.add_edge("multi_intent_handler", END)

    return graph.compile()


_agent_graph = None

def get_agent():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
@traceable(name="Slack CSV Agent", run_type="chain")
def run_agent(
    user_message: str,
    pending_action: dict = None,
    memory_context: str = None,
    session_id: str = None,
) -> dict:
    """
    Public entry point used by Slack/FastAPI.

    Handles:
    1. Confirmation replies
    2. Normal question/action/both messages
    3. Short-term memory context
    """

    memory_context = memory_context or "No previous memory."

    # ── Confirmation reply ────────────────────────────────────
    if pending_action:
        user_reply = user_message.strip().lower()
        is_yes = user_reply in (
            "yes", "yeah", "yep", "ok", "okay",
            "sure", "go ahead", "confirm", "proceed"
        )

        if not is_yes:
            df = load_csv()

            id_match = re.search(
                r"\b(?:row\s*)?(\d+)\b",
                user_message,
                re.IGNORECASE,
            )

            if id_match:
                row_id = int(id_match.group(1))
                id_col = "Row" if "Row" in df.columns else df.columns[0]
                row_mask = df[id_col].astype(str) == str(row_id)

                if row_mask.any():
                    row = df[row_mask].to_dict(orient="records")[0]
                    pending_action["confirmed_row_id"] = row_id
                    pending_action["confirmed_row"] = row

            else:
                name_matches = find_similar_rows(df, user_message)

                if not name_matches.empty:
                    best_row = None
                    best_len = 0

                    for _, r in name_matches.iterrows():
                        full = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip().lower()

                        overlap = sum(
                            1 for w in user_reply.split()
                            if w and w in full
                        )

                        if overlap > best_len:
                            best_len = overlap
                            best_row = r.to_dict()

                    if best_row:
                        pending_action["confirmed_row_id"] = best_row.get("Row")
                        pending_action["confirmed_row"] = best_row

        if (
            pending_action.get("confirmed_row_id") is None
            and pending_action.get("operation") in ("update", "delete")
        ):
            return {
                "answer": "❌ I could not identify the row. Please reply with the Row number, for example: Row 3",
                "needs_confirmation": True,
                "pending_action": pending_action,
            }

        result_state = execute_action({
            "user_message": user_message,
            "intent": "action",
            "action_plan": None,
            "similar_rows": None,
            "needs_confirmation": False,
            "confirmation_message": None,
            "pending_action": pending_action,
            "final_answer": None,
            "question_part": None,
            "action_part": None,
            "memory_context": memory_context,
            "session_id": session_id,
            "error": None,
        })

        return {
            "answer": result_state.get("final_answer", "Done."),
            "needs_confirmation": False,
            "pending_action": None,
        }

    # ── Normal flow: question / action / both ─────────────────
    result = get_agent().invoke({
        "user_message": user_message,
        "intent": None,
        "action_plan": None,
        "similar_rows": None,
        "needs_confirmation": None,
        "confirmation_message": None,
        "pending_action": None,
        "final_answer": None,
        "question_part": None,
        "action_part": None,
        "memory_context": memory_context,
        "session_id": session_id,
        "error": None,
    })

    if result is None:
        return {
            "answer": "❌ Agent returned no result.",
            "needs_confirmation": False,
            "pending_action": None,
        }

    return {
        "answer": result.get("final_answer", "I could not process that request."),
        "needs_confirmation": result.get("needs_confirmation", False),
        "pending_action": result.get("pending_action") if result.get("needs_confirmation") else None,
    }