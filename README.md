
# AI Agent for Q&A and Live Data Actions

An AI-powered Slack agent that can answer questions from spreadsheet data, perform live data updates, remember short-term conversation context, and trace every step using LangSmith.

This project shows a real business workflow where users can ask questions and update spreadsheet records directly from chat.

## Project Overview

This project is a Slack-based AI agent for spreadsheet automation.

The agent allows users to interact with CSV or Google Sheets data using natural language. Users can ask questions, update fields, change emails, check prices, get order dates, and use follow-up questions like “what is his email?” or “what is the price of him?”

The system is designed to work like a real AI assistant for business data.

---

## Key Features

* Ask questions from spreadsheet data inside Slack
* Update spreadsheet records using natural language
* Change fields like email, price, city, and order date
* Support short-term memory for follow-up questions
* Search data using RAG and ChromaDB
* Update only the changed row in the vector index
* Connect with CSV or Google Sheets
* Trace agent workflow using LangSmith
* Monitor LLM calls, cost, tokens, latency, errors, and outputs
* Ready to connect with other platforms through APIs

---

## Example Questions and Commands

### Question Answering

```text
@test_agent what is the email of Greggory Soutar?
```

```text
@test_agent what is the price of Torry McCrone?
```

```text
@test_agent what is the order date of Emmott Mack?
```

### Short-Term Memory

```text
@test_agent what is the order date of Emmott Mack?
```

```text
@test_agent what is his email?
```

The agent understands that “his” refers to the last discussed person.

### Live Data Action

```text
@test_agent please change this email susy5577@gmail.com into newsusy1010@gmail.com
```

The agent finds the correct row, updates the spreadsheet, and confirms the result in Slack.

---

## Screenshots

### Slack Q&A and Live Actions

![Slack AI Agent Demo](assets/slack-agent-demo.png)

The agent answers spreadsheet questions directly inside Slack and performs real update actions using natural language.

---

### LangSmith Tracing

![LangSmith Trace](assets/langsmith-tracing.png)

LangSmith shows the full agent workflow, including intent classification, RAG retrieval, LLM calls, tool execution, inputs, outputs, errors, tokens, cost, and latency.

---

## How It Works

```text
Slack User Message
        ↓
LangGraph Agent
        ↓
Intent Classifier
        ↓
Question / Action / Both Router
        ↓
RAG Retrieval or Spreadsheet Action
        ↓
CSV / Google Sheets Update
        ↓
ChromaDB Vector Index Update
        ↓
LangSmith Trace
        ↓
Slack Response
```

---

## Main Workflow

1. User sends a message in Slack.
2. The agent detects whether the message is a question, an action, or both.
3. For questions, the agent retrieves the correct spreadsheet row using RAG.
4. For actions, the agent finds the matching row and updates the spreadsheet.
5. The vector index is updated incrementally after data changes.
6. The agent sends a clean response back to Slack.
7. LangSmith records the complete trace for debugging and monitoring.

---

## Tech Stack

* Python
* Slack Bolt
* LangChain
* LangGraph
* ChromaDB
* Google Sheets API
* Pandas
* LangSmith
* Groq LLM
* RAG
* Short-term memory

---

## LangSmith Observability

LangSmith is used to trace and monitor the full AI agent workflow.

It helps inspect:

* User input
* Intent classification
* RAG retrieval
* Retrieved spreadsheet rows
* LLM calls
* Tool execution
* Final answer
* Errors
* Token usage
* Cost
* Latency

This makes the agent easier to debug, improve, and prepare for real-world use.

---

## Short-Term Memory

The agent includes short-term memory for natural follow-up questions.

Example:

```text
User: what is the order date of Emmott Mack?
Agent: The order date of Emmott Mack is 3/10/2026.

User: what is his email?
Agent: The email of Emmott Mack is emack4x@sakura.ne.jp.
```

This makes the conversation feel more natural and useful.

---

## Live Spreadsheet Updates

The agent can update spreadsheet data directly from Slack.

Example:

```text
User: please change this email old@email.com into new@email.com
Agent: Done! Updated Row 12: email changed from old@email.com to new@email.com
```

The update is reflected in the connected spreadsheet.

---

## Use Cases

This type of AI agent can be used for:

* Sales data management
* Customer records
* Internal business automation
* CRM workflows
* Order tracking
* Team support tools
* Data lookup and update tasks
* Slack-based operations
* Google Sheets automation

---

## Future Improvements

* Add role-based permissions
* Add undo for last action
* Add audit logs for all changes
* Add stronger validation for emails, dates, and prices
* Connect with more platforms like WhatsApp, Telegram, Discord, Instagram, CRMs, and websites
* Deploy the agent on AWS or another cloud platform

---

## Project Goal

The goal of this project was to build a real AI agent that can do more than chat.

It can answer questions, perform actions, update live data, remember short-term context, and provide full tracing for debugging and monitoring.

This project demonstrates how AI agents can help businesses automate daily spreadsheet and data workflows directly from chat platforms.
