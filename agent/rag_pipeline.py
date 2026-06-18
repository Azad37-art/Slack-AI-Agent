# agent/rag_pipeline.py
# ─────────────────────────────────────────────────────────────
# RAG pipeline using LCEL — Windows-compatible rebuild strategy
#
# THE REAL FIX FOR WinError 32:
#   Do NOT delete the chroma_db folder at all.
#   Instead use a fixed collection name and call
#   chroma_client.delete_collection() + recreate it.
#   This clears all embeddings in-place without touching
#   any files on disk — so Windows file locks are never an issue.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import chromadb

from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from tools.csv_tools import load_csv

from config import (
    #CSV_FILE_PATH,
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DIR,
)

# Fixed collection name — we always use this name so we can
# delete and recreate it without touching any folder on disk.
COLLECTION_NAME = "csv_agent_data"

# Module-level cache
_chain_cache = None
_embeddings  = None   # reuse the same embedding model — expensive to reload


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load CSV → Documents
# ─────────────────────────────────────────────────────────────
def load_csv_as_documents():
    """
    Read Google Sheet/CSV and turn every row into a LangChain Document.
    """
    df = load_csv()

    def row_to_text(row):
        parts = [f"{col}: {row[col]}" for col in df.columns]

        if "first_name" in df.columns and "last_name" in df.columns:
            full = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
            insert_at = next(
                (i for i, p in enumerate(parts) if p.startswith("first_name:")), 1
            )
            parts.insert(insert_at + 1, f"full_name: {full}")

        return " | ".join(parts)

    df["row_text"] = df.apply(row_to_text, axis=1)
    loader = DataFrameLoader(df, page_content_column="row_text")
    return loader.load()

# ─────────────────────────────────────────────────────────────
# STEP 2 — Chunking
# ─────────────────────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────────────────────
# STEP 3 — Get shared embedding model
# Load once, reuse forever — saves ~5s on every rebuild.
# ─────────────────────────────────────────────────────────────
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("📦 Loading embedding model (once)...")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("✅ Embedding model loaded.")
    return _embeddings


# ─────────────────────────────────────────────────────────────
# STEP 4 — Build fresh Chroma vector store
#
# THE KEY CHANGE — no folder deletion:
#   We use a persistent chromadb.Client pointed at CHROMA_DIR.
#   To "reset" we just delete the named collection and recreate it.
#   The folder and its file handles stay open — Windows is happy.
# ─────────────────────────────────────────────────────────────
def build_vector_store(docs):
    """
    Clear the Chroma collection and repopulate it from fresh CSV row documents.
    One CSV row = one Chroma document.
    """
    embeddings = get_embeddings()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Cleared old collection '{COLLECTION_NAME}'")

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    ids = []
    for doc in docs:
        row_id = (
            doc.metadata.get("Row")
            or doc.metadata.get("row_id")
            or doc.metadata.get("index")
        )
        ids.append(get_row_doc_id(row_id))

    vector_store.add_documents(docs, ids=ids)

    print(f"✅ Chroma collection rebuilt — {len(docs)} rows embedded.")
    return vector_store

# ─────────────────────────────────────────────────────────────
# STEP 5 — Build the LCEL chain
#
# Flow:
#   question (str)
#     → {"context": retriever | _format_docs,
#        "question": RunnablePassthrough()}
#     → ChatPromptTemplate
#     → ChatGroq
#     → StrOutputParser()
#     → answer (str)
# ─────────────────────────────────────────────────────────────
def _format_docs(docs) -> str:
    """Join retrieved Documents into one text block."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_lcel_chain(vector_store):
    llm       = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant that answers questions about \
customer and order data stored in a spreadsheet.

The spreadsheet columns are:
Row, first_name, last_name, email, gender, city, price, order_date

Rules:
- Answer using ONLY the rows provided in the context below.
- Be concise and direct.
- If multiple records match, list all of them clearly.
- If the answer is not in the context, say exactly:
  "I could not find that information in the spreadsheet."
- Never invent or guess data.

Retrieved spreadsheet rows:
{context}""",
        ),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def collection_exists() -> bool:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    return COLLECTION_NAME in existing


# ─────────────────────────────────────────────────────────────
# Public API — called by graph_agent.py
# ─────────────────────────────────────────────────────────────
def get_qa_chain():
    """Return the cached LCEL chain, opening existing Chroma if possible."""
    global _chain_cache

    if _chain_cache is not None:
        return _chain_cache

    print("🔧 Loading LCEL RAG pipeline...")

    if collection_exists():
        print("✅ Existing Chroma collection found. Opening without rebuild...")
        store = get_vector_store_without_rebuild()
    else:
        print("⚠️ No Chroma collection found. Building first index from CSV...")
        docs = load_csv_as_documents()
        store = build_vector_store(docs)

    _chain_cache = build_lcel_chain(store)

    print("✅ LCEL RAG pipeline ready.")
    return _chain_cache

def rebuild_index():
    """
    Rebuild after a CSV write so the next question sees fresh data.

    Uses collection-level reset (delete + recreate) instead of
    folder deletion — fully Windows-compatible, no WinError 32.
    """
    global _chain_cache
    _chain_cache = None   # invalidate cache first

    print("🔄 Rebuilding LCEL RAG index after data change...")
    docs = load_csv_as_documents()
    store = build_vector_store(docs)

    _chain_cache = build_lcel_chain(store)
    print("✅ LCEL RAG index is fresh and ready.")

def row_to_document(row_data: dict) -> Document:
    """
    Convert one CSV row into one LangChain Document.
    This lets us embed only one row instead of the full CSV.
    """
    parts = [f"{col}: {value}" for col, value in row_data.items()]

    if "first_name" in row_data and "last_name" in row_data:
        full_name = f"{row_data.get('first_name', '')} {row_data.get('last_name', '')}".strip()
        parts.append(f"full_name: {full_name}")

    row_text = " | ".join(parts)

    row_id = row_data.get("Row", row_data.get("row_id", "unknown"))

    return Document(
        page_content=row_text,
        metadata={
            "row_id": str(row_id)
        }
    )


def get_vector_store_without_rebuild():
    """
    Open the existing Chroma collection without deleting/rebuilding it.
    """
    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def get_row_doc_id(row_id):
    """
    Stable Chroma document ID for each CSV row.
    """
    return f"row_{row_id}"

def update_row_in_index(row_id, new_row: dict):
    """
    Update only one row in Chroma:
    1. Delete the old embedding for this row
    2. Add the new embedding for this row
    """
    global _chain_cache

    vector_store = get_vector_store_without_rebuild()

    doc_id = get_row_doc_id(row_id)
    doc = row_to_document(new_row)

    # Delete old version of this row
    try:
        vector_store.delete(ids=[doc_id])
    except Exception as e:
        print(f"⚠️ Could not delete old row {row_id} from index: {e}")

    # Add updated version
    vector_store.add_documents([doc], ids=[doc_id])

    # Do not rebuild embeddings. Just let future retrieval use updated Chroma.
    print(f"✅ Updated only row {row_id} in Chroma index.")


def add_row_to_index(new_row: dict):
    """
    Add only the new row to Chroma.
    """
    vector_store = get_vector_store_without_rebuild()

    row_id = new_row.get("Row", new_row.get("row_id"))
    doc_id = get_row_doc_id(row_id)
    doc = row_to_document(new_row)

    vector_store.add_documents([doc], ids=[doc_id])

    print(f"✅ Added only row {row_id} to Chroma index.")


def delete_row_from_index(row_id):
    """
    Delete only one row from Chroma.
    """
    vector_store = get_vector_store_without_rebuild()

    doc_id = get_row_doc_id(row_id)

    try:
        vector_store.delete(ids=[doc_id])
        print(f"✅ Deleted only row {row_id} from Chroma index.")
    except Exception as e:
        print(f"⚠️ Could not delete row {row_id} from index: {e}")