import os
from dotenv import load_dotenv

load_dotenv()

CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "data.csv")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

GOOGLE_SHEET_ID = os.getenv(
    "GOOGLE_SHEET_ID",
    "1vMQ90YV9JZ_HImf0r1yIOZ0f_3xZk5nsfO68cLDq_HY"
)

GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    "./service_account.json"
)


LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "slack-csv-agent")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")