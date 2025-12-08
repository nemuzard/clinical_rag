from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR/"data"
RAW_DIR = DATA_DIR/"raw"

STUDIES_META_PATH = DATA_DIR/"studies_meta.json"

CHROMA_DIR = BASE_DIR/"chrome_db"

CHROMA_COLLECTION_NAME = "clinical_evidence"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

LLM_BACKEND = os.getenv("LLM_BACKEND","none").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")