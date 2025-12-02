from typing import List,Dict,Any
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from config import(

	CHROMA_DIR,
	CHROMA_COLLECTION_NAME,
	EMBEDDING_MODEL,
	LLM_BACKEND,
	OLLAMA_MODEL,
	OLLAMA_BASE_URL,
)

try:
	from langchain_community.chat_models import ChatOllama
except ImportError:
	ChatOllama = None

SYSTEM_PROMPT = """
You are an assistant that summarizes and explains clinical practice guidelines. 
You are not allowed to give specific medical advice, dosing instructions, or treatment decision.
You are not a doctor. 
Base your answers strictly on the provided guideline experts and mention which guideline and year a conclusion comes from.
"""

class RAG:
	"""
	retrieve relevant chunks from local chroma vector lib
	generate a summary if llm is configured
	else, return only the retrieved evidence paragraph

	"""

	def __init__(

		self,
		k: int = 6,
	)->None:
		load_dotenv()
		# 1. embeddings -> same as ingest 
		self.embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)