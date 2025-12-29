from typing import List,Dict,Any
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

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

		#2. chroma
		self.vectordb = Chroma(

			persist_directory=str(CHROMA_DIR),
			collection_name = CHROMA_COLLECTION_NAME,
			embedding_function=self.embeddings,
		)

		# 3. retriever
		self.retriever = self.vectordb.as_retriever(
			search_kwargs={"k":k}
		)

		# 4. LLM backend 
		self.llm_backend = LLM_BACKEND # a
		self.llm = None # b
		self.prompt = None # c
		# a
		if self.llm_backend == "ollama":
			
			if ChatOllama is None:
				raise ImportError("Chatollama is none")
			# b 
			self.llm=ChatOllama(
				model=OLLAMA_MODEL,
				base_url = OLLAMA_BASE_URL,
				temperature=0.1,
			)
			# c
			self.prompt = ChatPromptTemplate.from_messages(

				[
					("system",SYSTEM_PROMPT),
					(
						"human",
                        "Question:\n{question}\n\n"
                        "Here are relevant guideline:\n\n{context}\n\n",
					),
				]

			)
	@staticmethod
	def _format_docs(docs:List[Document]) -> str:

		chunks = []
		for i,d in enumerate(docs, start = 1):
			meta = d.metadata or {}
			title = meta.get("title","Unknown title")
			year = meta.get("year","Unknown year")
			study_id = meta.get("study_id","N/A")
			condition = meta.get("condition","")
			page = meta.get("page","N/A")
			header =  f"[{i}] {study_id} | {title} ({year}) | p.{page}|{condition}"
			content = d.page_content
			chunks.append(f"{header}\n{content}")
		return "\n\n---\n\n".join(chunks)

	def ask(self,question):
		"""
		input - question
		output - 
				1. if llm : {answer:xxx, source:...}
				2. No llm:  {answer:none, source:..., context:xxx}
		"""
		docs: List[Document] = self.retriever.invoke(question)
		if not docs:

			return {
				"answer":" Cannot find relevant information",
				"sources" : [],
			}
		sources = []
		for d in docs:
			meta = d.metadata or {}
			sources.append(
				{
					"study_id":meta.get("study_id"),
					"title":meta.get("title"),
					"year":meta.get("year"),
        			"condition":meta.get("condition"),
        			"source_type":meta.get("source_type"),
        			"source_file":meta.get("source_file"),
        			"page":meta.get("page"),
        		}
        	)
		if self.llm_backend!="ollama":
			context = self._format_docs(docs)
			return {
        		"answer":None,
        		"sources":sources,
        		"raw_context":context,
        	}
		context = self._format_docs(docs)
		messages = self.prompt.format_messages(
        	question=question,
        	context = context,
        )
		resp = self.llm.invoke(messages)
		return {
        	"answer":resp.content,
        	"sources":sources,
        }



if __name__ == "__main__":
    rag = RAG(k=6)
    test_question = "According to KDIGO 2024, how is chronic kidney disease staged?"
    result = rag.ask(test_question)

    print("=== Answer ===")
    print(result.get("answer"))
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(f"- {s['study_id']}: {s['title']} ({s['year']}) p.{s['page']}| {s['condition']}")

    if "raw_context" in result:
        print("\n=== Raw Context (truncated) ===")
  
        print(result["raw_context"][:1000])



