from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional 
from rag_core import RAG

app = FastAPI(title="Clinical RAG project", version="1.0.0")
rag: Optional[RAG] = None

class QueryRequest(BaseModel):
	question : str = Field(...,min_length=3,max_length=2000)
	k:int = Field(6,ge=1,le=20)
	raw_context:bool = False

class Source(BaseModel):
	study_id:Optional[str]=None
	title:Optional[str]=None
	year:Optional[int]=None
	condition:Optional[str]=None
	source_type:Optional[str]=None
	source_file:Optional[str]=None
	page:Optional[int]=None

class QueryResponse(BaseModel):
	answer:Optional[str]=None
	sources:List[Source]
	raw_context:Optional[str]=None

@app.on_event("startup")
def startup():
	global rag
	rag = RAG(k=6)

@app.get("/healthz")
def healthz():
	return {"status":"ok"}

@app.get("/readyz")
def readyz():
	if rag is None:
		raise HTTPException(status_code=503,detail="RAG is none")
	try:
		collection = rag.vectordb._collection # chroma collection
		collection_count=collection.count()
		if collection_count==0:
			raise HTTPException(status_code=503,detail="Vectore store is empty.")
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=503,detail=f"Vectore store not ready: {type(e).__name__}")

	try:
		docs = rag.retriever.invoke("readiness-probe")
		return{
			"status":"ready",
			"vector_store":"ok",
			"collection_count":collection_count,
			"num_results":len(docs),
			"k":rag.retriever.search_kwargs.get("k")

		}
	except Exception as e:
		raise HTTPException(status_code=503, detail=f"Retriever not ready: {type(e).__name__}")

@app.post("/query",response_model=QueryResponse)
def query(req:QueryRequest):
	rag.retriever.search_kwargs={"k":req.k}
	result:Dict[str,Any] = rag.ask(req.question)
	if result.get("sources") is None:
		raise HTTPException(status_code=500, detail="Invalid, source is none")
	resp={
		"answer":result.get("answer"),
		"sources":result.get("sources",[])
	}
	if req.raw_context and "raw_context" in result:
		resp["raw_context"] = result["raw_context"]
	return resp