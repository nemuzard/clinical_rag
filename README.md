# Clinical Evidence Navigator

A local RAG system built on top of three 2024 clinical practice guidelines:

- **ADA 2024** – Standards of Care in Diabetes  
- **KDIGO 2024** – CKD Evaluation and Management Guideline  
- **GOLD 2024** – COPD Strategy Report  

The system ingests guideline PDFs, converts them into semantically indexed chunks using local embeddings and a Chroma vector store, and exposes a simple interface to:

- Retrieve guideline excerpts relevant to a natural-language question  
- Summarize those excerpts using a local LLM

---
## 0.Quick Start
### Install
`python -m pip install -r requirements.txt`

### Ingest pdfs
```python ingest.py```

### run api(fastAPI)
`unicorn app:aoo --reload --port 8000 `

- **Health Endpoints**
  - `/healthz` : liveness
  - `/readyz ` : check vector store and retriever

- **Query Endpoint**
  - /query
    - body: `{"question":"a question","k":6,"raw_context": false}`

---

## 1. Features

- **Page-level ingestion with traceable citations**
  - page text extraction via **PyMuPDF**
  - Each chunk preserves citation metadata (`study_id`, `page`, `chunk_index`) 

- **Semantic retrieval over clinical guidelines**
  - Local embeddings via **sentence-transformers** 
  - Persistent vector store backed by **Chroma** 
  - Top-k retrieval configurable via environment variables

- **Reliability guardrails**
  - Evidence thresholding (abstain) based on retrieval distances to avoid “confident wrong” answers when evidence is weak
  - Out-of-domain negative controls supported in evaluation to validate abstention behavior
<!--
- **Offline evaluation harness**
  - `data/eval_set.json` + `eval.py` computes **Hit@k**, **MRR**, and abstain metrics for regression tracking
  - Produces `eval_report.json` for repeatable comparisons across chunking/embedding/threshold changes
-->
- **Pluggable LLM backend**
  - **Retrieval-only mode**: returns evidence passages + citations
  - **Ollama mode**: summarizes retrieved excerpts using a local model (`llama3`), constrained to provided evidence

---

## 2. Project Structure

```bash
clinical-evidence-navigator/
├─ config.py              # Global configuration (paths, models, LLM )
├─ ingest.py              # PDF ingestion → chunks → embeddings → Chroma
├─ rag_core.py            # RAG core: retrieval and optional LLM summarization)
├─ app.py
├─ data/
│  ├─ raw/                # Raw guideline PDFs 
│  │  └─ some .pdf
│  ├─ studies_meta.json   # Guideline metadata.json 
|  └─ eval_set.json
|
└─ chroma_db/             # Chroma 
