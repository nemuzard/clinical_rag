python -m pip install -r requirements.txt



# Clinical Evidence Navigator

A local RAG system built on top of three 2024 clinical practice guidelines:

- **ADA 2024** – Standards of Care in Diabetes  
- **KDIGO 2024** – CKD Evaluation and Management Guideline  
- **GOLD 2024** – COPD Strategy Report  

The system ingests guideline PDFs, converts them into semantically indexed chunks using local embeddings and a Chroma vector store, and exposes a simple interface to:

- Retrieve guideline excerpts relevant to a natural-language question  
- Summarize those excerpts using a **local LLM via Ollama**  


---

## 1. Features

- **Local document ingestion**  
  - PDF → text extraction via **PyMuPDF**  
  - Simple cleaning and chunking strategy (~1200 characters with overlap)

- **Semantic search over guidelines**  
  - Local embeddings via **sentence-transformers** (`all-MiniLM-L6-v2` by default)  
  - Persistent vector store backed by **Chroma**

- **Structured metadata for each guideline**  
  - `study_id`, `title`, `year`, `condition`, `source_type`, `source_file`  
  - Enables source tracking and future filtering (e.g., by condition or source type)

- **Pluggable LLM backend**  
  - **Retrieval-only mode** (default): returns relevant passages and metadata  
  - **Ollama mode**: uses a local LLM (e.g., `llama3`) to summarize retrieved excerpts  

---

## 2. Project Structure

```bash
clinical-evidence-navigator/
├─ config.py              # Global configuration (paths, models, LLM backend)
├─ ingest.py              # PDF ingestion → chunks → embeddings → Chroma
├─ rag_core.py            # RAG core: retrieval and optional LLM summarization)
├─ data/
│  ├─ raw/                # Raw guideline PDFs 
│  │  └─ some .pdf
│  └─ studies_meta.json   # Guideline metadata.json 
└─ chroma_db/             # Chroma 
