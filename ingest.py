import json
from pathlib import Path
from typing import List
import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

from config import(
    DATA_DIR,
    RAW_DIR,
    STUDIES_META_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
def clean(text: str) -> str:
   
    text = text.replace("-\n", "")

    text = text.replace("\n", " ")

    import re
    text = re.sub(r"\s+", " ", text)

    return text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []

    for page in doc:
        page_text = page.get_text("text")
        texts.append(page_text)

    doc.close()
    full_text="\n\n".join(texts)
    return clean(full_text)

def load_studies_meta():
    if not STUDIES_META_PATH.exists():
        raise FileNotFoundError(f"File path does not exists: {STUDIES_META_PATH}")

    with STUDIES_META_PATH.open("r",encoding = "utf-8") as f:
        meta_list = json.load(f)
    if not isinstance(meta_list,list):
        raise ValueError("Not Json array")
    return meta_list

def build_docs(meta_list):
    documents = []
    for meta in meta_list:
        filename = meta.get("filename")
        if not filename:
            raise ValueError(f"no filename:{meta}")

        filepath = RAW_DIR/filename
        if not filepath.exists():
            raise ValueError(f"cannot find {filepath}")

        if filepath.suffix.lower()!=".pdf":
            raise ValueError("PDF required")
        text = extract_text_from_pdf(filepath)

        doc = Document(
                page_content = text,
                metadata = {
                "study_id":meta.get("study_id"),
                "title":meta.get("title"),
                "year":meta.get("year"),
                "condition":meta.get("condition"),
                "intervention":meta.get("intervention"),
                "comparator":meta.get("comparator"),
                "primary_outcome":meta.get("primary_outcome"),
                "sample_size":meta.get("sample_size"),
                "source_file":str(filepath),
                "source_type":meta.get("source_type","guideline")
            },
        )
        documents.append(doc)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n","\n"," ",""]

    )
    return splitter.split_documents(documents)


def ingest():
    load_dotenv()
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} not exist")
    meta_list  =  load_studies_meta()
    print(f"loaded {len(meta_list)} records.")

    documents = build_docs(meta_list)
    print(f"built {len(documents)} raw documents")

    splitted_docs = split_documents(documents)
    print(f"splitted into {len(splitted_docs)} chunks")

    CHROMA_DIR.mkdir(parents=True,exist_ok=True)

    print("saving to chroma vector to store ")
    vectorstore = Chroma.from_documents(
        documents=splitted_docs,
        embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION_NAME,

    )
    vectorstore.persist()
    print(f"Vector store at {CHROMA_DIR}")

if __name__=="__main__":
    ingest()