from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path so we can import rag_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline import (
    BASE_DIR,
    QAResources,
    build_qa_chain,
    build_vector_store,
    load_rag_documents,
)

# FastAPI instance
app = FastAPI(
    title="Student Mental Health Assistant",
    description="RAG-backed advisory service powered by Gemini and Chroma",
    version="0.1.0",
)

# Ensure .env secrets are loaded for Gemini access when the module initializes
load_dotenv(dotenv_path=BASE_DIR / ".env")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the dataset")
    k: int = Field(4, ge=1, le=10, description="Number of similar profiles to retrieve")


class SourceDocument(BaseModel):
    id: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    from_llm: bool = Field(
        False,
        description="True when Gemini generated the response. False falls back to retrieved context only.",
    )


@lru_cache()
def _load_resources(k: int = 4) -> tuple[Optional[object], QAResources]:
    try:
        documents = load_rag_documents()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    resources = build_vector_store(documents)
    retriever = resources.vector_store.as_retriever(search_kwargs={"k": k})
    chain = build_qa_chain(retriever)
    return chain, resources


@app.get("/health", summary="Service health probe")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, summary="Ask a student mental health question")
def query_service(payload: QueryRequest) -> QueryResponse:
    chain, resources = _load_resources(payload.k)
    retriever = resources.vector_store.as_retriever(search_kwargs={"k": payload.k})

    if chain:
        answer = chain.invoke(payload.question)
        docs = retriever.invoke(payload.question)
        sources = [SourceDocument(id=int(doc.metadata.get("id", -1)), text=doc.page_content) for doc in docs]
        return QueryResponse(answer=answer, sources=sources, from_llm=True)

    # No Gemini key present: fall back to retrieval context only
    logging.warning("GOOGLE_API_KEY not detected. Returning retrieved context only.")
    docs = retriever.invoke(payload.question)
    if not docs:
        raise HTTPException(status_code=404, detail="No matching profiles found")

    answer_lines = ["Unable to contact Gemini. Returning closest student profiles:\n"]
    for doc in docs:
        answer_lines.append(f"- Student #{doc.metadata.get('id', 'unknown')}: {doc.page_content}")
    answer = "\n".join(answer_lines)
    sources = [SourceDocument(id=int(doc.metadata.get("id", -1)), text=doc.page_content) for doc in docs]
    return QueryResponse(answer=answer, sources=sources, from_llm=False)


def create_app() -> FastAPI:
    return app
