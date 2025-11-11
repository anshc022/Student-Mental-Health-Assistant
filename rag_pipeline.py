from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DOC_PATH = BASE_DIR / "rag_documents.jsonl"
PERSIST_DIR = BASE_DIR / "rag_vector_store"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "models/gemini-flash-latest"

PROMPT_TEMPLATE = """You are an empathetic academic advisor. Use the provided student context to answer the question.
If the question asks for statistics, keep the answer grounded in the retrieved student cases and mention when the sample size is small.

Context:
{context}

Question: {question}
Helpful answer:"""


@dataclass
class QAResources:
    vector_store: Chroma
    retriever: any


def load_rag_documents(path: Path = DOC_PATH) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(f"RAG document file not found at {path}")

    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            docs.append(Document(page_content=item["text"], metadata={"id": item["id"]}))
    return docs


def build_vector_store(documents: Iterable[Document], persist_directory: Path = PERSIST_DIR) -> QAResources:
    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_MODEL_NAME)

    if persist_directory.exists() and any(persist_directory.iterdir()):
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )
    else:
        persist_directory.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=list(documents),
            embedding=embeddings,
            persist_directory=str(persist_directory),
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return QAResources(vector_store=vector_store, retriever=retriever)


def _format_documents(docs: List[Document]) -> str:
    if not docs:
        return "No matching student profiles were retrieved."
    lines = []
    for doc in docs:
        identifier = doc.metadata.get("id", "unknown")
        lines.append(f"Student #{identifier}: {doc.page_content}")
    return "\n\n".join(lines)


def build_qa_chain(retriever) -> Runnable | None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_GEMINI_MODEL,
        temperature=0.2,
        google_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    context_chain: Runnable = retriever | RunnableLambda(_format_documents)

    chain: Runnable = (
        {"context": context_chain, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def interactive_loop(chain: Runnable | None, retriever) -> None:
    print("Enter a question (or type 'exit'):")
    while True:
        query = input("> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        if chain:
            answer = chain.invoke(query)
            print("\nAnswer:\n" + answer)
            print("\nTop sources:")
            for doc in retriever.invoke(query):
                print(f"  - Student #{doc.metadata['id']}: {doc.page_content}")
        else:
            docs = retriever.invoke(query)
            print("\nRetrieved context (set GOOGLE_API_KEY for LLM answers):")
            for doc in docs:
                print(f"  - Student #{doc.metadata['id']}: {doc.page_content}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student mental health RAG demo")
    parser.add_argument("query", nargs="?", help="Optional one-shot query to run")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vector store")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    documents = load_rag_documents()

    if args.rebuild and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    resources = build_vector_store(documents)
    chain = build_qa_chain(resources.retriever)

    if args.query:
        if chain:
            answer = chain.invoke(args.query)
            print(answer)
            print("\nTop sources:")
            for doc in resources.retriever.invoke(args.query):
                print(f"  - Student #{doc.metadata['id']}: {doc.page_content}")
        else:
            docs = resources.retriever.invoke(args.query)
            print("No GOOGLE_API_KEY detected. Showing retrieved context instead:\n")
            for doc in docs:
                print(f"  - Student #{doc.metadata['id']}: {doc.page_content}")
    else:
        interactive_loop(chain, resources.retriever)


if __name__ == "__main__":
    main()
