from __future__ import annotations

import os
from typing import List

import requests
import streamlit as st

API_URL = os.getenv("MENTAL_HEALTH_API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="Student Mental Health Assistant", page_icon="🧠", layout="wide")
st.title("Student Mental Health Assistant")
st.write("Ask data-backed questions about student mental health indicators using the RAG assistant.")

question = st.text_area("Question", value="Which students reported both depression and anxiety?", height=120)
retrieval_k = st.slider("Top K student profiles", min_value=1, max_value=10, value=4)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying assistant..."):
            try:
                response = requests.post(API_URL, json={"question": question.strip(), "k": retrieval_k}, timeout=60)
            except requests.RequestException as exc:
                st.error(f"Failed to reach API: {exc}")
            else:
                if response.status_code != 200:
                    st.error(f"API returned {response.status_code}: {response.text}")
                else:
                    payload = response.json()
                    label = "Gemini answer" if payload.get("from_llm") else "Retrieved context"
                    st.subheader(label)
                    st.write(payload.get("answer", "No answer returned."))

                    sources: List[dict] = payload.get("sources", [])
                    if sources:
                        st.subheader("Source student profiles")
                        for source in sources:
                            st.markdown(f"**Student #{source.get('id', 'unknown')}**")
                            st.write(source.get("text", ""))
                    else:
                        st.info("No source documents returned.")

st.caption(
    "Set `MENTAL_HEALTH_API_URL` to point at a deployed FastAPI instance. Default assumes local `uvicorn app.main:app`."
)
