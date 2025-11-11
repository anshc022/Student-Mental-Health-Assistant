# Student Mental Health Assistant

End-to-end data science project that analyzes student mental-health survey responses, trains a risk model, and serves retrieval-augmented responses through a Gemini-backed assistant.

## Project Highlights

- **Data pipeline** – `mental_health_pipeline.py` cleans the raw CSV, trains a balanced logistic regression model (≈75% hold-out accuracy), reports cross-validation metrics, and exports RAG-friendly documents.
- **RAG stack** – `rag_pipeline.py` generates or loads a Chroma vector store and answers questions using Gemini (`models/gemini-flash-latest`) with contextual grounding in the dataset.
- **API + UI** – `app/main.py` exposes a FastAPI service for programmatic access, while `streamlit_app.py` delivers an interactive dashboard that consumes the API.
- **Secrets management** – `.env` holds the `GOOGLE_API_KEY`; the file is ignored by git to prevent accidental leaks.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

1. Place the survey CSV at `Student Mental health.csv` (already provided).
2. Store your Gemini key in `.env`:
   ```env
   GOOGLE_API_KEY=your-gemini-key
   ```
3. Build the analytics outputs (optional but recommended):
   ```powershell
   python mental_health_pipeline.py
   ```
4. Rebuild the vector store (first-time setup):
   ```powershell
   python rag_pipeline.py --rebuild
   ```

## Running the Services

### FastAPI (Recommended for deployment)
```powershell
uvicorn app.main:app --reload
```
- `POST /query` – Ask a question with `{ "question": "...", "k": 4 }`.
- `GET /health` – Simple readiness probe.

### Streamlit Frontend
```powershell
streamlit run streamlit_app.py
```
Set `MENTAL_HEALTH_API_URL` if the API runs on a non-default host.

## Deployment Notes

- **Containerization** – Add a Dockerfile that installs `requirements.txt`, runs `rag_pipeline.py --rebuild` on build (or entrypoint), and starts `uvicorn`.
- **Hosting** – FastAPI fits neatly on Render, Fly.io, or Azure App Service. Streamlit works well on Hugging Face Spaces or Streamlit Community Cloud.
- **CI/CD** – Configure GitHub Actions to lint (`ruff`, `black --check`) and run tests (pending) before deploying.

## Resume Bullet Inspiration

- Developed a student mental-health analytics assistant combining classical ML (logistic regression, 75% accuracy) with Gemini-powered RAG to surface personalized insights from survey data.
- Shipped an API + Streamlit interface backed by LangChain, Chroma, and sentence-transformer embeddings, secured via `.env` secrets and ready for containerized deployment.

## Next Steps

- Add unit tests around the data cleaning and inference layers.
- Capture API usage metrics and feedback for continuous improvement.
- Incorporate the logistic regression risk scores into RAG responses for proactive alerts.
