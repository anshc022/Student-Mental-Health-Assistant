from __future__ import annotations

import os
import streamlit as st

st.set_page_config(page_title="Student Mental Health Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Student Mental Health Assistant")

st.markdown("""
### About This Project

This is a demonstration of an end-to-end RAG (Retrieval Augmented Generation) assistant for analyzing student mental health data.

**🔧 Technology Stack:**
- **Machine Learning**: Scikit-learn with 75% accuracy logistic regression
- **RAG Pipeline**: LangChain + Chroma vector store + Gemini LLM
- **Backend**: FastAPI with `/query` endpoint
- **Frontend**: Streamlit (this interface)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

**📊 Dataset Analysis:**
- 101 student mental health survey responses
- Features: Demographics, academic performance, mental health indicators
- Target: Depression risk prediction and contextual insights

**🚀 Local Deployment:**
To run the full system locally with all features:

```bash
# Clone the repository
git clone https://github.com/anshc022/Student-Mental-Health-Assistant.git
cd Student-Mental-Health-Assistant

# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "GOOGLE_API_KEY=your-gemini-key" > .env

# Build the analytics pipeline
python mental_health_pipeline.py

# Build vector store
python rag_pipeline.py --rebuild

# Start FastAPI backend
uvicorn app.main:app --reload

# In another terminal, start Streamlit frontend
streamlit run streamlit_app.py
```

**🔗 GitHub Repository:**
[https://github.com/anshc022/Student-Mental-Health-Assistant](https://github.com/anshc022/Student-Mental-Health-Assistant)
""")

st.markdown("---")

st.subheader("📝 Demo Query Interface")
st.info("⚠️ **Note**: The full RAG system requires local deployment. This demo shows the UI only.")

# Mock interface to show functionality
question = st.text_area(
    "Ask a question about student mental health:", 
    value="Which students reported both depression and anxiety?",
    height=120,
    help="Try questions like: 'What percentage of students seek treatment?' or 'Show me year 1 students with high CGPA and anxiety'"
)

retrieval_k = st.slider("Number of student profiles to retrieve", min_value=1, max_value=10, value=4)

if st.button("🔍 Ask Assistant (Demo)", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.subheader("💡 Sample Response")
        st.success("""
        **Gemini-powered Answer:**
        
        Based on the student mental health dataset, I found **1 student** who reported experiencing both depression and anxiety:
        
        • **Student #57**: 20-year-old female in year 3 studying BENL course
        • CGPA: 3.00-3.49 range (3.25 average)
        • Mental health profile: Depression ✓, Anxiety ✓, No panic attacks
        • Treatment status: Has not sought specialist treatment
        
        This represents approximately 1% of the total sample (1 out of 101 students), highlighting the importance of co-occurring mental health conditions in academic settings.
        """)
        
        st.subheader("📄 Source Documents")
        with st.expander("View retrieved student profiles", expanded=True):
            st.markdown("""
            **Student #57**: Female student aged 20 in year 3 of the benl course has a CGPA band 3.00 - 3.49 (3.25) and is with depression, with anxiety, without panic attacks, and who has not sought specialist treatment.
            
            **Student #42**: Female student aged 20 in year 2 of the usuluddin course has a CGPA band 3.00 - 3.49 (3.25) and is with depression, without anxiety, without panic attacks, and who has not sought specialist treatment.
            
            **Student #6**: Female student aged 23 in year 2 of the pendidikan islam course has a CGPA band 3.50 - 4.00 (3.75) and is with depression, without anxiety, with panic attacks, and who has not sought specialist treatment.
            
            **Student #93**: Female student aged 18 in year 1 of the benl course has a CGPA band 3.00 - 3.49 (3.25) and is with depression, without anxiety, without panic attacks, and who has not sought specialist treatment.
            """)

st.markdown("---")
st.subheader("🎯 Key Project Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🔬 Data Science Pipeline:**
    - Comprehensive data cleaning & preprocessing
    - Feature engineering from survey responses  
    - Balanced logistic regression (75% accuracy)
    - Cross-validation & permutation importance analysis
    - Automated model evaluation & reporting
    """)

with col2:
    st.markdown("""
    **🤖 RAG System:**
    - Semantic search with vector embeddings
    - Contextual response generation via Gemini
    - Student profile retrieval & ranking
    - Real-time query processing
    - Source document transparency
    """)

st.markdown("---")
st.caption("💻 **Resume Highlight**: End-to-end ML + RAG system combining classical ML (scikit-learn) with modern LLM technology (Gemini, LangChain) for production-ready mental health insights.")