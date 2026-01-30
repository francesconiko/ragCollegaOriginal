"""
Legal RAG System - Main Entry Point

Landing page that introduces the application and provides navigation
to different sections via Streamlit multipage.
"""

import streamlit as st
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.config import RAGConfig

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# INITIALIZE CONFIG (CHECK VECTOR STORES)
# =====================================================================

@st.cache_resource
def load_config():
    """Load configuration once and cache it"""
    return RAGConfig()

config = load_config()

# =====================================================================
# MAIN PAGE CONTENT
# =====================================================================

st.title("⚖️ Legal RAG System")
st.markdown("### Multi-Agent Retrieval-Augmented Generation for Cross-Border Legal Knowledge")

st.divider()

# Introduction
st.markdown("""
## 👋 Welcome

This system uses **Retrieval-Augmented Generation (RAG)** and **agentic architectures** 
to answer complex legal questions about:

- 🇮🇹 **Italy** - Divorce and Inheritance
- 🇪🇪 **Estonia** - Divorce and Inheritance  
- 🇸🇮 **Slovenia** - Divorce and Inheritance

The system combines advanced information retrieval techniques with language models 
to provide accurate and well-documented answers on regulations and case law 
regarding family and inheritance law.
""")

st.divider()

# Navigation Guide
st.markdown("## 🧭 Navigation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 💬 Chatbot Interface
    
    Interact with the system to ask legal questions.
    
    **Features:**
    - Retrieval parameters configuration
    - Cited sources visualization
    - Reasoning trace (optional)
    - Export conversation to JSON
    
    👉 **Go to the Chatbot page in the sidebar** ⬅️
    """)

with col2:
    st.markdown("""
    ### 📊 Evaluation Dashboard
    
    Evaluate the quality of system responses.
    
    **Supported Metrics:**
    - **Faithfulness**: Coherence and adherence of the response to the provided documents only.
    - **Context Precision**: Ratio between relevant retrieved documents and total retrieved documents.
    - **Context Recall**: Presence of all necessary information within the retrieved context.
    - **Answer Relevancy**: Direct relevance of the response to the user's question.
    - **Answer Correctness**: Accuracy of the response compared to the reference solution (ground truth).
    
    👉 **Go to the Evaluation page in the sidebar** ⬅️
    """)

st.divider()

# Quick Start
st.markdown("## 🚀 Quick Start")

st.markdown("""
1. **Navigate** to the **Chatbot** page from the sidebar
2. **Configure** retrieval parameters (optional)
3. **Ask** your legal question
4. **Analyze** the provided sources and reasoning
5. **Export** the conversation for future analysis
6. **Evaluate** quality on the **Evaluation** page
""")

# Footer
st.divider()
st.caption("🔬 Legal RAG System | Developed for cross-border analysis of divorce and inheritance regulations")