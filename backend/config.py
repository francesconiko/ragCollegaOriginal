from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

import os  # Add os import if not already present

# --- CONFIG ROOT DEFINITION (Solves portability) ---
# Path of config.py file (e.g.: .../TEXTMINING/backend/config.py)
CONFIG_DIR = Path(__file__).parent 

# Project root (e.g.: .../TEXTMINING)
# Goes up one level from 'backend' folder
PROJECT_ROOT = CONFIG_DIR.parent 
# -----------------------------------------------------------------

def _find_all_vector_stores(base_dir: str) -> List[str]:
    """
    Scan the base directory to find all existing subdirectories
    that represent FAISS Vector Stores.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    # Returns a list of relative or absolute paths of subfolders
    # that are your DBs (e.g. 'vector_store/divorce_codes')
    # We use str(p) to maintain compatibility with List[str] type
    return [str(p) for p in base_path.iterdir() if p.is_dir()]


@dataclass
class RAGConfig:
    """
    Global configuration object for the Legal RAG system.
    """
    
    # ---------------- LLM & Embeddings (omessi per brevità) ----------------
    # Provider: "openrouter" or "huggingface"
    llm_provider: str = "openrouter"
    # OpenRouter model format: provider/model (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-haiku")
    # See https://openrouter.ai/models for available models
    llm_model_name: str = "openai/gpt-oss-120b"

    embedding_provider: str = "huggingface"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ---------------- Data (JSON corpus) (omessi per brevità) ----------------
    #data_base_dir: str = r"C:\Users\vikya\Desktop\TM\Progetto_claude\Contest_Data"
    data_base_dir: str = str(PROJECT_ROOT / "Contest_Data")
    countries: List[str] = field(default_factory=lambda: ["Italy", "Estonia", "slovenia"])
    
    document_types: dict = field(default_factory=lambda: {
        "Italy": {
            "divorce": "Divorce_italy",
            "inheritance": "Inheritance_italy", 
            "cases": "italian_cases_json_processed"
        },
        "Estonia": {
            "divorce": "Divorce_estonia",
            "inheritance": "Inheritance_estonia",
            "cases": "estonian_cases_json_processed"
        },
        "slovenia": {
            "divorce": "Divorce_slovenia",
            "inheritance": "Inheritance_slovenia",
            "cases": "slovenian_cases_json_processed"
        }
    })
    
    # ---------------- Vector stores (paths) ----------------
    # Root folder for all FAISS vector DBs
    vector_store_base_dir: str = "vector_store"
    
    # Default single vector store (per retrocompatibilità, se non usato viene ignorato)
    vector_store_dir: str = "vector_store/divorce"
    
    # Multi-DB: ORA CARICATA DINAMICAMENTE
    vector_store_dirs: List[str] = field(
        default_factory=lambda: _find_all_vector_stores("vector_store")
    )
    
    # ---------------- Retrieval & Agentic behavior (omessi per brevità) ----------------
    top_k: int = 30
    top_k_final: int = 20
    similarity_threshold: float = 0.3
    use_rerank: bool = False
    
    agentic_mode: str = "standard_rag"
    use_multiagent: bool = False
    
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512
    
    logs_dir: str = "logs"
    enable_logging: bool = True
    
    def __post_init__(self):
        """Ensure paths are correct and update the DB list"""
        # Recalculate the DB list at instance creation time
        # This is redundant but ensures that if base_dir is modified at runtime, it works
        self.vector_store_dirs = _find_all_vector_stores(self.vector_store_base_dir)


    def get_data_path(self, country: str, doc_type: str) -> Path:
        """Get the full path for a specific country and document type"""
        folder_name = self.document_types[country][doc_type]
        return Path(self.data_base_dir) / country / folder_name
    
    def get_all_data_paths(self) -> List[Path]:
        """Get all data paths for all countries and document types"""
        paths = []
        for country in self.countries:
            for doc_type in ["divorce", "inheritance", "cases"]:
                path = self.get_data_path(country, doc_type)
                if path.exists():
                    paths.append(path)
        return paths