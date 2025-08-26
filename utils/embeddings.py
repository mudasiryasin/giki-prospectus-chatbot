import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import streamlit as st

def load_sentence_model(name: str) -> SentenceTransformer:
    model = SentenceTransformer(name)
    return model


def e5_embed_texts(model: SentenceTransformer, texts: List[str], is_query: bool) -> np.ndarray:
    """Encode texts with E5-style prefixes if model is E5; otherwise plain encoding.
    Returns L2-normalized embeddings for cosine similarity with FAISS IP.
    """
    name = st.session_state.embed_model_name.lower()
    if "e5" in name:
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")