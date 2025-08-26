import numpy as np
import faiss
from typing import List, Dict, Tuple
from utils.embeddings import e5_embed_texts
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Tuple, Dict, Any

def init_faiss_index(dim: int) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(dim)
    return index


def add_to_index(index: faiss.IndexFlatIP, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
    assert embeddings.shape[0] == len(metadatas)
    index.add(embeddings)
    st.session_state.meta.extend(metadatas)


def mmr_rerank(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 5, lambda_mult: float = 0.7) -> List[int]:
    """A tiny MMR implementation to diversify results."""
    # Similarities with query (cosine, as vectors are normalized)
    sim = (doc_vecs @ query_vec.reshape(-1, 1)).flatten()
    selected: List[int] = []
    candidates = list(range(len(doc_vecs)))

    if not candidates:
        return []

    # pick best first
    best = int(sim.argmax())
    selected.append(best)
    candidates.remove(best)

    while len(selected) < min(k, len(doc_vecs)) and candidates:
        max_score = -1e9
        max_idx = candidates[0]
        for c in candidates:
            diversity = max((doc_vecs[c] @ doc_vecs[s] for s in selected), default=0.0)
            score = lambda_mult * sim[c] - (1 - lambda_mult) * diversity
            if score > max_score:
                max_score = score
                max_idx = c
        selected.append(max_idx)
        candidates.remove(max_idx)
    return selected


def search_top_k(query: str, k: int = 5, use_mmr: bool = True) -> Tuple[List[int], List[Dict[str, Any]]]:
    if st.session_state.index is None:
        return [], []
    model: SentenceTransformer = st.session_state.embed_model
    qvec = e5_embed_texts(model, [query], is_query=True)[0]
    D, I = st.session_state.index.search(qvec.reshape(1, -1), min(50, max(k * 5, k)))
    I = I[0]
    if use_mmr:
        # fetch candidate vectors
        vecs = []
        for idx in I:
            text = st.session_state.meta[idx]["text"]
            vec = e5_embed_texts(model, [text], is_query=False)[0]
            vecs.append(vec)
        vecs = np.stack(vecs, axis=0)
        sel_local = mmr_rerank(qvec, vecs, k=k)
        selected_global = [int(I[i]) for i in sel_local]
    else:
        selected_global = [int(x) for x in I[:k]]
    metas = [st.session_state.meta[i] for i in selected_global]
    return selected_global, metas

