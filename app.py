from __future__ import annotations
import os
import io
import re
import time
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any
from utils.embeddings import load_sentence_model, e5_embed_texts
from rag_pipeline import ingest_files, answer_question
from utils.retrieval import init_faiss_index
import numpy as np
import pandas as pd

import streamlit as st

# Document parsing
import fitz  # PyMuPDF
from docx import Document as DocxDocument

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector DB
import faiss

# LLMs
from transformers import pipeline

# PDF export
from fpdf import FPDF

# Optional OpenAI (new SDK style)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # handled later

st.set_page_config(page_title="GIKI Prospectus RAG Chatbot", page_icon="🎓", layout="wide")

if "index" not in st.session_state:
    st.session_state.index = None  # FAISS index
if "meta" not in st.session_state:
    st.session_state.meta: List[Dict[str, Any]] = []  # metadata per vector id
if "embed_model_name" not in st.session_state:
    st.session_state.embed_model_name = "intfloat/multilingual-e5-base"
if "embed_model" not in st.session_state:
    st.session_state.embed_model = None
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []  # {role: user/assistant, content}
if "dim" not in st.session_state:
    st.session_state.dim = None  # embedding dimension

def build_pdf(conversation: List[Dict[str, str]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.add_font('DejaVu', '', '', uni=True)
    try:
        pass
    except Exception:
        pass
    pdf.set_font("Arial", size=12)
    for turn in conversation:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        pdf.multi_cell(0, 6, f"{role.capitalize()}: {content}")
        pdf.ln(2)
    return pdf.output(dest='S').encode('latin-1', errors='ignore')


def download_conversation_buttons():
    if not st.session_state.history:
        return
    # Markdown export
    md_lines = [f"**{h['role'].capitalize()}:**\n\n{h['content']}" for h in st.session_state.history]
    md_text = "\n\n---\n\n".join(md_lines)
    st.download_button("⬇️ Download conversation (Markdown)", data=md_text, file_name="conversation.md", mime="text/markdown")

    # PDF export
    pdf_bytes = build_pdf(st.session_state.history)
    st.download_button("⬇️ Download conversation (PDF)", data=pdf_bytes, file_name="conversation.pdf", mime="application/pdf")

with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**Embedding model**")
    embed_choice = st.selectbox(
        "Select embedding model",
        [
            "intfloat/multilingual-e5-base",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        index=0 if st.session_state.embed_model_name == "intfloat/multilingual-e5-base" else 1,
    )

    if embed_choice != st.session_state.embed_model_name or st.session_state.embed_model is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embed_model_name = embed_choice
            st.session_state.embed_model = load_sentence_model(embed_choice)
            # set dimension
            dim = st.session_state.embed_model.get_sentence_embedding_dimension()
            st.session_state.dim = int(dim)

    st.markdown("**Generator (LLM)**")
    llm_options = ["OpenAI: gpt-4o-mini", "OpenAI: gpt-4o", "Local: flan-t5-base"]
    model_choice = st.selectbox("Answering model", llm_options, index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.0, 0.1)
    top_k = st.slider("Top-k retrieved chunks", 1, 10, 5)

    st.divider()
    st.markdown("### 📥 Upload Documents (max 5)")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload up to 5 documents",
    )
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("Only the first 5 files will be processed.")
        uploaded_files = uploaded_files[:5]

    colA, colB = st.columns(2)
    with colA:
        if st.button("📚 Build / Rebuild Index", use_container_width=True):
            with st.spinner("Parsing, chunking, embedding, and indexing..."):
                chunks, metas = ingest_files(uploaded_files or [])
                if not chunks:
                    st.error("No text found in uploaded documents.")
                else:
                    # embed passages
                    model = st.session_state.embed_model
                    vecs = e5_embed_texts(model, chunks, is_query=False)
                    index = init_faiss_index(vecs.shape[1])
                    index.add(vecs)
                    st.session_state.index = index
                    # attach text to meta and reset store
                    st.session_state.meta = []
                    for c, m in zip(chunks, metas):
                        m = {**m, "text": c}
                        st.session_state.meta.append(m)
                    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files or [])} file(s).")
    with colB:
        if st.button("🧹 Reset Index", type="secondary", use_container_width=True):
            st.session_state.index = None
            st.session_state.meta = []
            st.success("Index cleared.")

    st.divider()
    st.markdown("### 📊 Evaluation (optional)")
    st.caption("Upload CSV with columns: question, expected")
    eval_file = st.file_uploader("Upload evaluation CSV", type=["csv"], key="eval")
    if eval_file is not None and st.session_state.index is not None:
        df = pd.read_csv(eval_file)
        if not set(["question", "expected"]).issubset(set(df.columns)):
            st.error("CSV must have columns: question, expected")
        else:
            run_eval = st.button("▶️ Run evaluation", use_container_width=True)
            if run_eval:
                rows = []
                for _, row in df.iterrows():
                    q = str(row["question"])[:2000]
                    exp = str(row["expected"])[:4000]
                    ans, metas = answer_question(q, language="English", model_choice=model_choice, k=top_k, temperature=temperature)
                    # Simple semantic similarity using embedding cosine
                    vec_true = e5_embed_texts(st.session_state.embed_model, [exp], is_query=False)[0]
                    vec_pred = e5_embed_texts(st.session_state.embed_model, [ans], is_query=False)[0]
                    sim = float(np.dot(vec_true, vec_pred))
                    # Retrieval hit if any chunk contains a 6-word overlap (rough proxy)
                    def contains_overlap(a: str, b: str, n: int = 6) -> bool:
                        aw = [w for w in re.findall(r"\w+", a.lower())]
                        bw = [w for w in re.findall(r"\w+", b.lower())]
                        aset = {" ".join(aw[i:i+n]) for i in range(max(0, len(aw)-n+1))}
                        bset = {" ".join(bw[i:i+n]) for i in range(max(0, len(bw)-n+1))}
                        return len(aset & bset) > 0
                    hit = any(contains_overlap(exp, m["text"]) for m in metas)
                    rows.append({
                        "question": q,
                        "expected": exp,
                        "answer": ans,
                        "semantic_sim": round(sim, 4),
                        "retrieval_hit": bool(hit),
                        "sources": "; ".join([f"{m['source']} p.{m['page']}" if m.get('page') else m['source'] for m in metas]),
                    })
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download evaluation results (CSV)", data=csv, file_name="eval_results.csv", mime="text/csv")

st.title("🎓 GIKI Prospectus Q&A — RAG Chatbot")

left, right = st.columns([2, 1])
with right:
    st.markdown("### 🈯 Language")
    language = st.radio("Response language", ["English", "Urdu"], index=0, horizontal=True)
    st.markdown("---")
    st.markdown("### ⬇️ Downloads")
    download_conversation_buttons()

with left:
    st.markdown("#### Chat with your uploaded documents")
    if st.session_state.index is None or not st.session_state.meta:
        st.info("Upload up to 5 GIKI-related documents in the sidebar and click **Build / Rebuild Index**.")

    # Display chat history
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Chat input
    q = st.chat_input("Ask a question about the uploaded documents…")
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ans, metas = answer_question(q, language=language, model_choice=model_choice, k=top_k, temperature=temperature)
            st.markdown(ans)
            if metas:
                with st.expander("Sources"):
                    for m in metas:
                        src = m.get("source", "?")
                        page = m.get("page")
                        tag = f"{src} (p.{page})" if page else src
                        st.markdown(f"- **{tag}**\n\n{m['text'][:500]}…")
        st.session_state.history.append({"role": "assistant", "content": ans})