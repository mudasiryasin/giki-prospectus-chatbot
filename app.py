import os
import fitz  # PyMuPDF
import docx
import re
import streamlit as st
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from googletrans import Translator   # pip install googletrans==4.0.0-rc1
from transformers import pipeline     # pip install transformers accelerate bitsandbytes


# ---------------- Document Processing ---------------- #
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


def process_file(uploaded_file) -> List[Dict]:
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    temp_path = os.path.join("temp_" + uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if ext == ".pdf":
        text = extract_text_from_pdf(temp_path)
    elif ext == ".docx":
        text = extract_text_from_docx(temp_path)
    elif ext == ".txt":
        text = extract_text_from_txt(temp_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    os.remove(temp_path)
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    chunk_data = []
    for idx, chunk in enumerate(chunks):
        chunk_data.append({
            "text": chunk,
            "metadata": {
                "file": uploaded_file.name,
                "chunk_number": idx + 1
            }
        })
    return chunk_data


# ---------------- FAISS Vector Store ---------------- #
class VectorStoreFAISS:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.texts = []

    def build_index(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata = [c["metadata"] for c in chunks]
        self.texts = texts

    def search(self, query: str, top_k: int = 3):
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb.astype("float32"), top_k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx != -1:
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(dist)
                })
        return results


# ---------------- RAG Pipeline ---------------- #
def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n".join([f"From {c['metadata']['file']} (chunk {c['metadata']['chunk_number']}): {c['text']}" 
                           for c in retrieved_chunks])
    prompt = f"""
You are a helpful assistant for answering questions from GIKI documents.

Context:
{context}

Question: {query}
Answer:
"""
    return prompt


# --- OpenAI GPT (if credits) --- #
def generate_answer_openai(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a knowledgeable assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# --- HuggingFace Local Model --- #
# (small model for demo, replace with larger like mistralai/Mistral-7B if GPU available)
hf_pipeline = pipeline("text-generation", model="distilgpt2")

def generate_answer_hf(prompt: str) -> str:
    response = hf_pipeline(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]


# ---------------- Helper: Export Chat as PDF ---------------- #
def export_chat_pdf(history):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []
    for role, msg in history:
        story.append(Paragraph(f"<b>{role}:</b> {msg}", styles["Normal"]))
        story.append(Spacer(1, 12))
    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="GIKI Prospectus Q&A Chatbot", layout="wide")
st.title("🤖 GIKI Prospectus Q&A Chatbot using Retrieval-Augmented Generation (RAG)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "Upload up to 5 documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

col1, col2 = st.columns(2)
with col1:
    language = st.radio("🌐 Response Language:", ["English", "Urdu"])
with col2:
    model_choice = st.radio("🧠 Choose LLM:", ["OpenAI GPT-3.5", "HuggingFace Local"])

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        try:
            chunks = process_file(uploaded_file)
            all_chunks.extend(chunks)
            st.success(f"Processed {uploaded_file.name} with {len(chunks)} chunks")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    if all_chunks:
        store = VectorStoreFAISS()
        store.build_index(all_chunks)
        st.success("✅ FAISS index ready")

        query = st.text_input("🔍 Ask a question about the uploaded documents:")
        if query:
            retrieved = store.search(query, top_k=3)
            prompt = build_prompt(query, retrieved)

            # Choose LLM
            if model_choice == "OpenAI GPT-3.5":
                try:
                    answer = generate_answer_openai(prompt)
                except Exception as e:
                    st.error(f"OpenAI Error: {e}")
                    st.warning("⚠️ Falling back to HuggingFace local model...")
                    answer = generate_answer_hf(prompt)
            else:
                answer = generate_answer_hf(prompt)

            # Translate if Urdu selected
            if language == "Urdu":
                translator = Translator()
                answer = translator.translate(answer, src="en", dest="ur").text

            # Save to history
            st.session_state.chat_history.append(("User", query))
            st.session_state.chat_history.append(("Assistant", answer))

        # Display chat history
        st.subheader("💬 Chat History")
        for role, msg in st.session_state.chat_history:
            if role == "User":
                st.markdown(f"**🧑 {role}:** {msg}")
            else:
                st.markdown(f"**🤖 {role}:** {msg}")
                st.code(msg)

        if st.session_state.chat_history:
            pdf_buffer = export_chat_pdf(st.session_state.chat_history)
            st.download_button(
                "📥 Download Conversation as PDF",
                data=pdf_buffer,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )

        # Evaluation feedback
        st.subheader("📊 Feedback")
        feedback = st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"], index=None)
        if feedback:
            st.write("✅ Feedback recorded. Thank you!")
