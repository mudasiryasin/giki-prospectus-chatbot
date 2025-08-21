# 🤖 GIKI Prospectus Chatbot (RAG with Hugging Face)

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built using **Streamlit + FAISS + Hugging Face**.  
It allows users to upload **PDF, DOCX, or TXT documents** (e.g., GIKI prospectus) and then ask questions.  
The system retrieves relevant chunks of text and generates answers using **google/flan-t5-base** (instruction-tuned model).  

---

## ✨ Features
- 📂 Upload multiple files (PDF, DOCX, TXT)  
- 🔍 Automatic text cleaning, chunking, and FAISS vector search  
- 🧠 Question Answering with **Flan-T5** (lightweight, CPU-friendly)  
- 🌐 Answer translation (English ↔ Urdu)  
- 💬 Persistent chat history in session  
- 📥 Export chat as PDF  
- 👍/👎 User feedback collection  

---

## 🛠 Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/giki-prospectus-chatbot.git
cd giki-prospectus-chatbot