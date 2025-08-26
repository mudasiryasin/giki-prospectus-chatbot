# 🤖 GIKI Prospectus Q&A Chatbot using Retrieval-Augmented Generation (RAG)

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot for answering questions from GIKI-related documents (e.g., prospectus, fee structure, academic rules).  
Users can upload up to **5 documents (PDF, DOCX, TXT)**, and the chatbot will extract, chunk, embed, and retrieve relevant information to provide context-aware answers.

---

## ✨ Features
- Upload up to **5 documents**
- Supports **PDF, DOCX, TXT**
- Document ingestion & text chunking
- Embedding generation with **MiniLM / E5 / Instructor-XL**
- Vector database storage with **FAISS / ChromaDB**
- Retrieval of **top-k relevant chunks**
- Answer generation using **LLM (OpenAI, Mistral, etc.)**
- Web-based interface (**Streamlit/Gradio**)
- Multiple languages **English/Urdu language toggle**
- Export chat history as PDF
- Evaluation with known Q&A pairs

---

## 📂 Project Structure

```
giki-rag-chatbot/
├── app.py # Main chatbot app
├── rag_pipeline.py # RAG pipeline
├── utils/ # Utilities (text processing, embeddings, retrieval)
├── requirements.txt/ # Libraries
└── README.md
```

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mudasiryasin/giki-prospectus-chatbot
cd giki-prospectus-chatbot
pip install -r requirements.txt
```

## ▶️ Usage

Run the chatbot interface:
```
streamlit run app.py
```

## 🧪 Evaluation

* Place test questions in `evaluation/eval.csv`.
* Run evaluation:
   ```
   python evaluation/eval_runner.py
   ```
Results will be saved in `evaluation/eval_results.json`.

## 📊 Results & Report

Detailed report is available in `docs/report.md`
.
It covers:

* Overview
* Methods
* Challenges
* Results
* Limitations
* Future improvements

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.

## 📜 License

MIT License © 2025