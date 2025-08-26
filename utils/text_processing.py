import re
import io
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from typing import List, Tuple, Dict, Any

def sentence_split(text: str) -> List[str]:
    # Simple sentence splitter: splits on ., !, ?, or line breaks, keeping delimiters
    parts = re.split(r"([.!?])", text)
    sentences = []
    for i in range(0, len(parts), 2):
        sent = parts[i]
        if i + 1 < len(parts):
            sent += parts[i + 1]
        sent = sent.strip()
        if sent:
            sentences.append(sent)
    # Also split long lines by newlines
    final = []
    for s in sentences:
        chunks = [c.strip() for c in s.splitlines() if c.strip()]
        final.extend(chunks if chunks else [s])
    return final


def chunk_sentences(sentences: List[str], target_chars: int = 1800, overlap_chars: int = 300) -> List[str]:
    """Greedy chunk by char length (≈450 tokens proxy)."""
    chunks = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= target_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            # start new with overlap from tail of previous buffer
            if overlap_chars > 0 and chunks:
                tail = chunks[-1][-overlap_chars:]
                buf = (tail + " " + s).strip()
            else:
                buf = s
    if buf:
        chunks.append(buf)
    return chunks


def parse_pdf(file: bytes, filename: str) -> List[Tuple[str, Dict[str, Any]]]:
    doc = fitz.open(stream=file, filetype="pdf")
    results = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if not text:
            continue
        sentences = sentence_split(text)
        for chunk in chunk_sentences(sentences):
            results.append((chunk, {"source": filename, "page": i + 1}))
    return results


def parse_docx(file: bytes, filename: str) -> List[Tuple[str, Dict[str, Any]]]:
    with io.BytesIO(file) as bio:
        doc = DocxDocument(bio)
    full_text = []
    for para in doc.paragraphs:
        txt = para.text.strip()
        if txt:
            full_text.append(txt)
    text = "\n".join(full_text)
    sentences = sentence_split(text)
    results = [(chunk, {"source": filename, "page": None}) for chunk in chunk_sentences(sentences)]
    return results


def parse_txt(file: bytes, filename: str) -> List[Tuple[str, Dict[str, Any]]]:
    try:
        text = file.decode("utf-8", errors="ignore")
    except Exception:
        text = file.decode(errors="ignore")
    sentences = sentence_split(text)
    results = [(chunk, {"source": filename, "page": None}) for chunk in chunk_sentences(sentences)]
    return results
