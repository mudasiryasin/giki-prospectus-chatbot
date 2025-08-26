import os
from typing import List, Dict, Tuple
import streamlit as st
from transformers import pipeline
from openai import OpenAI
from utils.text_processing import parse_pdf, parse_docx, parse_txt
from utils.retrieval import search_top_k
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

# Load variables from .env into os.environ
load_dotenv()

def ingest_files(uploaded_files: List[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    all_chunks: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    for uf in uploaded_files:
        name = uf.name
        data = uf.read()
        if name.lower().endswith(".pdf"):
            pairs = parse_pdf(data, name)
        elif name.lower().endswith(".docx"):
            pairs = parse_docx(data, name)
        elif name.lower().endswith(".txt"):
            pairs = parse_txt(data, name)
        else:
            st.warning(f"Unsupported file type: {name}")
            continue
        for chunk, meta in pairs:
            all_chunks.append(chunk)
            all_meta.append(meta)
    return all_chunks, all_meta

def format_context(metas: List[Dict[str, Any]]) -> str:
    lines = []
    for m in metas:
        src = m.get("source", "?")
        page = m.get("page")
        tag = f"{src} p.{page}" if page else src
        lines.append(f"[{tag}]\n{m['text']}")
    return "\n\n".join(lines)


def build_system_prompt(language: str) -> str:
    lang_hint = "Respond in Urdu." if language.lower().startswith("urdu") else "Respond in English."
    return (
        "You are a helpful assistant for GIKI documents. Answer ONLY using the provided context.\n"
        "- If the answer is not in the context, say you couldn't find it in the uploaded documents.\n"
        "- When you state facts, cite sources inline like [source: <filename> p.<page>].\n"
        f"- {lang_hint}\n"
        "- Keep answers concise and directly relevant to the question.\n"
    )


def build_user_prompt(question: str, context: str) -> str:
    return (
        "Context:\n" + context + "\n\n" +
        "Question: " + question + "\n" +
        "Instructions: Use ONLY the context above. If missing, say it's not found in the documents."
    )


def use_openai() -> bool:
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    return bool(key and OpenAI is not None)


def openai_chat(messages: List[Dict[str, str]], model_name: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 500) -> str:
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    client = OpenAI(api_key=key)
    # Try chat.completions first, then fallback to responses API
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        resp = client.responses.create(
            model=model_name,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # unify content extraction
        if resp.output and len(resp.output) > 0 and hasattr(resp.output[0], "content"):
            # new SDK object structure
            parts = resp.output[0].content
            if parts and len(parts) > 0 and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        # fallback
        return str(resp)


def local_generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
    """Local tiny model for demo when no OpenAI key is set."""
    gen = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device_map="auto",
    )
    full = f"{system_prompt}\n\n{user_prompt}"
    out = gen(full, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"].strip()
    return out


def answer_question(question: str, language: str, model_choice: str, k: int = 5, temperature: float = 0.0) -> Tuple[str, List[Dict[str, Any]]]:
    idxs, metas = search_top_k(question, k=k, use_mmr=True)
    if not metas:
        return ("I couldn't find relevant information. Please upload documents first.", [])
    ctx = format_context(metas)
    system = build_system_prompt(language)
    user = build_user_prompt(question, ctx)

    if model_choice.startswith("OpenAI") and use_openai():
        model_map = {
            "OpenAI: gpt-4o-mini": "gpt-4o-mini",
            "OpenAI: gpt-4o": "gpt-4o",
        }
        model_name = model_map.get(model_choice, "gpt-4o-mini")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = openai_chat(messages, model_name=model_name, temperature=temperature)
    else:
        resp = local_generate(system, user)

    return resp, metas