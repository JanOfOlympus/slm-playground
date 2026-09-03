import streamlit as st
import requests
import numpy as np

st.title("SLM Agent Playground")

# --- Sample corpus (dummy docs for the demo) ---
DOCS = [
    "The company's return policy allows refunds within 30 days of purchase with a valid receipt.",
    "Employees are entitled to 15 days of paid annual leave per year, accrued monthly.",
    "The office WiFi password is changed every quarter; check with IT for the current one.",
    "Standard shipping takes 3-5 business days; express shipping takes 1-2 business days.",
    "The Q3 budget review meeting is scheduled for the last Friday of each quarter.",
]

OLLAMA_URL = "http://localhost:11434"
GEN_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"


def call_model(prompt, model=GEN_MODEL):
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    resp.raise_for_status()
    return resp.json()["response"].strip()


def embed(text):
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    resp.raise_for_status()
    return np.array(resp.json()["embedding"])


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@st.cache_resource
def build_index():
    return [embed(doc) for doc in DOCS]


def retrieve(query, top_k=2):
    query_vec = embed(query)
    doc_vecs = build_index()
    sims = [cosine_sim(query_vec, v) for v in doc_vecs]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(DOCS[i], sims[i]) for i in top_indices]


def classify_task(text):
    prompt = f"""Classify the input into exactly ONE category. Respond with ONLY the category word, nothing else.

Categories:
- rephrase: informal/casual text that could be made more formal, with no question being asked
- extract: text containing structured data like amounts, dates, names, invoices, receipts
- rag_qa: a question likely about company policy, leave, shipping, WiFi, or budget meetings
- none: doesn't fit any of the above (jokes, small talk, unrelated topics, general knowledge questions)

Input: {text}

Category:"""
    result = call_model(prompt).lower().strip()
    for valid in ["rephrase", "extract", "rag_qa", "none"]:
        if valid in result:
            return valid
    return "none"  # safer fallback when classification is unclear


def strip_json_fences(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    end = raw.rfind("}")
    if end != -1:
        raw = raw[:end + 1]
    return raw.strip()


st.write("Paste any input — the agent decides what to do with it.")
text = st.text_area("Input text")

if st.button("Run") and text.strip():
    with st.spinner("Classifying..."):
        task = classify_task(text)
    st.info(f"🤖 Agent decided: **{task}**")

    if task == "none":
        st.warning("This doesn't match a supported task (rephrase, extract, or company Q&A). No action taken.")
    else:
        if task == "rag_qa":
            results = retrieve(text)
            context = "\n".join([f"- {doc}" for doc, score in results])
            with st.expander("Retrieved context (debug)"):
                for doc, score in results:
                    st.write(f"**{score:.3f}** — {doc}")
            prompt = f"""Answer using ONLY the context below. If not covered, say so.

Context:
{context}

Question: {text}

Answer:"""
        elif task == "extract":
            prompt = f"""Today's date is 2026-09-03. Extract as JSON. Return ONLY the JSON object, no markdown, no explanation.

Text: {text}"""
        else:  # rephrase
            prompt = f"Rephrase more formally: {text}"

        with st.spinner("Generating..."):
            result = call_model(prompt)
            if task == "extract":
                result = strip_json_fences(result)

        st.code(result, height=200)