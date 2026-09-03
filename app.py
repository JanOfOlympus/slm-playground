import streamlit as st
import requests
import numpy as np

st.title("SLM Playground")

# --- Sample corpus (dummy docs for the demo) ---
DOCS = [
    "The company's return policy allows refunds within 30 days of purchase with a valid receipt.",
    "Employees are entitled to 15 days of paid annual leave per year, accrued monthly.",
    "The office WiFi password is changed every quarter; check with IT for the current one.",
    "Standard shipping takes 3-5 business days; express shipping takes 1-2 business days.",
    "The Q3 budget review meeting is scheduled for the last Friday of each quarter.",
]

def embed(text):
    resp = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
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

task = st.selectbox("Task", ["Rephrase formally", "Extract as JSON", "RAG Q&A"])
text = st.text_area("Input text")

if st.button("Run"):
    if task == "RAG Q&A":
        results = retrieve(text)
        context = "\n".join([f"- {doc}" for doc, score in results])

        with st.expander("Retrieved context (debug)"):
            for doc, score in results:
                st.write(f"**{score:.3f}** — {doc}")

        prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {text}

Answer:"""
    else:
        prompt = f"{task}: {text}"

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False
    })
    st.code(response.json()["response"], height=200)