import streamlit as st
import requests

st.title("SLM Playground")
task = st.selectbox("Task", ["Translate to Thai", "Rephrase formally", "Extract as JSON"])
text = st.text_area("Input text")

if st.button("Run"):
    prompt = f"{task}: {text}"
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False
    })
    st.code(response.json()["response"])