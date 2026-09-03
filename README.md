# slm-playground

## .venv
- py -m venv .venv
- .venv\Scripts\Activate.ps1
- deactivate

## pip
- py -m pip install streamlit requests numpy chromadb

## streamlit
- streamlit run app.py

## model useds
- command: ollama pull <models>
- decoding model: qwen2.5:3b
- embedding model: nomic-embed-text

## vector store
- chroma