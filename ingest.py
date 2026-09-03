"""
Ingests a text document into a persistent ChromaDB vector store.
Run this once (or whenever the source document changes) to (re)build the index.

Usage:
    python ingest.py
"""

import re
import requests
import chromadb

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DOC_PATH = "hr_policy.txt"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "hr_policy"


def embed(text: str) -> list[float]:
    """Call Ollama's embedding endpoint and return the vector as a plain list."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def chunk_document(path: str) -> list[str]:
    """
    Simple chunking strategy: split on 'SECTION N:' headers.
    Each chunk is a full section (header + body) — fine for a document
    with a small, predictable number of clearly-delimited sections.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = re.split(r"(?=SECTION \d+:)", text)
    chunks = [p.strip() for p in parts if "SECTION" in p]
    return chunks


def main():
    print(f"Reading document: {DOC_PATH}")
    chunks = chunk_document(DOC_PATH)
    print(f"Split into {len(chunks)} chunks")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Start clean each run so re-ingesting doesn't create duplicate entries
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # collection didn't exist yet — fine

    collection = client.create_collection(name=COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}...")
        vector = embed(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": DOC_PATH, "chunk_index": i}],
        )

    print(f"\nDone. {collection.count()} chunks stored in '{CHROMA_PATH}' "
          f"under collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
