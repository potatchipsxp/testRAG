#!/usr/bin/env python3
"""
build_index.py

Index log data into ChromaDB using a specific embedding + chunking config.
Edit the CONFIG section below, then run:

    python build_index.py

Each unique combination of settings gets its own ChromaDB collection,
so you can build multiple indexes without overwriting each other.
"""

import os
import glob
import json
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================================
# CONFIG — edit these before running
# ============================================================================

DATA_DIR          = "./data"           # folder containing your .txt log files
DB_PATH           = "./chroma_db"      # where ChromaDB persists to disk
BASE_COLLECTION   = "logs"             # prefix for collection names
COLLECTION_SUFFIX = ""                 # optional label, e.g. "fast" or "test"
FORCE_REINDEX     = False              # True = always delete and rebuild

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # sentence-transformers model name
CHUNK_METHOD      = "character"           # "character", "sentence", or "paragraph"
CHUNK_SIZE        = 500                   # chars / sentences / paragraphs depending on method
CHUNK_OVERLAP     = 50                    # same unit as chunk_size


# ============================================================================
# COLLECTION NAME HELPER
# ============================================================================

def make_collection_name(base, embedding_model, chunk_size, chunk_overlap,
                         chunk_method, suffix=""):
    model_short = embedding_model.split("/")[-1].replace("-", "_")
    parts = [base, model_short, f"chunk{chunk_size}",
             f"overlap{chunk_overlap}", chunk_method]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


# ============================================================================
# CHROMA HELPERS
# ============================================================================

def get_chroma_client(db_path=DB_PATH):
    return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(client, name):
    try:
        col = client.get_collection(name=name)
        print(f"  Loaded existing collection: {name}")
    except Exception:
        col = client.create_collection(name=name)
        print(f"  Created new collection: {name}")
    return col


def delete_collection(client, name):
    try:
        client.delete_collection(name=name)
        print(f"  Deleted collection: {name}")
    except Exception:
        print(f"  Collection not found, skipping delete: {name}")


def list_collections(db_path=DB_PATH):
    client = get_chroma_client(db_path)
    return [{"name": c.name, "count": c.count()} for c in client.list_collections()]


# ============================================================================
# DOCUMENT LOADING
# ============================================================================

def load_documents(data_dir):
    documents = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append({"filename": filename, "content": content, "path": filepath})
        print(f"  Loaded: {filename}  ({len(content):,} chars)")
    print(f"  Total documents: {len(documents)}\n")
    return documents


# ============================================================================
# CHUNKING STRATEGIES
# ============================================================================

def chunk_by_characters(text, chunk_size, chunk_overlap, metadata=None):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "text": text[start:end],
            "metadata": dict(metadata or {}),
            "char_start": start,
            "char_end": end,
        })
        start += chunk_size - chunk_overlap
    return chunks


def chunk_by_sentences(text, sentences_per_chunk, overlap_sentences, metadata=None):
    import re
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.strip() for s in raw if s.strip()]

    chunks = []
    i = 0
    while i < len(sentences):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append({
            "text": " ".join(group),
            "metadata": dict(metadata or {}),
            "sentence_start": i,
            "sentence_end": i + len(group),
        })
        i += sentences_per_chunk - overlap_sentences
    return chunks


def chunk_by_paragraphs(text, paragraphs_per_chunk, overlap_paragraphs, metadata=None):
    import re
    raw = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw if p.strip()]

    chunks = []
    i = 0
    while i < len(paragraphs):
        group = paragraphs[i : i + paragraphs_per_chunk]
        chunks.append({
            "text": "\n\n".join(group),
            "metadata": dict(metadata or {}),
            "para_start": i,
            "para_end": i + len(group),
        })
        i += paragraphs_per_chunk - max(0, overlap_paragraphs)
    return chunks


def chunk_text(text, chunk_method, chunk_size, chunk_overlap, metadata=None):
    if chunk_method == "character":
        return chunk_by_characters(text, chunk_size, chunk_overlap, metadata)
    elif chunk_method == "sentence":
        return chunk_by_sentences(text, chunk_size, chunk_overlap, metadata)
    elif chunk_method == "paragraph":
        return chunk_by_paragraphs(text, chunk_size, chunk_overlap, metadata)
    else:
        raise ValueError(f"Unknown chunk_method: {chunk_method}")


def chunk_documents(documents, chunk_method, chunk_size, chunk_overlap):
    all_chunks = []
    for doc in documents:
        metadata = {"filename": doc["filename"], "source": doc["path"]}
        chunks = chunk_text(doc["content"], chunk_method, chunk_size, chunk_overlap, metadata)
        all_chunks.extend(chunks)
        print(f"  {doc['filename']}: {len(chunks)} chunks")
    print(f"  Total chunks: {len(all_chunks)}\n")
    return all_chunks


# ============================================================================
# INDEXING
# ============================================================================

def build_index(data_dir, db_path, collection_name, embedding_model,
                chunk_method, chunk_size, chunk_overlap, force_reindex=False):
    """
    Load documents, chunk them, embed them, and store in ChromaDB.
    Returns the number of chunks indexed.
    """
    print("=" * 70)
    print(f"BUILD INDEX")
    print(f"  Collection : {collection_name}")
    print(f"  Model      : {embedding_model}")
    print(f"  Chunking   : {chunk_method}  size={chunk_size}  overlap={chunk_overlap}")
    print("=" * 70)

    client = get_chroma_client(db_path)

    # Handle existing collection
    try:
        existing = client.get_collection(name=collection_name)
        if existing.count() > 0:
            if force_reindex:
                print(f"\n  Collection exists ({existing.count()} chunks) — deleting (FORCE_REINDEX=True)")
                delete_collection(client, collection_name)
            else:
                print(f"\n  Collection already has {existing.count()} chunks.")
                print("  Set FORCE_REINDEX = True to rebuild, or change config for a new collection.")
                return existing.count()
    except Exception:
        pass  # collection doesn't exist yet, that's fine

    collection = get_or_create_collection(client, collection_name)

    # Load → chunk → embed → store
    print("\nLoading documents...")
    documents = load_documents(data_dir)

    print("Chunking documents...")
    chunks = chunk_documents(documents, chunk_method, chunk_size, chunk_overlap)

    print(f"Generating embeddings with {embedding_model}...")
    model = SentenceTransformer(embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print()

    print("Storing in ChromaDB...")
    config_meta = {
        "embedding_model": embedding_model,
        "chunk_method": chunk_method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [dict(c["metadata"], config=json.dumps(config_meta)) for c in chunks]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        j = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:j],
            embeddings=embeddings[i:j].tolist(),
            documents=texts[i:j],
            metadatas=metadatas[i:j],
        )

    print(f"\n  Indexed {len(chunks)} chunks into '{collection_name}'")
    print("=" * 70)
    return len(chunks)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    collection_name = make_collection_name(
        BASE_COLLECTION, EMBEDDING_MODEL, CHUNK_SIZE,
        CHUNK_OVERLAP, CHUNK_METHOD, COLLECTION_SUFFIX
    )

    build_index(
        data_dir        = DATA_DIR,
        db_path         = DB_PATH,
        collection_name = collection_name,
        embedding_model = EMBEDDING_MODEL,
        chunk_method    = CHUNK_METHOD,
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        force_reindex   = FORCE_REINDEX,
    )
