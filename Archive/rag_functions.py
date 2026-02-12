#!/usr/bin/env python3
"""
RAG Pipeline - Functional Implementation

A retrieval-augmented generation system using:
- Sentence Transformers for embeddings
- ChromaDB for vector storage
- Ollama + Llama 3.2 for generation

All implemented as functions rather than classes.
"""

import os
import glob
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# ============================================================================
# GLOBAL STATE (loaded once, reused across function calls)
# ============================================================================

_EMBEDDING_MODEL = None
_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get or initialize the embedding model.
    Cached globally to avoid reloading.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print(f"Loading embedding model: {model_name}...")
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        print("✓ Embedding model loaded\n")
    return _EMBEDDING_MODEL


def get_chroma_collection(collection_name: str = "presidents", 
                          db_path: str = "./chroma_db") -> chromadb.Collection:
    """
    Get or create ChromaDB collection.
    Cached globally to avoid reconnecting.
    """
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    
    if _CHROMA_COLLECTION is None:
        print("Initializing ChromaDB...")
        _CHROMA_CLIENT = chromadb.PersistentClient(path=db_path)
        
        try:
            _CHROMA_COLLECTION = _CHROMA_CLIENT.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
        except:
            _CHROMA_COLLECTION = _CHROMA_CLIENT.create_collection(name=collection_name)
            print(f"✓ Created new collection: {collection_name}")
        print()
    
    return _CHROMA_COLLECTION


# ============================================================================
# DOCUMENT LOADING
# ============================================================================

def load_documents(data_dir: str) -> List[Dict]:
    """
    Load all text documents from a directory.
    
    Args:
        data_dir: Path to directory containing .txt files
        
    Returns:
        List of document dictionaries with filename, content, path
    """
    documents = []
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    print(f"Loading documents from {data_dir}...")
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents.append({
            'filename': filename,
            'content': content,
            'path': filepath
        })
        print(f"  ✓ Loaded: {filename} ({len(content):,} chars)")
    
    print(f"Loaded {len(documents)} documents\n")
    return documents


# ============================================================================
# CHUNKING
# ============================================================================

def chunk_text(text: str, 
               chunk_size: int = 500, 
               chunk_overlap: int = 50,
               metadata: Optional[Dict] = None) -> List[Dict]:
    """
    Split text into overlapping chunks.
    
    Simple character-based chunking strategy.
    
    Args:
        text: The text to chunk
        chunk_size: Number of characters per chunk
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata to attach to each chunk
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Extract chunk
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Create chunk with metadata
        chunk = {
            'text': chunk_text,
            'metadata': metadata or {},
            'char_start': start,
            'char_end': end
        }
        chunks.append(chunk)
        
        # Move to next chunk with overlap
        start += chunk_size - chunk_overlap
    
    return chunks


def chunk_documents(documents: List[Dict],
                   chunk_size: int = 500,
                   chunk_overlap: int = 50) -> List[Dict]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of document dicts from load_documents()
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all chunks from all documents
    """
    print("Chunking documents...")
    all_chunks = []
    
    for doc in documents:
        metadata = {
            'filename': doc['filename'],
            'source': doc['path']
        }
        chunks = chunk_text(doc['content'], chunk_size, chunk_overlap, metadata)
        all_chunks.extend(chunks)
    
    print(f"✓ Created {len(all_chunks)} chunks\n")
    return all_chunks


# ============================================================================
# EMBEDDING & INDEXING
# ============================================================================

def embed_chunks(chunks: List[Dict], 
                model_name: str = "all-MiniLM-L6-v2") -> List:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Sentence transformer model name
        
    Returns:
        Numpy array of embeddings
    """
    model = get_embedding_model(model_name)
    
    print("Generating embeddings...")
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print()
    
    return embeddings


def store_in_chromadb(chunks: List[Dict],
                     embeddings: List,
                     collection_name: str = "presidents") -> int:
    """
    Store chunks and embeddings in ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: Corresponding embeddings
        collection_name: ChromaDB collection name
        
    Returns:
        Number of chunks stored
    """
    collection = get_chroma_collection(collection_name)
    
    print("Storing in ChromaDB...")
    
    # Prepare data
    chunk_texts = [chunk['text'] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Add in batches (ChromaDB has limits)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            documents=chunk_texts[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
    
    print(f"✓ Stored {len(chunks)} chunks in ChromaDB\n")
    return len(chunks)


def index_documents(data_dir: str,
                   collection_name: str = "presidents",
                   chunk_size: int = 500,
                   chunk_overlap: int = 50,
                   embedding_model: str = "all-MiniLM-L6-v2") -> int:
    """
    Complete indexing pipeline: load, chunk, embed, store.
    
    This is the main function to build your vector database.
    
    Args:
        data_dir: Directory containing text files
        collection_name: ChromaDB collection name
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        embedding_model: Sentence transformer model
        
    Returns:
        Number of chunks indexed
    """
    print("=" * 70)
    print("INDEXING PHASE")
    print("=" * 70)
    print()
    
    # Step 1: Load documents
    documents = load_documents(data_dir)
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    
    # Step 3: Generate embeddings
    embeddings = embed_chunks(chunks, embedding_model)
    
    # Step 4: Store in vector database
    num_indexed = store_in_chromadb(chunks, embeddings, collection_name)
    
    print("=" * 70)
    print(f"INDEXING COMPLETE: {num_indexed} chunks indexed")
    print("=" * 70)
    print()
    
    return num_indexed


# ============================================================================
# RETRIEVAL
# ============================================================================

def retrieve(query: str,
            n_results: int = 5,
            collection_name: str = "presidents",
            embedding_model: str = "all-MiniLM-L6-v2") -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query.
    
    Args:
        query: User's question
        n_results: Number of chunks to retrieve
        collection_name: ChromaDB collection name
        embedding_model: Model used for query embedding
        
    Returns:
        List of relevant chunks with metadata and similarity scores
    """
    # Get model and collection
    model = get_embedding_model(embedding_model)
    collection = get_chroma_collection(collection_name)
    
    # Embed the query
    query_embedding = model.encode([query])[0]
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    # Format results
    retrieved_chunks = []
    for i in range(len(results['ids'][0])):
        retrieved_chunks.append({
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None
        })
    
    return retrieved_chunks


# ============================================================================
# GENERATION
# ============================================================================

def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    """
    Build a prompt for the LLM with retrieved context.
    
    Args:
        query: User's question
        context_chunks: Retrieved relevant chunks
        
    Returns:
        Formatted prompt string
    """
    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk['metadata'].get('filename', 'unknown')
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create the prompt
    prompt = f"""You are a helpful assistant answering questions based on provided context.

Context information:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to answer the question fully, say so."""

    return prompt


def generate(query: str,
            context_chunks: List[Dict],
            llm_model: str = "llama3.2") -> Dict:
    """
    Generate an answer using Ollama with retrieved context.
    
    Args:
        query: User's question
        context_chunks: Retrieved relevant chunks
        llm_model: Ollama model name
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    # Build prompt
    prompt = build_prompt(query, context_chunks)
    
    # Call Ollama
    response = ollama.generate(
        model=llm_model,
        prompt=prompt
    )
    
    return {
        'answer': response['response'],
        'sources': [chunk['metadata'] for chunk in context_chunks],
        'model': llm_model
    }


# ============================================================================
# COMPLETE RAG QUERY
# ============================================================================

def query(question: str,
         n_results: int = 5,
         collection_name: str = "presidents",
         embedding_model: str = "all-MiniLM-L6-v2",
         llm_model: str = "llama3.2",
         verbose: bool = True) -> Dict:
    """
    Complete RAG pipeline: retrieve relevant chunks and generate answer.
    
    This is the main function to answer questions.
    
    Args:
        question: User's question
        n_results: Number of chunks to retrieve
        collection_name: ChromaDB collection name
        embedding_model: Model for embeddings
        llm_model: Ollama model for generation
        verbose: Whether to print progress
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"QUERY: {question}")
        print("=" * 70)
        print("\nStep 1: Retrieving relevant chunks...")
    
    # Retrieve
    chunks = retrieve(question, n_results, collection_name, embedding_model)
    
    if verbose:
        print(f"✓ Retrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('filename', 'unknown')
            preview = chunk['text'][:100].replace('\n', ' ')
            distance = chunk.get('distance', 0)
            print(f"  {i}. {source} (distance: {distance:.3f})")
            print(f"     {preview}...")
        
        print("\nStep 2: Generating answer with LLM...")
    
    # Generate
    result = generate(question, chunks, llm_model)
    
    if verbose:
        print(f"✓ Generated answer using {result['model']}\n")
    
    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_collection_stats(collection_name: str = "presidents") -> Dict:
    """
    Get statistics about the ChromaDB collection.
    
    Args:
        collection_name: Collection to inspect
        
    Returns:
        Dictionary with collection statistics
    """
    collection = get_chroma_collection(collection_name)
    count = collection.count()
    
    # Get sample of metadata to see sources
    sample = collection.peek(10)
    sources = set()
    if sample and 'metadatas' in sample:
        for metadata in sample['metadatas']:
            if 'filename' in metadata:
                sources.add(metadata['filename'])
    
    return {
        'collection_name': collection_name,
        'total_chunks': count,
        'sample_sources': list(sources)
    }


def reset_collection(collection_name: str = "presidents"):
    """
    Delete and recreate a collection (useful for re-indexing).
    
    Args:
        collection_name: Collection to reset
    """
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        _CHROMA_CLIENT.delete_collection(name=collection_name)
        print(f"✓ Deleted collection: {collection_name}")
    except:
        print(f"Collection {collection_name} did not exist")
    
    _CHROMA_COLLECTION = None
    print(f"✓ Collection reset. Call index_documents() to rebuild.")
