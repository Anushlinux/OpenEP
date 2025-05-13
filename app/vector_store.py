# app/vector_store.py
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from .config import CHROMA_PERSIST_DIRECTORY, DEFAULT_COLLECTION_NAME

# Initialize embedding model (consider moving to a central place if used elsewhere)
# Using a small, efficient model for local development
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

def get_or_create_collection(collection_name: str = DEFAULT_COLLECTION_NAME):
    return client.get_or_create_collection(name=collection_name, embedding_function=None) # We'll provide embeddings directly

def add_documents_to_collection(collection_name: str, doc_chunks: List[str], pdf_id: str):
    """
    Adds document chunks to the specified ChromaDB collection.
    pdf_id is used to create unique IDs for chunks from this PDF.
    """
    collection = get_or_create_collection(collection_name)
    chunk_embeddings = embeddings.embed_documents(doc_chunks)
    
    # Create unique IDs for each chunk and metadata
    ids = [f"{pdf_id}_chunk_{i}" for i in range(len(doc_chunks))]
    metadatas = [{"pdf_id": pdf_id, "source_chunk_index": i} for i in range(len(doc_chunks))]

    collection.add(
        embeddings=chunk_embeddings,
        documents=doc_chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(doc_chunks)} chunks from PDF {pdf_id} to collection {collection_name}.")

def query_collection(collection_name: str, query_text: str, n_results: int = 5) -> List[str]:
    """Queries the collection and returns relevant document chunks."""
    collection = get_or_create_collection(collection_name)
    query_embedding = embeddings.embed_query(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents'] # We only need the document text for the LLM
    )
    
    # Results['documents'] is a list of lists, get the first list
    return results['documents'][0] if results['documents'] else []

