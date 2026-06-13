import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Use free HuggingFace embeddings instead of OpenAI

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )



def create_vector_store(chunks):
    """Creates FAISS vector store from text chunks."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    return vector_store

def save_vector_store(vector_store, path: str):
    """Saves vector store to disk."""
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(path: str):
    """Loads vector store from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

