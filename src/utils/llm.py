import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

def get_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
    """Returns a ChatGroq instance for complex tasks."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=GROQ_API_KEY
    )

def get_fast_llm():
    """Faster model for simple tasks."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

def get_embeddings():
    """Returns OpenAI embeddings for vector store (still needs OpenAI key)."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )