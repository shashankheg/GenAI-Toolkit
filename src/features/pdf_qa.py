import fitz  # PyMuPDF
from typing import Annotated, TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.llm import get_llm
from src.utils.vector_store import create_vector_store

# === PROMPT ===
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert at answering questions based on provided document context.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
)


# === STATE ===
class PDFQAState(TypedDict):
    pdf_text: str
    question: str
    context: str
    answer: str

# === PDF LOADING ===
def load_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# === NODES ===
def retrieve_node(state: PDFQAState) -> PDFQAState:
    """Retrieves relevant chunks from the PDF."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(state["pdf_text"])
    vector_store = create_vector_store(chunks)

    # Get top 3 relevant chunks
    docs = vector_store.similarity_search(state["question"], k=3)
    context = "\n\n".join([doc.page_content for doc in docs]) 
    return {**state, "context": context}

def answer_node(state: PDFQAState) -> PDFQAState:
    """Generates answer from context."""
    llm = get_llm()
    prompt = QA_PROMPT
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    answer = chain.invoke(
        {"context": state["context"],
          "question": state["question"]}
          )

    return {**state, "answer": answer}


#building the graph 
def build_pdf_qa_graph():
    graph = StateGraph(PDFQAState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()



# === MAIN FUNCTION ===
def ask_pdf(pdf_path: str, question: str) -> str:
    """
    Ask a question about a PDF document.

    Args:
        pdf_path: Path to the PDF file
        question: Question to ask

    Returns:
        Answer string
    """
    if not question.strip():
        return "⚠️ Please enter a question."

    print(f"📄 Loading PDF: {pdf_path}")
    pdf_text = load_pdf(pdf_path)

    if not pdf_text.strip():
        return "⚠️ Could not extract text from PDF."

    print(f"🔍 Finding relevant context...")
    graph = build_pdf_qa_graph()
    result = graph.invoke({
        "pdf_text": pdf_text,
        "question": question,
        "context": "",
        "answer": ""
    })

    return result["answer"]
