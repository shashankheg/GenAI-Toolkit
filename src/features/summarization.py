from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.llm import get_llm, get_fast_llm

# Prompt for short text summarization

SHORT_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert at summarizing content clearly and concisely.

            Summarize the following text in a clear, concise way. 
            Capture the key points and main ideas.

    Text : {text}

    Summary :"""

)


# Prompt for summarizing each chunk (map step)

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Summarize the following section, capturing all key points:

{text}

Section Summary:"""
)


# Prompt for combining chunk summaries (reduce step)
REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are given multiple summaries of different sections of a document.
Combine them into one final, coherent summary that captures all key points.

Summaries:
{text}

Final Summary:"""
)

def summarize_text(text: str, mode: str = "concise") -> str:
    """
    Summarizes text. Handles both short and long documents.

    Args:
        text: Text to summarize
        mode: 'concise' for short summary, 'detailed' for longer summary

    Returns:
        Summary string
    """
    if not text.strip():
        return "⚠️ Please enter text to summarize."
    
    if len(text.split())<1000 :
        llm =get_fast_llm()
        chain = SHORT_SUMMARY_PROMPT | llm | StrOutputParser()
        return chain.invoke({"text": text})
    #Use MAP REDUCE concept for longer string 

    return _map_reduce_summarize(text, mode)

def _map_reduce_summarize(text: str, mode: str) -> str:
    """Handles long document summarization using map-reduce."""
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    print(f"📄 Split into {len(chunks)} chunks for summarization...")

    llm = get_fast_llm()
    parser = StrOutputParser()

    # Map — summarize each chunk
    map_chain = MAP_PROMPT | llm | parser
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = map_chain.invoke({"text": chunk})
        chunk_summaries.append(summary)
        print(f"📝 Summarized chunk {i + 1}/{len(chunks)}")

    # Reduce — combine all chunk summaries
    reduce_chain = REDUCE_PROMPT | llm | parser
    final_summary = reduce_chain.invoke({"text": "\n".join(chunk_summaries)})

    return final_summary