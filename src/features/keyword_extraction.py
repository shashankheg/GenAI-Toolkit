from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm import get_fast_llm

KEYWORD_PROMPT = PromptTemplate(
    input_variables=["text", "num_keywords"],
    template="""You are an expert at extracting the most important keywords and key phrases from text.
    Extract the top {num_keywords} keywords and key phrases from the following text.
    Return them as a numbered list, with a brief explanation for each keyword.

Text:
{text}

Keywords:"""
)


def extract_keywords(text: str, num_keywords: int = 10) -> str:
    """
    Extracts keywords from the given text.

    Args:
        text: Text to extract keywords from
        num_keywords: Number of keywords to extract

    Returns:
        Numbered list of keywords with explanations
    """
    if not text.strip():
        return "⚠️ Please enter text to extract keywords from."

    if len(text.split()) < 10:
        return "⚠️ Please enter more text for meaningful keyword extraction."

    llm = get_fast_llm()
    chain = KEYWORD_PROMPT | llm | StrOutputParser()

    result = chain.invoke({
        "text": text,
        "num_keywords": num_keywords
    })

    return result