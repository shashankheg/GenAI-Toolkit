from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm import get_fast_llm



# Prompt template for translation
TRANSLATION_PROMPT = PromptTemplate(
    input_variables=["text", "source_lang", "target_lang"],
    template=(
                """You are an expert translator.
                    Translate the following text from {source_lang} to {target_lang}.
                    Preserve the original tone, style, and meaning as accurately as possible.

Text to translate:
{text}
Translation:"""
))

#Functions to translate text
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates text from source language to target language.
    Args:
        text (str): The text to translate.
        source_lang (str): The language of the input text.
        target_lang (str): The language to translate the text into.
    Returns:   
        str: The translated text.
    """
    llm = get_fast_llm()
    chain = TRANSLATION_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    })
    
    return result



