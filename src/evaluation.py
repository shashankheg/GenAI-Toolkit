import os
import json
import time 
from typing import Dict, List, Any
from dotenv import load_dotenv
from src.utils.llm import get_fast_llm

load_dotenv()

from src.features.translation import translate_text
from src.features.summarization import summarize_text
from src.features.keyword_extraction import extract_keywords
from src.features.email_writer import write_email, write_cover_letter
from src.graphs.chat_graph import ChatSession



# ── SAMPLE TEST DATA ────────────────────────────────────────────────────────


TRANSLATION_TESTS = [
    {
        "text": "Artificial intelligence is transforming the world.",
        "source": "English", "target": "French",
        "reference": "L'intelligence artificielle transforme le monde."
    },

    {
        "text": "Machine learning enables computers to learn from data.",
        "source": "English", "target": "Spanish",
        "reference": "El aprendizaje automático permite a las computadoras aprender de los datos."
    },

    {
        "text": "Deep learning uses neural networks to solve complex problems.",
        "source": "English", "target": "German",
        "reference": "Deep Learning verwendet neuronale Netze, um komplexe Probleme zu lösen."
    },

    
]


SUMMARIZATION_TESTS = [

    {"text": """Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to the natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, 
        which refers to any system that perceives its environment and takes actions 
        that maximize its chance of achieving its goals. The term "artificial intelligence" 
        had previously been used to describe machines that mimic and display human cognitive 
        skills associated with the human mind, such as learning and problem-solving. 
        This definition has since been rejected by major AI researchers who now describe 
        AI in terms of rationality and acting rationally, which does not limit how 
        intelligence can be articulated. AI applications include advanced web search engines, 
        recommendation systems, understanding human speech, self-driving cars, generative 
        tools and competing at the highest level in strategic games.""",
        "reference": "AI is machine-demonstrated intelligence that perceives environments and takes goal-maximizing actions, with applications ranging from search engines to self-driving cars."
    },
    {
        "text": """LangChain is a framework designed to simplify the creation of applications 
        using large language models. As a language model integration framework, LangChain's 
        use-cases largely overlap with those of language models in general, including document 
        analysis and summarization, chatbots, and code analysis. LangChain was launched in 
        October 2022 as an open source project. It provides a standard interface for chains, 
        lots of integrations with other tools, and end-to-end chains for common applications. 
        The framework allows AI developers to develop applications based on combining LLMs 
        with other sources of computation or knowledge.""",
        "reference": "LangChain is an open-source framework that simplifies building LLM applications through standard interfaces, tool integrations, and pre-built chains."
    },
]


KEYWORD_TESTS = [
    {
        "text": """Natural language processing (NLP) is a subfield of linguistics, 
        computer science, and artificial intelligence concerned with the interactions 
        between computers and human language, in particular how to program computers 
        to process and analyze large amounts of natural language data. The goal is 
        a computer capable of understanding the contents of documents, including the 
        contextual nuances of the language within them.""",
        "expected_keywords": ["natural language processing", "NLP", "artificial intelligence", 
                               "computer science", "linguistics", "language", "computers"]
    },
]


EMAIL_TESTS = [
    {
        "type": "follow-up",
        "context": "Following up on a job application for ML Engineer position submitted last week",
        "tone": "professional",
        "recipient": "Hiring Manager",
        "sender": "Shashank Hegde",
        "check_words": ["follow", "application", "ML Engineer", "Shashank"]
    },

{
        "type": "thank you",
        "context": "Thanking the interviewer after a data science interview",
        "tone": "professional",
        "recipient": "Dr. Smith",
        "sender": "Shashank",
        "check_words": ["thank", "interview", "opportunity"]
    },
]

CHAT_TESTS = [
    {"input": "What is machine learning?", "check": ["learning", "data", "model", "algorithm"]},
    {"input": "Explain neural networks in simple terms.", "check": ["network", "neuron", "layer", "brain"]},
    {"input": "What are the differences between supervised and unsupervised learning?", 
     "check": ["supervised", "unsupervised", "label", "data"]},
]

COVER_LETTER_TESTS = [
    {
        "job": "ML Engineer",
        "company": "Google DeepMind",
        "skills": "Python, PyTorch, LangChain, MLflow",
        "experience": "3 years building ML pipelines and deploying models to production",
        "name": "Shashank Hegde",
        "check_words": ["ML Engineer", "Google", "Python", "Shashank"]
    }
]


# ── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def simple_bleu( hypothesis:str, reference:str) -> float:
    """Simple word-overlap BLEU approximation (no external library needed)."""
    hyp_words = set(hypothesis.lower().split())
    ref_words = set(reference.lower().split())

    if not ref_words:
        return 0.0

    overlap  = hyp_words & ref_words
    return len(overlap) / len(ref_words)   
    

def rouge_L(hypothesis:str, reference:str) -> float:
    """Simple ROUGE-L approximation (no external library needed)."""
    # This is a simplified version of ROUGE-L, which measures the longest common subsequence
    # between the hypothesis and reference.
    hyp = list(hypothesis.lower())
    ref = list(reference.lower())

    if not ref:
        return 0.0

    common = set(hyp) & set(ref)
    precision = len(common) / len(hyp) if hyp else 0
    recall = len(common) / len(ref) if ref else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def keyword_precision(extracted: str, expected: List[str], k: int = 10) -> float:
    """Check how many expected keywords appear in extracted output."""
    extracted_lower = extracted.lower()
    found = sum(1 for kw in expected if kw.lower() in extracted_lower)
    return found / len(expected)


def check_words_present(text: str, check_words: List[str]) -> float:
    """Check what fraction of expected words appear in the output."""
    text = text.lower()
    found = sum(1 for w in check_words if w.lower() in text)
    return found / len(check_words) 


def llm_judge(question: str, response: str) -> Dict[str, Any]:
    """Use Groq LLM to evaluate chat response quality."""
    from src.utils.llm import get_fast_llm
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate(
        input_variables=["question", "response"],
        template="""Evaluate this AI response. Return ONLY a JSON object, no extra text.
    Question: {question}
    Response: {response}

    Return exactly this JSON format:
    {{"relevance": <1-10>, "accuracy": <1-10>, "clarity": <1-10>, "overall": <1-10>, "comment": "<one sentence>"}}"""
    )
    try:
        llm = get_fast_llm()
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "response": response})
        # Extract JSON from response
        start = result.find("{")
        end = result.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(result[start:end])

    except Exception as e:
        pass

    return {"relevance": 0, "accuracy": 0, "clarity": 0, "overall": 0, "comment": "Error occurred while evaluating the response."}

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(label: str, score: float, details: str = ""):
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"  {label:<25} [{bar}] {score:.2%}  {details}")

# ── EVALUATION FUNCTIONS ─────────────────────────────────────────────────────

def evaluate_translation() -> Dict:
    print_header("🌐 TRANSLATION EVALUATION")
    scores = []

    for i, test in enumerate(TRANSLATION_TESTS, 1):
        print(f"\n  Test {i}: English → {test['target']}")
        print(f"  Input:     {test['text'][:60]}...")

        t0 = time.time()
        result = translate_text(test["text"], test["source"], test["target"])
        latency = time.time() - t0

        score = simple_bleu(result, test["reference"])
        scores.append(score)

        print(f"  Output:    {result[:80]}...")
        print(f"  Reference: {test['reference'][:80]}...")
        print_result("Word Overlap Score", score, f"| {latency:.2f}s")

    avg = sum(scores) / len(scores)
    print(f"\n  ✅ Average Score: {avg:.2%}")
    return {"feature": "Translation", "avg_score": avg, "scores": scores}

def evaluate_summarization() -> Dict:
    print_header("📝 SUMMARIZATION EVALUATION")
    scores = []

    for i, test in enumerate(SUMMARIZATION_TESTS, 1):
        print(f"\n  Test {i}: Document ({len(test['text'].split())} words)")

        t0 = time.time()
        result = summarize_text(test["text"], "concise")
        latency = time.time() - t0

        score = rouge_L(result, test["reference"])
        scores.append(score)

        print(f"  Summary:   {result[:100]}...")
        print(f"  Reference: {test['reference'][:100]}...")
        print_result("ROUGE-L Score", score, f"| {latency:.2f}s")

    avg = sum(scores) / len(scores)
    print(f"\n  ✅ Average Score: {avg:.2%}")
    return {"feature": "Summarization", "avg_score": avg, "scores": scores}

def evaluate_keywords() -> Dict:
    print_header("🔍 KEYWORD EXTRACTION EVALUATION")
    scores = []

    for i, test in enumerate(KEYWORD_TESTS, 1):
        print(f"\n  Test {i}: ({len(test['text'].split())} words)")

        t0 = time.time()
        result = extract_keywords(test["text"], num_keywords=10)
        latency = time.time() - t0

        score = keyword_precision(result, test["expected_keywords"])
        scores.append(score)

        print(f"  Output (first 150 chars): {result[:150]}...")
        print_result("Keyword Precision", score, f"| {latency:.2f}s")

    avg = sum(scores) / len(scores)
    print(f"\n  ✅ Average Score: {avg:.2%}")
    return {"feature": "Keywords", "avg_score": avg, "scores": scores}


def evaluate_email_writer() -> Dict:
    print_header("✉️ EMAIL WRITER EVALUATION")
    scores = []

    # Email tests
    print("\n  📧 Email Generation:")
    for i, test in enumerate(EMAIL_TESTS, 1):
        t0 = time.time()
        result = write_email(
            test["type"], test["context"],
            test["tone"], test["recipient"], test["sender"]
        )
        latency = time.time() - t0

        score = check_words_present(result, test["check_words"])
        scores.append(score)
        print(f"\n  Test {i} ({test['type']}):")
        print(f"  Output: {result[:120]}...")
        print_result("Content Check", score, f"| {latency:.2f}s")

    # Cover letter test
    print("\n  📄 Cover Letter Generation:")
    for test in COVER_LETTER_TESTS:
        t0 = time.time()
        result = write_cover_letter(
            test["job"], test["company"],
            test["skills"], test["experience"], test["name"]
        )
        latency = time.time() - t0

        score = check_words_present(result, test["check_words"])
        scores.append(score)
        print(f"  Output: {result[:120]}...")
        print_result("Content Check", score, f"| {latency:.2f}s")

    avg = sum(scores) / len(scores)
    print(f"\n  ✅ Average Score: {avg:.2%}")
    return {"feature": "Email Writer", "avg_score": avg, "scores": scores}


def evaluate_chat() -> Dict:
    print_header("💬 CHAT ASSISTANT EVALUATION")
    session = ChatSession()
    scores = []
    judge_scores = []

    for i, test in enumerate(CHAT_TESTS, 1):
        print(f"\n  Test {i}: {test['input']}")

        t0 = time.time()
        response = session.chat(test["input"])
        latency = time.time() - t0

        # Word check score
        word_score = check_words_present(response, test["check"])
        scores.append(word_score)

        # LLM Judge score
        print(f"  Response: {response[:120]}...")
        print_result("Word Check Score", word_score, f"| {latency:.2f}s")

        print(f"  🤖 Running LLM Judge...")
        judge = llm_judge(test["input"], response)
        overall = judge.get("overall", 0) / 10
        judge_scores.append(overall)
        print(f"  LLM Judge → Relevance: {judge.get('relevance')}/10 | "
              f"Accuracy: {judge.get('accuracy')}/10 | "
              f"Clarity: {judge.get('clarity')}/10 | "
              f"Overall: {judge.get('overall')}/10")
        print(f"  Comment: {judge.get('comment', '')}")

    avg_word = sum(scores) / len(scores)
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0

    print(f"\n  ✅ Average Word Check: {avg_word:.2%}")
    print(f"  ✅ Average LLM Judge:  {avg_judge:.2%}")
    return {
        "feature": "Chat",
        "avg_word_score": avg_word,
        "avg_judge_score": avg_judge,
        "scores": scores
    }

def evaluate_pdf_qa() -> Dict:
    """PDF Q&A evaluation using a sample generated PDF."""
    print_header("📄 PDF Q&A EVALUATION")

    # Create a sample text file to simulate PDF content
    sample_text = """
    LangGraph is a library for building stateful, multi-actor applications with LLMs.
    It extends LangChain by adding support for cyclic graphs, which are essential for
    creating agent runtimes. LangGraph allows developers to define flows that involve
    cycles, enabling more complex agent behaviors.

    Key features of LangGraph include:
    1. Stateful graphs with persistent memory
    2. Support for multi-agent architectures
    3. Built-in streaming support
    4. Human-in-the-loop capabilities
    5. Time travel and debugging features

    LangGraph is particularly useful for building chatbots, agents, and other
    applications that require maintaining state across multiple interactions.
    """

    qa_tests = [
        {"question": "What is LangGraph?", "expected_words": ["library", "stateful", "LLM", "graph"]},
        {"question": "What are the key features of LangGraph?", "expected_words": ["stateful", "streaming", "memory", "agent"]},
        {"question": "What is LangGraph useful for?", "expected_words": ["chatbot", "agent", "state"]},
    ]

    scores = []

    try:
        from src.features.pdf_qa import build_pdf_qa_graph

        for i, test in enumerate(qa_tests, 1):
            print(f"\n  Test {i}: {test['question']}")

            t0 = time.time()
            graph = build_pdf_qa_graph()
            result = graph.invoke({
                "pdf_text": sample_text,
                "question": test["question"],
                "context": "",
                "answer": ""
            })
            latency = time.time() - t0

            answer = result.get("answer", "")
            score = check_words_present(answer, test["expected_words"])
            scores.append(score)

            print(f"  Answer: {answer[:120]}...")
            print_result("Relevance Score", score, f"| {latency:.2f}s")

    except Exception as e:
        print(f"  ⚠️ PDF Q&A evaluation skipped: {e}")
        return {"feature": "PDF Q&A", "avg_score": 0, "scores": []}

    avg = sum(scores) / len(scores)
    print(f"\n  ✅ Average Score: {avg:.2%}")
    return {"feature": "PDF Q&A", "avg_score": avg, "scores": scores}


# ── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_all_evaluations():
    print("\n" + "🔬 " * 20)
    print("   GENAI TOOLKIT — FULL EVALUATION REPORT")
    print("🔬 " * 20)

    results = []
    start_time = time.time()

    # Run all evaluations
    results.append(evaluate_translation())
    results.append(evaluate_summarization())
    results.append(evaluate_keywords())
    results.append(evaluate_email_writer())
    results.append(evaluate_chat())
    results.append(evaluate_pdf_qa())

    total_time = time.time() - start_time

    # Final summary
    print_header("📊 FINAL SUMMARY")
    print()

    overall_scores = []
    for r in results:
        score = r.get("avg_score") or r.get("avg_judge_score", 0)
        overall_scores.append(score)
        print_result(r["feature"], score)

    overall = sum(overall_scores) / len(overall_scores)
    print(f"\n  {'─'*55}")
    print_result("OVERALL TOOLKIT SCORE", overall)
    print(f"\n  ⏱️  Total evaluation time: {total_time:.1f}s")

    # Save results to JSON
    output = {
        "overall_score": overall,
        "total_time_seconds": total_time,
        "features": results
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  💾 Results saved to artifacts/evaluation_results.json")
    print("\n" + "✅ " * 20 + "\n")

    return output


if __name__ == "__main__":
    run_all_evaluations()




