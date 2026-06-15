---
title: GenAI Toolkit
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🤖 GenAI Toolkit

> A powerful, all-in-one Generative AI application combining multiple NLP features into a single, clean interface — powered by LangChain, LangGraph, and Groq LLaMA 3.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🌐 **Language Translation** | Translate text across 20+ languages instantly |
| 📝 **Text Summarization** | Condense long documents using map-reduce chunking |
| 🔍 **Keyword Extraction** | Extract key topics and phrases from any text |
| ✉️ **Email Writer** | Generate professional emails and cover letters |
| 💬 **Chat Assistant** | Multi-turn AI conversations with memory |
| 📄 **PDF Q&A** | Upload PDFs and ask questions using RAG |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq LLaMA 3.3 70B + LLaMA 3.1 8B |
| **Orchestration** | LangChain + LangGraph |
| **Vector Store** | FAISS + HuggingFace Embeddings |
| **UI** | Gradio |
| **Deployment** | Docker + Hugging Face Spaces |

---

## 🏗️ Architecture
GenAI-Toolkit/
├── src/
│   ├── features/
│   │   ├── translation.py        # LangChain PromptTemplate
│   │   ├── summarization.py      # Map-Reduce summarization
│   │   ├── keyword_extraction.py # Structured output extraction
│   │   ├── email_writer.py       # Email + Cover letter generation
│   │   └── pdf_qa.py             # RAG pipeline with FAISS
│   ├── graphs/
│   │   └── chat_graph.py         # LangGraph conversation graph
│   ├── utils/
│   │   ├── llm.py                # Groq LLM setup
│   │   └── vector_store.py       # FAISS vector store
│   └── app.py                    # Gradio UI
├── Dockerfile
└── requirements.txt

---

## 🚀 How It Works

### Language Translation
Uses LangChain `PromptTemplate` with Groq LLaMA 3.1 to translate text between 20+ languages while preserving tone and context.

### Text Summarization
Implements **map-reduce chunking** — splits long documents into chunks, summarizes each chunk, then combines into a final summary. Handles documents of any length.

### Keyword Extraction
Uses structured LLM output to extract the most relevant keywords with context and relevance explanations.

### Email & Cover Letter Writer
Template-driven generation with customizable tone, email type, and recipient details. Cover letter writer tailors content to specific job roles and companies.

### Chat Assistant
Built with **LangGraph StateGraph** — maintains full conversation history across turns using a stateful graph architecture.

### PDF Q&A
Implements **RAG (Retrieval Augmented Generation)**:
1. Upload PDF → extract text with PyMuPDF
2. Chunk text with RecursiveCharacterTextSplitter
3. Embed chunks with HuggingFace sentence-transformers
4. Store in FAISS vector store
5. Retrieve top-k relevant chunks per question
6. Generate answer with LangGraph agent

---

## 🔧 Local Setup

```bash
# Clone the repo
git clone https://github.com/shashankheg/GenAI-Toolkit.git
cd GenAI-Toolkit

# Create virtual environment
uv venv
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Add API key to .env
echo "GROQ_API_KEY=your-key-here" > .env

# Run the app
python -m src.app
```

---

## 🐳 Docker

```bash
docker build -t genai-toolkit .
docker run -p 7860:7860 -e GROQ_API_KEY=your-key genai-toolkit
```

---

## 🔑 API Keys

- **Groq API** (free) → https://console.groq.com
- No OpenAI key required — fully powered by Groq

---

## 📊 Model Details

| Task | Model | Speed |
|------|-------|-------|
| Translation, Keywords, Email | LLaMA 3.1 8B Instant | ~0.5s |
| Summarization, Chat, PDF Q&A | LLaMA 3.3 70B Versatile | ~1-2s |

---

## 👨‍💻 Author


- HuggingFace: [@shashankheg](https://huggingface.co/shashankheg)