
The project structure is below.


GenAI-Toolkit/
├── src/
│   ├── features/
│   │   ├── translation.py        # LangChain PromptTemplate
│   │   ├── summarization.py      # LangChain MapReduceChain
│   │   ├── chat.py               # LangGraph conversation graph
│   │   ├── pdf_qa.py             # LangChain RAG + LangGraph agent
│   │   ├── email_writer.py       # LangChain PromptTemplate
│   │   └── keyword_extraction.py # LangChain structured output
│   ├── graphs/
│   │   ├── chat_graph.py         # LangGraph chat workflow
│   │   └── pdf_agent.py          # LangGraph PDF Q&A agent
│   ├── utils/
│   │   ├── llm.py                # OpenAI client setup
│   │   └── vector_store.py       # FAISS setup
│   └── app.py                    # Gradio UI
├── data/
├── artifacts/
├── scripts/
│   └── download_datasets.py
├── .env
├── requirements.txt
├── Dockerfile
└── README.md