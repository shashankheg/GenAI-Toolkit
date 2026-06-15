import gradio as gr
from src.features.translation import translate_text
from src.features.summarization import summarize_text
from src.features.keyword_extraction import extract_keywords
from src.features.email_writer import write_email, write_cover_letter
from src.features.pdf_qa import load_pdf, build_pdf_qa_graph
from src.graphs.chat_graph import ChatSession

chat_session = ChatSession()

LANGUAGES = [
    "English", "French", "Spanish", "German", "Italian",
    "Portuguese", "Chinese", "Japanese", "Arabic", "Hindi",
    "Korean", "Russian", "Dutch", "Swedish", "Turkish",
    "Polish", "Ukrainian", "Vietnamese", "Thai", "Indonesian"
]

custom_css = """
:root {
    --primary: #0D9488;
    --primary-light: #5EEAD4;
    --surface: #ffffff;
    --surface-2: #f0fdfa;
    --border: #ccfbf1;
    --text: #134e4a;
    --text-muted: #6b7280;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow: 0 4px 24px rgba(13,148,136,0.10);
}

body, .gradio-container {
    background: linear-gradient(135deg, #f0fdfa 0%, #e6fffa 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.gradio-container { max-width: 1200px !important; margin: 0 auto !important; padding: 2rem !important; }

.app-header {
    background: linear-gradient(135deg, #0D9488, #14B8A6);
    border-radius: var(--radius);
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: var(--shadow);
}

.tab-nav { background: white !important; border-radius: var(--radius) !important; padding: 6px !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow) !important; gap: 4px !important; }
.tab-nav button { border-radius: var(--radius-sm) !important; border: none !important; color: var(--text-muted) !important; font-weight: 500 !important; padding: 10px 20px !important; transition: all 0.2s !important; }
.tab-nav button.selected { background: linear-gradient(135deg, #0D9488, #14B8A6) !important; color: white !important; box-shadow: 0 4px 12px rgba(13,148,136,0.3) !important; }

.feature-card { background: white !important; border-radius: var(--radius) !important; border: 1px solid var(--border) !important; padding: 1.5rem !important; box-shadow: var(--shadow) !important; }

textarea, input[type=text], input[type=number] { border-radius: var(--radius-sm) !important; border: 1.5px solid var(--border) !important; background: var(--surface-2) !important; font-size: 14px !important; transition: border-color 0.2s !important; }
textarea:focus, input:focus { border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(13,148,136,0.15) !important; }

button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #0D9488, #14B8A6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 4px 12px rgba(13,148,136,0.3) !important;
}

.secondary-btn { background: white !important; border: 1.5px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--primary) !important; font-weight: 500 !important; }
.wrap { border-radius: var(--radius-sm) !important; border: 1.5px solid var(--border) !important; background: var(--surface-2) !important; }
label span { font-weight: 600 !important; color: var(--text) !important; font-size: 13px !important; }
.chatbot { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow) !important; }
.section-title { font-size: 22px !important; font-weight: 700 !important; color: var(--text) !important; margin-bottom: 4px !important; }
.section-subtitle { font-size: 14px !important; color: var(--text-muted) !important; margin-bottom: 1.5rem !important; }
.badge { display: inline-block; background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); border-radius: 20px; padding: 4px 14px; font-size: 12px; color: white; font-weight: 600; margin: 4px; }
.upload-btn { border-radius: var(--radius-sm) !important; border: 2px dashed var(--primary-light) !important; background: var(--surface-2) !important; }
input[type=range] { accent-color: var(--primary) !important; }
.output-box { background: var(--surface-2) !important; border: 1.5px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
"""


def create_app():
    with gr.Blocks(title="GenAI Toolkit",css=custom_css) as app:


        gr.HTML("""
        <div class="app-header">
            <div style="font-size:40px;margin-bottom:10px">🤖</div>
            <h1 style="font-size:32px;font-weight:800;margin:0 0 8px">GenAI Toolkit</h1>
            <p style="font-size:16px;opacity:0.9;margin:0">Powered by LangChain · LangGraph · Groq LLaMA 3</p>
            <div style="margin-top:16px">
                <span class="badge">🌐 Translation</span>
                <span class="badge">📝 Summarization</span>
                <span class="badge">💬 Chat</span>
                <span class="badge">📄 PDF Q&A</span>
                <span class="badge">✉️ Email Writer</span>
                <span class="badge">🔍 Keywords</span>
            </div>
        </div>
        """)

        with gr.Tabs(elem_classes="tab-nav"):

            # TRANSLATION
            with gr.Tab("🌐 Translation"):
                gr.HTML('<p class="section-title">Language Translation</p><p class="section-subtitle">Translate text instantly across 20+ languages using LLaMA 3</p>')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        t_input = gr.Textbox(label="Source Text", placeholder="Type or paste text to translate...", lines=8)
                        with gr.Row():
                            t_src = gr.Dropdown(label="From", choices=LANGUAGES, value="English")
                            t_tgt = gr.Dropdown(label="To", choices=LANGUAGES, value="French")
                        t_btn = gr.Button("Translate ➜", variant="primary")
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        t_out = gr.Textbox(label="Translation", lines=8, interactive=False, elem_classes="output-box")
                        gr.HTML('<p style="font-size:12px;color:#6b7280;margin-top:8px">⚡ Powered by LLaMA 3.1 · Fast & accurate</p>')
                t_btn.click(fn=translate_text, inputs=[t_input, t_src, t_tgt], outputs=t_out)

            # SUMMARIZATION
            with gr.Tab("📝 Summarize"):
                gr.HTML('<p class="section-title">Text Summarization</p><p class="section-subtitle">Condense long articles, reports, and documents into clear summaries</p>')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        s_input = gr.Textbox(label="Text to Summarize", placeholder="Paste your article, report, or any long text here...", lines=12)
                        s_mode = gr.Radio(label="Summary Style", choices=["concise", "detailed"], value="concise")
                        s_btn = gr.Button("Summarize ✦", variant="primary")
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        s_out = gr.Textbox(label="Summary", lines=12, interactive=False, elem_classes="output-box")
                        gr.HTML('<p style="font-size:12px;color:#6b7280;margin-top:8px">📊 Supports long documents via map-reduce chunking</p>')
                s_btn.click(fn=summarize_text, inputs=[s_input, s_mode], outputs=s_out)

            # KEYWORDS
            with gr.Tab("🔍 Keywords"):
                gr.HTML('<p class="section-title">Keyword Extraction</p><p class="section-subtitle">Identify the most important topics and phrases in any text</p>')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        k_input = gr.Textbox(label="Input Text", placeholder="Paste any text to extract key topics...", lines=10)
                        k_num = gr.Slider(label="Number of Keywords", minimum=5, maximum=20, value=10, step=1)
                        k_btn = gr.Button("Extract Keywords 🔍", variant="primary")
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        k_out = gr.Textbox(label="Extracted Keywords", lines=10, interactive=False, elem_classes="output-box")
                        gr.HTML('<p style="font-size:12px;color:#6b7280;margin-top:8px">🏷️ Returns keywords with context and relevance</p>')
                k_btn.click(fn=extract_keywords, inputs=[k_input, k_num], outputs=k_out)

            # EMAIL WRITER
            with gr.Tab("✉️ Email Writer"):
                gr.HTML('<p class="section-title">Email & Cover Letter Writer</p><p class="section-subtitle">Generate professional emails and cover letters instantly</p>')
                with gr.Tabs():
                    with gr.Tab("📧 Email"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, elem_classes="feature-card"):
                                e_type = gr.Dropdown(label="Email Type", choices=["follow-up","introduction","apology","request","thank you","complaint","proposal","invitation"], value="follow-up")
                                e_ctx  = gr.Textbox(label="Purpose / Context", placeholder="Describe the goal of this email...", lines=4)
                                e_tone = gr.Dropdown(label="Tone", choices=["professional","friendly","formal","casual","urgent"], value="professional")
                                with gr.Row():
                                    e_to   = gr.Textbox(label="Recipient", placeholder="Hiring Manager")
                                    e_from = gr.Textbox(label="Your Name", placeholder="Shashank")
                                e_btn = gr.Button("Generate Email ✉️", variant="primary")
                            with gr.Column(scale=1, elem_classes="feature-card"):
                                e_out = gr.Textbox(label="Generated Email", lines=18, interactive=False, elem_classes="output-box")
                        e_btn.click(fn=write_email, inputs=[e_type, e_ctx, e_tone, e_to, e_from], outputs=e_out)

                    with gr.Tab("📄 Cover Letter"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, elem_classes="feature-card"):
                                cl_name = gr.Textbox(label="Your Name", placeholder="Shashank Hegde")
                                cl_job  = gr.Textbox(label="Job Title Applying For", placeholder="ML Engineer")
                                cl_co   = gr.Textbox(label="Company Name", placeholder="Google DeepMind")
                                cl_sk   = gr.Textbox(label="Key Skills", placeholder="Python, LangChain, MLflow, XGBoost...", lines=3)
                                cl_exp  = gr.Textbox(label="Experience Summary", placeholder="3 years building ML pipelines...", lines=3)
                                cl_btn  = gr.Button("Generate Cover Letter 📄", variant="primary")
                            with gr.Column(scale=1, elem_classes="feature-card"):
                                cl_out = gr.Textbox(label="Cover Letter", lines=18, interactive=False, elem_classes="output-box")
                        cl_btn.click(fn=write_cover_letter, inputs=[cl_job, cl_co, cl_sk, cl_exp, cl_name], outputs=cl_out)

            # CHAT
            with gr.Tab("💬 Chat"):
                gr.HTML('<p class="section-title">AI Chat Assistant</p><p class="section-subtitle">Multi-turn conversations powered by LangGraph + LLaMA 3.3 70B</p>')
                with gr.Column(elem_classes="feature-card"):
                    chatbot = gr.Chatbot(height=450, label="", elem_classes="chatbot", show_label=False,type="messages")    
                    with gr.Row():
                        c_input = gr.Textbox(label="", placeholder="Ask me anything...", scale=5, container=False)
                        c_btn   = gr.Button("Send ➜", variant="primary", scale=1)
                    with gr.Row():
                        c_clear = gr.Button("🗑️ Clear Conversation", elem_classes="secondary-btn")
                        gr.HTML('<p style="font-size:12px;color:#6b7280;padding-top:10px">🧠 Maintains full conversation context via LangGraph state</p>')

                def respond(message, history):
                    if not message.strip():
                        return "", history
                    response = chat_session.chat(message)
                    history = history or []
                    # history.append((message, response))
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response})
                    return "", history

                def clear():
                    chat_session.clear_history()
                    return []

                c_btn.click(fn=respond, inputs=[c_input, chatbot], outputs=[c_input, chatbot])
                c_input.submit(fn=respond, inputs=[c_input, chatbot], outputs=[c_input, chatbot])
                c_clear.click(fn=clear, outputs=chatbot)

            # PDF Q&A
            with gr.Tab("📄 PDF Q&A"):
                gr.HTML('<p class="section-title">PDF Question & Answer</p><p class="section-subtitle">Upload any PDF and ask questions — powered by RAG + LangGraph</p>')
                pdf_state = gr.State("")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        p_file   = gr.File(label="Upload PDF", file_types=[".pdf"], elem_classes="upload-btn")
                        p_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        p_q      = gr.Textbox(label="Your Question", placeholder="What is the main topic of this document?", lines=4)
                        p_btn    = gr.Button("Ask Question 🔍", variant="primary")
                        gr.HTML('<p style="font-size:12px;color:#6b7280;margin-top:8px">📚 Uses FAISS vector search + semantic chunking</p>')
                    with gr.Column(scale=1, elem_classes="feature-card"):
                        p_out = gr.Textbox(label="Answer", lines=18, interactive=False, elem_classes="output-box")

                def load_pdf_file(file):
                    if file is None:
                        return "", "⚠️ No file uploaded."
                    text = load_pdf(file.name)
                    word_count = len(text.split())
                    return text, f"✅ PDF loaded — {word_count:,} words extracted"

                def answer_pdf_question(question, pdf_text):
                    if not pdf_text:
                        return "⚠️ Please upload a PDF first."
                    if not question.strip():
                        return "⚠️ Please enter a question."
                    graph = build_pdf_qa_graph()
                    result = graph.invoke({
                        "pdf_text": pdf_text,
                        "question": question,
                        "context": "",
                        "answer": ""
                    })
                    return result["answer"]

                p_file.change(fn=load_pdf_file, inputs=p_file, outputs=[pdf_state, p_status])
                p_btn.click(fn=answer_pdf_question, inputs=[p_q, pdf_state], outputs=p_out)

        gr.HTML("""
        <div style="text-align:center;padding:2rem;color:#6b7280;font-size:13px;margin-top:1rem">
            Built with LangChain · LangGraph · Groq · Gradio &nbsp;|&nbsp;
            <a href="https://github.com/shashankheg" style="color:#0D9488;text-decoration:none">GitHub</a>
        </div>
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        
        

    )
