
#  Main App using Gradio 



#import all the libraries and functions 
import gradio as gr
from src.features.translation import translate_text
from src.features.summarization import summarize_text
from src.features.keyword_extraction import extract_keywords
from src.features.email_writer import write_email, write_cover_letter
from src.features.pdf_qa import ask_pdf, load_pdf
from src.graphs.chat_graph import ChatSession



# === CHAT SESSION (persistent across turns) ===
chat_session = ChatSession()


# === TRANSLATION TAB ===
def translation_tab():
    with gr.Tab("🌐 Translation"):
        gr.Markdown("## Language Translation\nTranslate text between 100+ languages.")
        with gr.Row():
            with gr.Column():
                translate_input = gr.Textbox(
                    label="Text to Translate",
                    placeholder="Enter text here...",
                    lines=5
                )
                with gr.Row():
                    source_lang = gr.Dropdown(
                        label="Source Language",
                        choices=["English", "French", "Spanish", "German",
                                 "Italian", "Portuguese", "Chinese", "Japanese",
                                 "Arabic", "Hindi", "Korean", "Russian"],
                        value="English"
                    )
                    target_lang = gr.Dropdown(
                        label="Target Language",
                        choices=["English", "French", "Spanish", "German",
                                 "Italian", "Portuguese", "Chinese", "Japanese",
                                 "Arabic", "Hindi", "Korean", "Russian"],
                        value="French"
                    )
                translate_btn = gr.Button("Translate 🌐", variant="primary")
            with gr.Column():
                translate_output = gr.Textbox(
                    label="Translation",
                    lines=5
                )

        translate_btn.click(
            fn=translate_text,
            inputs=[translate_input, source_lang, target_lang],
            outputs=translate_output
        )



#Summerization 
def summarization_tab():
    with gr.Tab("📝 Summarization"):
        gr.Markdown("## Text Summarization\nSummarize long documents and articles.")
        with gr.Row():
            with gr.Column():
                summary_input = gr.Textbox(
                    label="Text to Summarize",
                    placeholder="Paste your text here...",
                    lines=10
                )
                summary_mode = gr.Radio(
                    label="Summary Mode",
                    choices=["concise", "detailed"],
                    value="concise"
                )
                summary_btn = gr.Button("Summarize 📝", variant="primary")
            with gr.Column():
                summary_output = gr.Textbox(
                    label="Summary",
                    lines=10
                )

        summary_btn.click(
            fn=summarize_text,
            inputs=[summary_input, summary_mode],
            outputs=summary_output
        )

# === KEYWORD EXTRACTION TAB ===
def keyword_tab():
    with gr.Tab("🔍 Keywords"):
        gr.Markdown("## Keyword Extraction\nExtract key topics from any text.")
        with gr.Row():
            with gr.Column():
                keyword_input = gr.Textbox(
                    label="Text",
                    placeholder="Paste your text here...",
                    lines=8
                )
                num_keywords = gr.Slider(
                    label="Number of Keywords",
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1
                )
                keyword_btn = gr.Button("Extract Keywords 🔍", variant="primary")
            with gr.Column():
                keyword_output = gr.Textbox(
                    label="Keywords",
                    lines=8
                )

        keyword_btn.click(
            fn=extract_keywords,
            inputs=[keyword_input, num_keywords],
            outputs=keyword_output
        )



# === EMAIL WRITER TAB ===
def email_tab():
    with gr.Tab("✉️ Email Writer"):
        gr.Markdown("## Email & Cover Letter Writer")
        with gr.Tabs():
            with gr.Tab("Email"):
                with gr.Row():
                    with gr.Column():
                        email_type = gr.Dropdown(
                            label="Email Type",
                            choices=["follow-up", "introduction", "apology",
                                     "request", "thank you", "complaint",
                                     "proposal", "invitation"],
                            value="follow-up"
                        )
                        email_context = gr.Textbox(
                            label="Context / Purpose",
                            placeholder="Describe the purpose of the email...",
                            lines=4
                        )
                        email_tone = gr.Dropdown(
                            label="Tone",
                            choices=["professional", "friendly", "formal",
                                     "casual", "urgent"],
                            value="professional"
                        )
                        with gr.Row():
                            email_recipient = gr.Textbox(
                                label="Recipient Name",
                                placeholder="Hiring Manager"
                            )
                            email_sender = gr.Textbox(
                                label="Your Name",
                                placeholder="Shashank"
                            )
                        email_btn = gr.Button("Write Email ✉️", variant="primary")
                    with gr.Column():
                        email_output = gr.Textbox(label="Generated Email", lines=15)

                email_btn.click(
                    fn=write_email,
                    inputs=[email_type, email_context, email_tone,
                            email_recipient, email_sender],
                    outputs=email_output
                )

            with gr.Tab("Cover Letter"):
                with gr.Row():
                    with gr.Column():
                        cl_name     = gr.Textbox(label="Your Name", placeholder="Shashank Hegde")
                        cl_job      = gr.Textbox(label="Job Title", placeholder="ML Engineer")
                        cl_company  = gr.Textbox(label="Company", placeholder="Google")
                        cl_skills   = gr.Textbox(label="Key Skills", placeholder="Python, ML, LangChain...", lines=3)
                        cl_exp      = gr.Textbox(label="Experience Summary", placeholder="3 years in ML...", lines=3)
                        cl_btn      = gr.Button("Write Cover Letter 📄", variant="primary")
                    with gr.Column():
                        cl_output = gr.Textbox(label="Cover Letter", lines=15)

                cl_btn.click(
                    fn=write_cover_letter,
                    inputs=[cl_job, cl_company, cl_skills, cl_exp, cl_name],
                    outputs=cl_output
                )


# === CHAT TAB ===
def chat_tab():
    with gr.Tab("💬 Chat Assistant"):
        gr.Markdown("## AI Chat Assistant\nAsk me anything!")
        chatbot = gr.Chatbot(height=400)
        with gr.Row():
            chat_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                scale=4
            )
            chat_btn = gr.Button("Send 💬", variant="primary", scale=1)
        clear_btn = gr.Button("Clear Conversation 🗑️")
        
        def respond(message, history):
            if not message.strip():
                return "", history
            response = chat_session.chat(message)
            history.append((message, response))
            return "", history
        
        def clear():
            chat_session.clear_history()
            return []

        chat_btn.click(
            fn=respond,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )

        clear_btn.click(
            fn=clear,
            inputs=[],
            outputs=[chatbot]
        )

        chat_input.submit(
            fn=respond,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )

        clear_btn.click(fn=clear, outputs=chatbot)


# === PDF Q&A TAB ===
def pdf_tab():
    pdf_text_state = gr.State("")

    with gr.Tab("📄 PDF Q&A"):
        gr.Markdown("## PDF Q&A\nUpload a PDF and ask questions about it.")
        with gr.Row():
            with gr.Column():
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"]
                )
                pdf_status = gr.Textbox(label="Status", interactive=False)
                pdf_question = gr.Textbox(
                    label="Ask a Question",
                    placeholder="What is this document about?",
                    lines=3
                )
                pdf_btn = gr.Button("Ask 🔍", variant="primary")
            with gr.Column():
                pdf_output = gr.Textbox(label="Answer", lines=15)

        def load_pdf_file(file):
            if file is None:
                return "", "⚠️ No file uploaded."
            text = load_pdf(file.name)
            return text, f"✅ PDF loaded: {len(text.split())} words"

        def answer_question(question, pdf_text):
            if not pdf_text:
                return "⚠️ Please upload a PDF first."
            if not question.strip():
                return "⚠️ Please enter a question."
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                             delete=False) as f:
                f.write(pdf_text)
                tmp_path = f.name

            from src.features.pdf_qa import build_pdf_qa_graph
            graph = build_pdf_qa_graph()
            result = graph.invoke({
                "pdf_text": pdf_text,
                "question": question,
                "context": "",
                "answer": ""
            })
            os.unlink(tmp_path)
            return result["answer"]

        pdf_upload.change(
            fn=load_pdf_file,
            inputs=pdf_upload,
            outputs=[pdf_text_state, pdf_status]
        )
        pdf_btn.click(
            fn=answer_question,
            inputs=[pdf_question, pdf_text_state],
            outputs=pdf_output
        )

# === MAIN APP ===
def create_app():
    with gr.Blocks(
        title="🤖 GenAI Toolkit",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown("""
        # 🤖 GenAI Toolkit
        ### Powered by LangChain + LangGraph + Groq
        """)

        translation_tab()
        summarization_tab()
        keyword_tab()
        email_tab()
        chat_tab()
        pdf_tab()

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
