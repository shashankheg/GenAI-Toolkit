FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV GROQ_API_KEY=""

EXPOSE 7860

CMD ["sh", "-c", "python -c 'import gradio; print(\"Gradio version:\", gradio.__version__)' && python -m src.app"]
