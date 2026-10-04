FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Build the fake database and download the embedding model now, so the container starts fast and offline.
RUN python -m data.seed \
 && python -c "from app.embeddings import get_embedder; get_embedder().embed_passages(['warm up'])"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
