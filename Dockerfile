# ---- build stage: install deps into a venv --------------------------------
FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir --upgrade pip \
 && /venv/bin/pip install --no-cache-dir .

# ---- runtime stage: only what's needed to serve ---------------------------
FROM python:3.12-slim
WORKDIR /app
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    MODEL_PATH=/app/models/tfidf_logreg.joblib \
    LOG_PATH=/app/logs/predictions.jsonl \
    PORT=8000
COPY --from=build /venv /venv
COPY --from=build /app/src /app/src
COPY data/corpus ./data/corpus
COPY models/tfidf_logreg.joblib ./models/tfidf_logreg.joblib
RUN useradd -m appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn complaint_triage.api:app --host 0.0.0.0 --port ${PORT}"]
