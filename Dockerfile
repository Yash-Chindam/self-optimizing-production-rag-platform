FROM python:3.13-slim AS runtime

# Which optional extras the image carries. The default image runs fully in-process; a
# deployment that sets store, tracing or event endpoints builds with the matching extras, e.g.
#   docker build --build-arg RAG_EXTRAS=stores,observability,events,workflow .
ARG RAG_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ -n "$RAG_EXTRAS" ]; then python -m pip install --no-cache-dir ".[${RAG_EXTRAS}]"; \
    else python -m pip install --no-cache-dir .; fi
COPY data ./data

USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["python", "-m", "rag_platform", "serve", "--host", "0.0.0.0", "--port", "8000"]
