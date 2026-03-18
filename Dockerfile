FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md rfc9420.txt /app/
COPY src /app/src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["sh", "-c", "rfc-rag init-db && rfc-rag serve-mcp --host 0.0.0.0 --port 8000"]
