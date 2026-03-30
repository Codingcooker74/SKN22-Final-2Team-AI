FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_MODEL=intfloat/multilingual-e5-large \
    FASTEMBED_CACHE_PATH=/opt/fastembed-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/prewarm_fastembed.py ./scripts/prewarm_fastembed.py

ARG PREWARM_FASTEMBED=1
RUN if [ "$PREWARM_FASTEMBED" = "1" ]; then python scripts/prewarm_fastembed.py; fi

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
