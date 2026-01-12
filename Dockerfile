FROM python:3.11-slim

WORKDIR /app

# 1) deps
COPY requirements.api.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.api.txt

# 2) code + assets
COPY src ./src
COPY models ./models
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
