FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for some python packages like psycopg2 or llama-parse if any)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port Uvicorn runs on (7860 is required for Hugging Face Spaces)
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
