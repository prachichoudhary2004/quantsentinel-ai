# Base Image
FROM python:3.9-slim

# System updates and dependencies (required for PySpark / Java fallback)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy Requirements
COPY requirements.txt .

# Install Packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Code
COPY . .

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501
ENV API_PORT=8000

# Expose Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

# Exec entrypoint
CMD ["sh", "-c", "python src/download_data.py && python src/bronze_to_silver.py && python src/silver_to_gold.py && python src/ml_engine.py && uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0"]
