FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY retrieval_app/ ./retrieval_app/
COPY scripts/ ./scripts/

# Create user matching host user to avoid permission issues
RUN useradd -m -u 1002 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "retrieval_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]