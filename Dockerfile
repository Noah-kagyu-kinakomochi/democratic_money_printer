FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for compiling some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories that the app expects to exist
RUN mkdir -p data db data_store/parquet logs

# Create a non-root user to run the application (Security Best Practice)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Set entrypoint (default to running the trading cycle, but can be overridden)
ENTRYPOINT ["python", "main.py"]
CMD ["run"]
