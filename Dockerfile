# Base image for Python applications
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive
ENV OMP_NUM_THREADS 1
ENV MKL_NUM_THREADS 1
ENV OPENBLAS_NUM_THREADS 1
ENV FASTEMBED_THREADS 1
ENV FASTEMBED_CACHE_PATH /app/model_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    iptables \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Tailscale binaries from official Tailscale image
COPY --from=docker.io/tailscale/tailscale:stable /usr/local/bin/tailscaled /usr/local/bin/tailscaled
COPY --from=docker.io/tailscale/tailscale:stable /usr/local/bin/tailscale /usr/local/bin/tailscale
RUN mkdir -p /var/run/tailscale /var/cache/tailscale /var/lib/tailscale

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FastEmbed model into container layer to avoid runtime memory consumption in tmpfs
RUN mkdir -p /app/model_cache && \
    python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='nomic-ai/nomic-embed-text-v1.5', cache_dir='/app/model_cache', threads=1)"

# Copy application code
COPY . .

# Ensure start script is executable
RUN chmod +x scripts/start_cloudrun.sh

# Expose API port
EXPOSE 8080

# Run the application
CMD ["/app/scripts/start_cloudrun.sh"]
