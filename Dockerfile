# Base image for Python applications
FROM python:3.12-slim

# Set environment variables (strictly bounding threads and memory)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FASTEMBED_THREADS=1 \
    FASTEMBED_CACHE_PATH=/app/model_cache \
    EMBEDDING_BATCH_SIZE=32

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    iptables \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Tailscale binaries from official Tailscale image for GPU cluster connectivity
COPY --from=docker.io/tailscale/tailscale:stable /usr/local/bin/tailscaled /usr/local/bin/tailscaled
COPY --from=docker.io/tailscale/tailscale:stable /usr/local/bin/tailscale /usr/local/bin/tailscale
RUN mkdir -p /var/run/tailscale /var/cache/tailscale /var/lib/tailscale

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FastEmbed quantized model into container layer to avoid runtime memory spikes in tmpfs
RUN mkdir -p /app/model_cache && \
    python -c "from fastembed import TextEmbedding; m = TextEmbedding(model_name='nomic-ai/nomic-embed-text-v1.5-Q', cache_dir='/app/model_cache', threads=1); list(m.embed(['warmup']))"

# Pre-copy and compile C++ Forecaster Engine for build-layer caching
COPY agents/core_financial/Forecaster_Agent/mathematics/algorithm /app/agents/core_financial/Forecaster_Agent/mathematics/algorithm
RUN g++ -O3 -shared -fPIC -std=c++17 \
    -I/app/agents/core_financial/Forecaster_Agent/mathematics/algorithm \
    -o /app/agents/core_financial/Forecaster_Agent/mathematics/hybrid_forecaster.so \
    /app/agents/core_financial/Forecaster_Agent/mathematics/algorithm/algorithm.cpp \
    /app/agents/core_financial/Forecaster_Agent/mathematics/algorithm/hybrid_algorithm.cpp

# Copy remaining application code
COPY . .

# Ensure start script is executable
RUN chmod +x scripts/start_cloudrun.sh

# Expose API port
EXPOSE 8080

# Run the application
CMD ["/app/scripts/start_cloudrun.sh"]
