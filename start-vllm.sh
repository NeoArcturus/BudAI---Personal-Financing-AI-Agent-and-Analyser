#!/bin/bash

lsof -ti:8000,8001 | xargs kill -9 2>/dev/null

echo "Starting vLLM-MLX High-Reasoning Brain..."
echo "Target Model: mlx-community/Qwen2.5-7B-Instruct-4bit"

vllm-mlx serve mlx-community/Qwen2.5-7B-Instruct-4bit \
    --host 0.0.0.0 \
    --port 8000 \
    --continuous-batching \
    --max-num-seqs 4 \
    --prefill-batch-size 1 \
    --use-paged-cache \
    --chunked-prefill-tokens 512 \
    --gpu-memory-utilization 0.4 \
    --trust-remote-code
