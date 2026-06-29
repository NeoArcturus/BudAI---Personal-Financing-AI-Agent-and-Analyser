#!/bin/bash

export HF_TOKEN="${HF_TOKEN:-your_token_here}"

lsof -ti:8000,8001 | xargs kill -9 2>/dev/null

echo "Installing Rapid-MLX..."

echo "Starting Rapid-MLX Qwen3.5-4B Model 4Bit Quantized (Reasoning Sub-Agents)"
echo "Target Model: mlx-community/Qwen3.5-4B-4bit on PORT 8000"

rapid-mlx serve mlx-community/Qwen3.5-4B-4bit \
    --host 0.0.0.0 \
    --port 8000 \
    --continuous-batching \
    --max-num-seqs 4 \
    --prefill-batch-size 1 \
    --use-paged-cache \
    --chunked-prefill-tokens 512 \
    --gpu-memory-utilization 0.8 &
