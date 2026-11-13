#!/bin/bash
set -e

echo "Starting BudAI backend..."
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python main.py
fi
