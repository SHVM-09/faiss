#!/bin/bash
# Run shard 1 on port 5002

cd "$(dirname "$0")"

export SHARD_ID=1
export SHARD_COUNT=2
export PORT=5002

echo "Starting Shard 1 on port 5002..."
python app_v2.py
