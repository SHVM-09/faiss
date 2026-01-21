#!/bin/bash
# Run shard 0 on port 5001

cd "$(dirname "$0")"

export SHARD_ID=0
export SHARD_COUNT=2
export PORT=5001

echo "Starting Shard 0 on port 5001..."
python app_v2.py
