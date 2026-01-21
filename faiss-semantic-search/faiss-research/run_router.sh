#!/bin/bash
# Run router on port 5003

cd "$(dirname "$0")"

export SHARDS="http://127.0.0.1:5001,http://127.0.0.1:5002"
export ROUTER_PORT=5003

echo "Starting Router on port 5003..."
echo "Shards: $SHARDS"
python router.py
