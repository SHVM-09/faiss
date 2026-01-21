.PHONY: shard0 shard1 router all stop help

# Run shard 0
shard0:
	@echo "Starting Shard 0 on port 5001..."
	@SHARD_ID=0 SHARD_COUNT=2 PORT=5001 python app_v2.py

# Run shard 1
shard1:
	@echo "Starting Shard 1 on port 5002..."
	@SHARD_ID=1 SHARD_COUNT=2 PORT=5002 python app_v2.py

# Run router
router:
	@echo "Starting Router on port 5003..."
	@SHARDS="http://127.0.0.1:5001,http://127.0.0.1:5002" ROUTER_PORT=5003 python router.py

# Run all services in background (requires GNU parallel or similar)
all:
	@echo "Starting all services..."
	@echo "Note: Run each in a separate terminal or use a process manager"
	@echo ""
	@echo "Terminal 1: make shard0"
	@echo "Terminal 2: make shard1"
	@echo "Terminal 3: make router"

help:
	@echo "FAISS Sharded Vector DB - Development Commands"
	@echo ""
	@echo "Commands:"
	@echo "  make shard0    - Start shard 0 on port 5001"
	@echo "  make shard1    - Start shard 1 on port 5002"
	@echo "  make router    - Start router on port 5000"
	@echo "  make all       - Show instructions for running all services"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Or use shell scripts:"
	@echo "  ./run_shard0.sh"
	@echo "  ./run_shard1.sh"
	@echo "  ./run_router.sh"
