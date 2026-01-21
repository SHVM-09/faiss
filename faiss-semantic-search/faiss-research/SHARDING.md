# Sharding Guide

## Overview

The FAISS vector database supports **horizontal sharding** across multiple shard processes. This allows you to:
- Distribute data across multiple shards for scalability
- Run multiple shard instances on the same machine (for testing) or different machines (for production)
- Use a router service to automatically route requests to the correct shard

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTP Requests
       │
┌──────▼──────────────────┐
│   Router (Port 5003)    │
│  - Routes writes        │
│  - Broadcasts searches  │
│  - Merges results       │
└──────┬────────┬─────────┘
       │        │
       │        │
┌──────▼──┐  ┌──▼──────┐
│ Shard 0 │  │ Shard 1 │
│ Port    │  │ Port    │
│ 5001    │  │ 5002    │
└─────────┘  └─────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Start Shard 0

```bash
# Terminal 1
cd faiss-research
make shard0
# or
./run_shard0.sh
```

### 3. Start Shard 1

```bash
# Terminal 2
cd faiss-research
make shard1
# or
./run_shard1.sh
```

### 4. Start Router

```bash
# Terminal 3
cd faiss-research
make router
# or
./run_router.sh
```

### 5. Use the Router

All client requests should go to the router (port 5003), not directly to shards.

```bash
# Ingest documents (routed to correct shard based on namespace)
curl -X POST http://localhost:5003/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "docs_path": "../docs",
    "namespace": "alpha"
  }'

# Search (broadcast to all shards, results merged)
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 5
  }'
```

## How Sharding Works

### Routing Policy

The router uses **stable hashing** to distribute documents across shards:

**For Ingest:**
- Lists all files in `docs_path`
- Routes each file to a shard based on `hash(filename + namespace) % SHARD_COUNT`
- Files in the same namespace are distributed across shards for parallel processing
- Creates temporary subdirectories per shard with symlinks/copies of assigned files

**For Delete/Restore:**
- Routes based on `doc_id` hash: `hash(doc_id) % SHARD_COUNT`
- Same document always routes to the same shard
- If `chunk_id` provided, extracts `doc_id` from it

**For Search:**
- Broadcasts to all shards in parallel
- Merges results by score

- Uses `xxhash` (fast) or `SHA1` (fallback) for deterministic hashing
- Even distribution across shards

### Data Storage

Each shard stores data in its own directory:

```
data/
├── shards/
│   ├── shard0/
│   │   ├── <namespace>/
│   │   │   ├── text_index.faiss
│   │   │   ├── image_index.faiss
│   │   │   └── manifest.json
│   │   └── metadata/
│   │       └── metadata.db
│   └── shard1/
│       ├── <namespace>/
│       │   ├── text_index.faiss
│       │   ├── image_index.faiss
│       │   └── manifest.json
│       └── metadata/
│           └── metadata.db
```

**Key Points:**
- Each shard has its own SQLite database
- Namespace directories are isolated per shard
- No file collisions between shards

### Request Routing

#### Write Operations (Routed to One Shard)
- `/ingest` - Routes to shard based on namespace
- `/delete` - Routes to shard based on namespace
- `/restore` - Routes to shard based on namespace
- `/reset` - Routes to shard if namespace provided, otherwise broadcasts

#### Read Operations (Broadcast to All Shards)
- `/search` - Broadcasts to all shards, merges results by score
- `/stats` - Aggregates statistics from all shards

### Error Handling

- If a shard is down during search, router returns results from healthy shards
- Warnings are included in response if any shards fail
- Write operations fail if target shard is down (expected behavior)

## Configuration

### Environment Variables

#### Shard Configuration
- `SHARD_ID` - Shard identifier (0, 1, 2, ...)
- `SHARD_COUNT` - Total number of shards (default: 1)
- `PORT` - Port to run this shard on (default: 5001)
- `DATA_ROOT` - Base directory for all data (default: ../data)

#### Router Configuration
- `SHARDS` - Comma-separated list of shard URLs (default: "http://127.0.0.1:5001,http://127.0.0.1:5002")
- `ROUTER_PORT` - Port for router service (default: 5003)

### Example: Running 3 Shards

```bash
# Shard 0
SHARD_ID=0 SHARD_COUNT=3 PORT=5001 python app_v2.py

# Shard 1
SHARD_ID=1 SHARD_COUNT=3 PORT=5002 python app_v2.py

# Shard 2
SHARD_ID=2 SHARD_COUNT=3 PORT=5003 python app_v2.py

# Router
SHARDS="http://127.0.0.1:5001,http://127.0.0.1:5002,http://127.0.0.1:5003" ROUTER_PORT=5003 python router.py
```

## API Endpoints

### Router Endpoints

All endpoints maintain the same JSON schema as shard endpoints, with additional routing metadata:

- `GET /` - Router information
- `GET /health` - Health check (checks all shards)
- `GET /whoami` - Router information
- `POST /ingest` - Ingest documents (routed by namespace)
- `POST /search` - Search vectors (broadcast, merged results)
- `POST /delete` - Delete vectors (routed by namespace)
- `POST /restore` - Restore vectors (routed by namespace)
- `GET /stats` - Statistics (aggregated from all shards)
- `POST /reset` - Reset data (routed by namespace or broadcast)

### Shard Endpoints

Each shard exposes the same endpoints as before, plus:

- `GET /whoami` - Returns shard_id, shard_count, data_dir, namespaces_count

## Postman Examples

### 1. Check Router Status

```http
GET http://localhost:5003/whoami
```

**Response:**
```json
{
  "type": "router",
  "shards": ["http://127.0.0.1:5001", "http://127.0.0.1:5002"],
  "shard_count": 2,
  "port": 5000
}
```

### 2. Check Shard Status

```http
GET http://localhost:5001/whoami
```

**Response:**
```json
{
  "shard_id": 0,
  "shard_count": 2,
  "is_shard_mode": true,
  "data_dir": "/path/to/data/shards/shard0",
  "port": 5001,
  "namespaces_count": 1
}
```

### 3. Ingest to Namespace "alpha" (Routes to Shard X)

```http
POST http://localhost:5003/ingest
Content-Type: application/json

{
  "docs_path": "../docs",
  "chunk_size": 800,
  "overlap": 120,
  "namespace": "alpha"
}
```

**Response includes routing info:**
```json
{
  "success": true,
  "message": "Documents ingested successfully",
  "routed_to_shard": 0,
  "shard_url": "http://127.0.0.1:5001",
  "stats": { ... }
}
```

### 4. Ingest to Namespace "beta" (Routes to Shard Y)

```http
POST http://localhost:5003/ingest
Content-Type: application/json

{
  "docs_path": "../docs",
  "namespace": "beta"
}
```

**Response:**
```json
{
  "success": true,
  "routed_to_shard": 1,
  "shard_url": "http://127.0.0.1:5002",
  ...
}
```

### 5. Search (Broadcast to All Shards, Merged Results)

```http
POST http://localhost:5003/search
Content-Type: application/json

{
  "query": "machine learning",
  "k": 10,
  "vector_type": "both"
}
```

**Response:**
```json
{
  "query": "machine learning",
  "k": 10,
  "text_results": [
    {
      "vector_id": 123,
      "score": 0.95,
      "namespace": "alpha",
      "metadata": { ... }
    },
    ...
  ],
  "image_results": [ ... ],
  "shards_queried": 2,
  "total_shards": 2
}
```

### 6. Get Aggregated Statistics

```http
GET http://localhost:5003/stats
```

**Response:**
```json
{
  "aggregated": {
    "total_active_vectors": 1500,
    "total_deleted_vectors": 50,
    "namespace_counts": {
      "alpha": 800,
      "beta": 700
    },
    "namespaces": {
      "alpha": { ... },
      "beta": { ... }
    }
  },
  "shards": {
    "http://127.0.0.1:5001": { ... },
    "http://127.0.0.1:5002": { ... }
  },
  "shard_count": 2,
  "shards_queried": 2
}
```

### 7. Delete from Namespace (Routed)

```http
POST http://localhost:5003/delete
Content-Type: application/json

{
  "doc_id": "document.txt",
  "namespace": "alpha"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Deleted 5 vector(s)",
  "routed_to_shard": 0,
  "shard_url": "http://127.0.0.1:5001",
  "details": { ... }
}
```

## Testing Sharding

### Verify Namespace Routing

1. Ingest to namespace "alpha" and check which shard it goes to:
   ```bash
   curl -X POST http://localhost:5003/ingest -H "Content-Type: application/json" \
     -d '{"docs_path": "../docs", "namespace": "alpha"}'
   ```

2. Check shard 0:
   ```bash
   curl http://localhost:5001/whoami
   ```

3. Check shard 1:
   ```bash
   curl http://localhost:5002/whoami
   ```

4. Verify "alpha" is on the correct shard based on hash.

### Verify Search Merging

1. Ingest different documents to different namespaces on different shards
2. Search via router - should return results from both shards
3. Verify de-duplication works (same document shouldn't appear twice)

## Production Deployment

### Same Machine (Development/Testing)
- Use different ports for each shard
- Use shard-specific data directories (automatic)
- Run router on a separate port

### Different Machines (Production)
- Set `SHARDS` environment variable to actual machine URLs:
  ```bash
  SHARDS="http://shard0.example.com:5001,http://shard1.example.com:5001" python router.py
  ```
- Each shard runs on its own machine with `SHARD_ID` and `SHARD_COUNT` set
- Router can run on a separate machine or load balancer

## Troubleshooting

### Shard Not Responding
- Check shard logs
- Verify shard is running: `curl http://localhost:5001/health`
- Router will include warnings in response if shard is down

### Namespace on Wrong Shard
- Verify `SHARD_COUNT` is the same on all shards
- Check that namespace hash is consistent (use `/whoami` to verify shard_id)

### Search Results Missing
- Check if namespace exists on the shard
- Verify search is broadcasting to all shards (check router logs)
- Check for errors in response warnings

## Design Decisions

1. **Namespace-based routing**: Simple and deterministic. All data for a namespace goes to one shard.

2. **Stable hashing**: Uses xxhash (fast) or SHA1 (fallback) to ensure consistent routing across restarts.

3. **Separate data directories**: Each shard has its own `data/shards/shard{N}/` directory to avoid file collisions.

4. **Thin router**: Router is a proxy - doesn't change request/response schema, just routes and merges.

5. **Parallel search**: Uses ThreadPoolExecutor to query all shards concurrently for low latency.

6. **Graceful degradation**: If one shard is down, search still returns results from healthy shards.
