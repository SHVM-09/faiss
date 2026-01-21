# Complete Usage Guide - Sharded FAISS Vector DB

## Quick Start

### 1. Start All Services

Open **3 separate terminals**:

**Terminal 1 - Shard 0:**
```bash
cd faiss-research
make shard0
# or: ./run_shard0.sh
```

**Terminal 2 - Shard 1:**
```bash
cd faiss-research
make shard1
# or: ./run_shard1.sh
```

**Terminal 3 - Router:**
```bash
cd faiss-research
make router
# or: ./run_router.sh
```

### 2. Verify Services Are Running

```bash
# Check router
curl http://localhost:5003/

# Check shard 0
curl http://localhost:5001/whoami

# Check shard 1
curl http://localhost:5002/whoami
```

---

## All API Endpoints via Router (Port 5003)

### Base URL
All requests go to: `http://localhost:5003`

---

## 1. Router Information

### GET `/` - Router Info
```bash
curl http://localhost:5003/
```

**Response:**
```json
{
  "message": "FAISS Semantic Search Router",
  "version": "1.0.0",
  "shards": ["http://127.0.0.1:5001", "http://127.0.0.1:5002"],
  "shard_count": 2,
  "endpoints": { ... }
}
```

---

## 2. Health Check

### GET `/health` - Check All Shards
```bash
curl http://localhost:5003/health
```

**Response:**
```json
{
  "ok": true,
  "shards": {
    "http://127.0.0.1:5001": {"ok": true},
    "http://127.0.0.1:5002": {"ok": true}
  }
}
```

---

## 3. Router Identity

### GET `/whoami` - Router Info
```bash
curl http://localhost:5003/whoami
```

**Response:**
```json
{
  "type": "router",
  "shards": ["http://127.0.0.1:5001", "http://127.0.0.1:5002"],
  "shard_count": 2,
  "port": 5003
}
```

---

## 4. Ingest Documents

### POST `/ingest` - Add Documents to Vector DB

**Routes to shard based on namespace hash**

```bash
curl -X POST http://localhost:5003/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "docs_path": "../docs",
    "chunk_size": 800,
    "overlap": 120,
    "extract_images": true,
    "namespace": "alpha"
  }'
```

**Parameters:**
- `docs_path` (required): Path to folder with documents
- `chunk_size` (optional, default: 800): Max characters per chunk
- `overlap` (optional, default: 120): Characters to overlap between chunks
- `extract_images` (optional, default: true): Load and index image files
- `namespace` (optional, default: "default"): Namespace for this data

**Response:**
```json
{
  "success": true,
  "message": "Documents ingested successfully",
  "routed_to_shard": 0,
  "shard_url": "http://127.0.0.1:5001",
  "index_type": "IVF_PQ",
  "was_trained": false,
  "stats": {
    "files_loaded": 10,
    "text_files_loaded": 8,
    "image_files_loaded": 2,
    "chunks_created": 45,
    "text_vectors_stored": 45,
    "namespace": "alpha"
  }
}
```

**Example: Ingest to Different Namespaces**
```bash
# This will route to shard 0 or 1 based on namespace hash
curl -X POST http://localhost:5003/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "../docs", "namespace": "alpha"}'

curl -X POST http://localhost:5003/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "../docs", "namespace": "beta"}'
```

---

## 5. Search Vectors

### POST `/search` - Search Across All Shards

**Broadcasts to all shards and merges results**

```bash
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 10,
    "namespace": "alpha",
    "vector_type": "both"
  }'
```

**Parameters:**
- `query` (required): Text query to search for
- `k` (optional, default: 5): Number of results to return
- `namespace` (optional): Filter by namespace (searches all if not provided)
- `vector_type` (optional, default: "both"): "text", "image", or "both"

**Response:**
```json
{
  "query": "machine learning",
  "k": 10,
  "namespace": "alpha",
  "text_results": [
    {
      "vector_id": 123,
      "score": 0.95,
      "namespace": "alpha",
      "metadata": {
        "chunk_text": "...",
        "source_file": "document.pdf",
        "type": "text"
      }
    },
    ...
  ],
  "image_results": [
    {
      "vector_id": 456,
      "score": 0.88,
      "namespace": "alpha",
      "metadata": {
        "image_id": "document.pdf::img_0",
        "source_file": "document.pdf",
        "type": "image"
      }
    },
    ...
  ],
  "shards_queried": 2,
  "total_shards": 2
}
```

**Examples:**
```bash
# Search text only
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "k": 5, "vector_type": "text"}'

# Search images only
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "car", "k": 5, "vector_type": "image"}'

# Search all namespaces (don't specify namespace)
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "technology", "k": 10}'
```

---

## 6. Delete Vectors

### POST `/delete` - Delete by Document or Chunk

**Routes to shard based on namespace**

```bash
# Delete all vectors for a document
curl -X POST http://localhost:5003/delete \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "document.pdf",
    "namespace": "alpha",
    "hard_delete": false
  }'
```

**Parameters:**
- `doc_id` OR `chunk_id` (required): Document ID or specific chunk ID
- `namespace` (optional): Namespace filter
- `hard_delete` (optional, default: false): If true, permanently deletes (cannot restore)

**Response:**
```json
{
  "success": true,
  "message": "Deleted 5 vector(s) for doc_id: document.pdf",
  "routed_to_shard": 0,
  "shard_url": "http://127.0.0.1:5001",
  "details": {
    "doc_id": "document.pdf",
    "namespace": "alpha",
    "total_deleted": 5,
    "text_vectors_deleted": 3,
    "image_vectors_deleted": 2,
    "index_updated": true,
    "database_updated": true,
    "deletion_type": "soft"
  }
}
```

**Examples:**
```bash
# Soft delete (can restore)
curl -X POST http://localhost:5003/delete \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "file.txt", "namespace": "alpha"}'

# Hard delete (permanent)
curl -X POST http://localhost:5003/delete \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "file.txt", "namespace": "alpha", "hard_delete": true}'

# Delete specific chunk
curl -X POST http://localhost:5003/delete \
  -H "Content-Type: application/json" \
  -d '{"chunk_id": "file.txt::chunk_0", "namespace": "alpha"}'
```

---

## 7. Restore Vectors

### POST `/restore` - Restore Soft-Deleted Vectors

**Routes to shard based on namespace**

```bash
curl -X POST http://localhost:5003/restore \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "document.pdf",
    "namespace": "alpha"
  }'
```

**Parameters:**
- `doc_id` OR `chunk_id` (required): Document ID or specific chunk ID
- `namespace` (optional): Namespace filter

**Response:**
```json
{
  "success": true,
  "message": "Restored 5 vectors for doc_id: document.pdf",
  "routed_to_shard": 0,
  "shard_url": "http://127.0.0.1:5001",
  "details": {
    "doc_id": "document.pdf",
    "namespace": "alpha",
    "total_restored": 5,
    "text_vectors_restored": 3,
    "image_vectors_restored": 2,
    "database_updated": true,
    "note": "Vectors are restored in database but not in FAISS index. Re-ingest document to add them back to index."
  }
}
```

**Examples:**
```bash
# Restore document
curl -X POST http://localhost:5003/restore \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "file.txt", "namespace": "alpha"}'

# Restore specific chunk
curl -X POST http://localhost:5003/restore \
  -H "Content-Type: application/json" \
  -d '{"chunk_id": "file.txt::chunk_0", "namespace": "alpha"}'
```

---

## 8. Get Statistics

### GET `/stats` - Aggregated Statistics from All Shards

**Aggregates stats from all shards**

```bash
curl http://localhost:5003/stats
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
      "alpha": {
        "index_type": "IVF_PQ",
        "trained": true,
        "text_vectors": 600,
        "image_vectors": 200,
        "nlist": 256,
        "nprobe": 16
      },
      "beta": {
        "index_type": "IVF_PQ",
        "trained": true,
        "text_vectors": 500,
        "image_vectors": 200,
        "nlist": 256,
        "nprobe": 16
      }
    }
  },
  "shards": {
    "http://127.0.0.1:5001": {
      "total_active_vectors": 800,
      "namespace_counts": {"alpha": 800},
      "namespaces": { ... }
    },
    "http://127.0.0.1:5002": {
      "total_active_vectors": 700,
      "namespace_counts": {"beta": 700},
      "namespaces": { ... }
    }
  },
  "shard_count": 2,
  "shards_queried": 2
}
```

---

## 9. Reset/Clear All Data

### POST `/reset` - Clear All Data

**Routes to shard if namespace provided, otherwise broadcasts to all**

```bash
# Reset specific namespace
curl -X POST http://localhost:5003/reset \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "alpha"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Database and index reset successfully - all data cleared",
  "routed_to_shard": 0,
  "shard_url": "http://127.0.0.1:5001"
}
```

**Reset All Namespaces (Broadcast):**
```bash
# Reset all data on all shards
curl -X POST http://localhost:5003/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**
```json
{
  "success": true,
  "shards": {
    "http://127.0.0.1:5001": {
      "success": true,
      "message": "Database and index reset successfully - all data cleared"
    },
    "http://127.0.0.1:5002": {
      "success": true,
      "message": "Database and index reset successfully - all data cleared"
    }
  },
  "shard_count": 2
}
```

---

## Complete Workflow Example

### Step 1: Check System Status
```bash
curl http://localhost:5003/health
```

### Step 2: Ingest Documents to Different Namespaces
```bash
# Ingest to namespace "alpha" (routes to shard 0 or 1)
curl -X POST http://localhost:5003/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "../docs", "namespace": "alpha"}'

# Ingest to namespace "beta" (routes to different shard)
curl -X POST http://localhost:5003/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "../docs", "namespace": "beta"}'
```

### Step 3: Check Which Shard Has Which Namespace
```bash
# Check shard 0
curl http://localhost:5001/whoami

# Check shard 1
curl http://localhost:5002/whoami
```

### Step 4: Search Across All Shards
```bash
# Search all namespaces
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "technology", "k": 10}'

# Search specific namespace
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "technology", "k": 10, "namespace": "alpha"}'
```

### Step 5: Get Statistics
```bash
curl http://localhost:5003/stats
```

### Step 6: Delete a Document
```bash
curl -X POST http://localhost:5003/delete \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "document.pdf", "namespace": "alpha"}'
```

### Step 7: Restore a Document
```bash
curl -X POST http://localhost:5003/restore \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "document.pdf", "namespace": "alpha"}'
```

---

## Using with Postman

### Import Collection

Create a Postman collection with these requests:

1. **Router Info**
   - Method: GET
   - URL: `http://localhost:5003/`

2. **Health Check**
   - Method: GET
   - URL: `http://localhost:5003/health`

3. **Ingest Documents**
   - Method: POST
   - URL: `http://localhost:5003/ingest`
   - Body (JSON):
     ```json
     {
       "docs_path": "../docs",
       "namespace": "alpha"
     }
     ```

4. **Search**
   - Method: POST
   - URL: `http://localhost:5003/search`
   - Body (JSON):
     ```json
     {
       "query": "machine learning",
       "k": 10,
       "vector_type": "both"
     }
     ```

5. **Delete**
   - Method: POST
   - URL: `http://localhost:5003/delete`
   - Body (JSON):
     ```json
     {
       "doc_id": "document.pdf",
       "namespace": "alpha"
     }
     ```

6. **Restore**
   - Method: POST
   - URL: `http://localhost:5003/restore`
   - Body (JSON):
     ```json
     {
       "doc_id": "document.pdf",
       "namespace": "alpha"
     }
     ```

7. **Stats**
   - Method: GET
   - URL: `http://localhost:5003/stats`

8. **Reset**
   - Method: POST
   - URL: `http://localhost:5003/reset`
   - Body (JSON):
     ```json
     {
       "namespace": "alpha"
     }
     ```

---

## Python Client Example

```python
import requests

ROUTER_URL = "http://localhost:5003"

# Ingest documents
response = requests.post(f"{ROUTER_URL}/ingest", json={
    "docs_path": "../docs",
    "namespace": "alpha"
})
print(response.json())

# Search
response = requests.post(f"{ROUTER_URL}/search", json={
    "query": "machine learning",
    "k": 10
})
results = response.json()
print(f"Found {len(results['text_results'])} text results")
print(f"Found {len(results['image_results'])} image results")

# Get stats
response = requests.get(f"{ROUTER_URL}/stats")
stats = response.json()
print(f"Total vectors: {stats['aggregated']['total_active_vectors']}")
```

---

## Troubleshooting

### Router Can't Connect to Shards
- Verify shards are running: `curl http://localhost:5001/health`
- Check router logs for connection errors
- Verify `SHARDS` environment variable in router

### Search Returns Empty Results
- Check if data exists: `curl http://localhost:5003/stats`
- Verify namespace exists on shards
- Check router response for warnings about failed shards

### Wrong Shard for Namespace
- Verify `SHARD_COUNT` is same on all shards
- Check `/whoami` endpoint on each shard
- Namespace routing is deterministic based on hash

---

## Key Points

1. **Always use router (port 5003)** - Don't call shards directly
2. **Namespace determines shard** - Same namespace always goes to same shard
3. **Search broadcasts** - Searches all shards and merges results
4. **Writes are routed** - Ingest/delete/restore go to specific shard
5. **Stats are aggregated** - Combines statistics from all shards
