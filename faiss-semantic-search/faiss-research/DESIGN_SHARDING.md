# Sharding Design Summary

## Overview

This document explains the design and implementation of the two-shard sharding mode for the FAISS vector database.

## Design Principles

1. **Single Codebase**: The same `app_v2.py` runs on different ports with different configurations - no code duplication.
2. **Isolated Storage**: Each shard uses its own data directory (`data/shards/shard{N}/`) to prevent file collisions.
3. **Deterministic Routing**: Uses stable hashing (xxhash/SHA1) to route namespaces to shards consistently.
4. **Thin Router**: Router is a stateless proxy that routes requests without changing business logic.

## Architecture

### Data Flow

```
Client Request
    ↓
Router (Port 5003)
    ├─ Write Operations → Route to shard based on namespace hash
    └─ Read Operations → Broadcast to all shards, merge results
```

### Storage Layout

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
│       └── (same structure)
```

**Why separate directories?**
- Prevents file collisions when running multiple shards on the same machine
- Each shard has its own SQLite database
- Easy to backup/restore individual shards
- Can be moved to separate machines later

## Implementation Details

### 1. Configuration System (`src/config.py`)

- Reads environment variables: `SHARD_ID`, `SHARD_COUNT`, `PORT`, `DATA_ROOT`
- Computes shard-specific data directory: `data/shards/shard{SHARD_ID}/`
- Provides singleton pattern via `get_config()`

**Key Design Decision**: Configuration is loaded at module import time, ensuring consistent behavior across the application.

### 2. Shard Mode (`app_v2.py`)

**Changes:**
- Imports and uses `config` to get data directory and port
- `VectorStoreV2` initialized with `config.get_data_dir()` instead of hardcoded path
- Server runs on `config.port` instead of hardcoded 5001
- Added `/whoami` endpoint to identify shard

**Backward Compatibility**: If `SHARD_COUNT=1` (default), behaves exactly as before (no sharding).

### 3. Router Service (`router.py`)

**Routing Logic:**

For Ingest (file distribution):
```python
# Each file is routed independently
for file in files_in_folder:
    shard_index = stable_hash(f"{filename}:{namespace}") % SHARD_COUNT
    # File goes to that shard
```

For Delete/Restore (document-based):
```python
shard_index = stable_hash(doc_id) % SHARD_COUNT
```

**Stable Hashing:**
- Uses `xxhash` (fast, deterministic) if available
- Falls back to `SHA1` (slower but always available)
- Ensures same namespace always routes to same shard

**Write Operations** (Routed):
- `/ingest` - Routes to shard based on namespace
- `/delete` - Routes to shard based on namespace  
- `/restore` - Routes to shard based on namespace
- `/reset` - Routes if namespace provided, otherwise broadcasts

**Read Operations** (Broadcast):
- `/search` - Broadcasts to all shards in parallel using `ThreadPoolExecutor`
- Merges results by score (descending)
- De-duplicates by `(source_file, chunk_id/image_id)` or `vector_id`
- Returns top-k results
- Includes warnings if any shards fail

- `/stats` - Aggregates statistics from all shards
- Returns per-shard breakdown + aggregated totals

**Error Handling:**
- Uses `requests.Session` with retry strategy
- If shard fails during search, returns results from healthy shards
- Includes error details in response warnings
- Write operations fail if target shard is down (expected)

### 4. Development Scripts

**Shell Scripts:**
- `run_shard0.sh` - Sets env vars and runs shard 0
- `run_shard1.sh` - Sets env vars and runs shard 1
- `run_router.sh` - Sets env vars and runs router

**Makefile:**
- `make shard0` - Run shard 0
- `make shard1` - Run shard 1
- `make router` - Run router
- `make help` - Show usage

## Routing Example

### Ingest: File Distribution
When ingesting folder with files: `["doc1.pdf", "doc2.pdf", "doc3.pdf"]` in namespace "alpha":

```python
hash("doc1.pdf:alpha") % 2 = 0  # doc1.pdf → Shard 0
hash("doc2.pdf:alpha") % 2 = 1  # doc2.pdf → Shard 1
hash("doc3.pdf:alpha") % 2 = 0  # doc3.pdf → Shard 0
```

Result: Files distributed across both shards for parallel processing!

### Delete/Restore: Document-based
```python
hash("doc1.pdf") % 2 = 0  # Routes to shard 0
hash("doc2.pdf") % 2 = 1  # Routes to shard 1
```

### Search Query
```
Client → Router → [Shard 0, Shard 1] (parallel)
                ↓
            Merge results
                ↓
            Return top-k
```

## Why This Design?

1. **No Code Duplication**: Single codebase runs in shard mode via environment variables.

2. **Deterministic Routing**: Same namespace always goes to same shard, enabling consistent behavior.

3. **Isolated Storage**: Each shard's data is completely separate, preventing corruption.

4. **Scalable**: Easy to add more shards by updating `SHARD_COUNT` and adding shard URLs to router.

5. **Fault Tolerant**: Search continues working even if one shard is down.

6. **Thin Router**: Router doesn't need to understand vector operations, just routes HTTP requests.

## Testing Strategy

1. **Verify Routing**: Ingest to different namespaces, check which shard receives them.
2. **Verify Merging**: Search should return results from both shards, properly merged.
3. **Verify Isolation**: Check that shard 0's data doesn't appear in shard 1's directory.
4. **Verify Fault Tolerance**: Stop one shard, verify search still works with warnings.

## Production Considerations

1. **Multiple Machines**: Set `SHARDS` to actual machine URLs in router.
2. **Load Balancing**: Router can be behind a load balancer.
3. **Monitoring**: Each shard exposes `/whoami` and `/health` for monitoring.
4. **Backup**: Each shard's data directory can be backed up independently.

## Future Enhancements

1. **Replication**: Add read replicas for each shard.
2. **Dynamic Sharding**: Add/remove shards without downtime.
3. **Cross-Shard Queries**: Support queries that span multiple shards with custom logic.
4. **Shard Metadata**: Track which namespaces are on which shards for debugging.
