# How Search Works in Sharded Mode

## Overview

**YES, search queries BOTH shards and merges results!** Here's exactly how it works:

## Search Flow

```
Client Request
    ↓
Router (Port 5003)
    ↓
    ├─→ Shard 0 (Port 5001) ──┐
    │   POST /search            │
    │   Query: "machine learning"│
    │                           │
    └─→ Shard 1 (Port 5002) ───┤
        POST /search            │
        Query: "machine learning"│
                                │
        ┌───────────────────────┘
        ↓
    Router merges results:
    - Combines text_results from both shards
    - Combines image_results from both shards
    - De-duplicates (same document won't appear twice)
    - Sorts by score (highest first)
    - Returns top-k results
        ↓
    Client receives merged results
```

## Step-by-Step Process

### 1. Client Sends Search Request

```bash
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 10
  }'
```

### 2. Router Broadcasts to ALL Shards (Parallel)

The router **simultaneously** sends the same search request to:
- `http://127.0.0.1:5001/search` (Shard 0)
- `http://127.0.0.1:5002/search` (Shard 1)

**Code:**
```python
with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
    futures = {
        executor.submit(forward_request, shard_url, "POST", "/search", json_data=data): shard_url
        for shard_url in SHARDS  # Both shards!
    }
```

### 3. Each Shard Searches Its Own Data

- **Shard 0** searches its namespace indices and returns results
- **Shard 1** searches its namespace indices and returns results

Each shard returns:
```json
{
  "query": "machine learning",
  "k": 10,
  "text_results": [
    {"vector_id": 123, "score": 0.95, "metadata": {...}},
    {"vector_id": 456, "score": 0.88, "metadata": {...}}
  ],
  "image_results": [...]
}
```

### 4. Router Merges Results

The router:
1. **Collects** results from both shards
2. **Merges** text_results from shard 0 + shard 1
3. **Merges** image_results from shard 0 + shard 1
4. **De-duplicates** by (source_file, chunk_id) or vector_id
5. **Sorts** by score (descending)
6. **Takes top-k** results

**Code:**
```python
# Merge text results from all shards
for shard_url, result in shard_results.items():
    text_results = result.get("text_results", [])
    for item in text_results:
        # De-duplicate
        if dedup_key not in seen_text:
            merged_text_results.append(item)

# Sort by score and take top-k
merged_text_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
final_text_results = merged_text_results[:k]
```

### 5. Client Receives Merged Results

```json
{
  "query": "machine learning",
  "k": 10,
  "text_results": [
    // Results from BOTH shards, sorted by score
    {"vector_id": 123, "score": 0.95, "metadata": {...}},  // From shard 0
    {"vector_id": 789, "score": 0.92, "metadata": {...}},  // From shard 1
    {"vector_id": 456, "score": 0.88, "metadata": {...}},  // From shard 0
    ...
  ],
  "image_results": [...],
  "shards_queried": 2,
  "total_shards": 2,
  "shards_searched": [
    "http://127.0.0.1:5001",
    "http://127.0.0.1:5002"
  ],
  "shard_result_counts": {
    "http://127.0.0.1:5001": {
      "text_results": 5,
      "image_results": 2
    },
    "http://127.0.0.1:5002": {
      "text_results": 3,
      "image_results": 1
    }
  }
}
```

## Example: Real Search Scenario

### Setup
- **Shard 0** has: `doc1.pdf`, `doc2.pdf`, `doc3.pdf` in namespace "alpha"
- **Shard 1** has: `doc4.pdf`, `doc5.pdf`, `doc6.pdf` in namespace "alpha"

### Search Request
```bash
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "technology", "k": 5}'
```

### What Happens

1. **Router sends to Shard 0:**
   - Searches: `doc1.pdf`, `doc2.pdf`, `doc3.pdf`
   - Returns: 3 results

2. **Router sends to Shard 1:**
   - Searches: `doc4.pdf`, `doc5.pdf`, `doc6.pdf`
   - Returns: 2 results

3. **Router merges:**
   - Combines: 3 + 2 = 5 results
   - Sorts by score
   - Returns top 5

4. **Client gets:**
   - Results from **both** shards in one response!

## Verification

### Check Which Shards Were Queried

The response includes:
```json
{
  "shards_searched": [
    "http://127.0.0.1:5001",
    "http://127.0.0.1:5002"
  ],
  "shard_result_counts": {
    "http://127.0.0.1:5001": {"text_results": 5},
    "http://127.0.0.1:5002": {"text_results": 3}
  }
}
```

This shows:
- ✅ Both shards were queried
- ✅ How many results each shard returned
- ✅ Total merged results = sum of both

### Test It Yourself

1. **Ingest to different namespaces:**
   ```bash
   # This will distribute files across shards
   curl -X POST http://localhost:5003/ingest \
     -H "Content-Type: application/json" \
     -d '{"docs_path": "../docs", "namespace": "test"}'
   ```

2. **Search:**
   ```bash
   curl -X POST http://localhost:5003/search \
     -H "Content-Type: application/json" \
     -d '{"query": "your query", "k": 10}'
   ```

3. **Check the response:**
   - Look for `shards_searched` - should show both shards
   - Look for `shard_result_counts` - shows results from each shard
   - `text_results` and `image_results` contain merged results from both

## Key Points

✅ **Search ALWAYS queries both shards** - no routing, always broadcast  
✅ **Results are merged** - combined from both shards  
✅ **De-duplication** - same document won't appear twice  
✅ **Sorted by score** - best matches first  
✅ **Parallel execution** - both shards queried simultaneously (fast!)  
✅ **Fault tolerant** - if one shard fails, results from the other are still returned  

## Console Logs

When you search, you'll see in the router console:
```
Search: Broadcasting to 2 shards: ['http://127.0.0.1:5001', 'http://127.0.0.1:5002']
Search SUCCESS [Shard http://127.0.0.1:5001]: Got 5 text results, 2 image results
Search SUCCESS [Shard http://127.0.0.1:5002]: Got 3 text results, 1 image results
Search COMPLETE: Merged 8 text + 3 image results from 2/2 shards
```

This confirms both shards are being queried and results are merged!
