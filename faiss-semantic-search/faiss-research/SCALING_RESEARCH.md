# Scaling Guide for FAISS Vector Database
## Handling Large Datasets and Multi-Machine Deployment

---

## Table of Contents
1. The Problem
2. Scaling on a Single Machine
3. Scaling Across Multiple Machines
4. Step-by-Step Implementation Plan
5. Key Design Considerations
6. Summary

---

## 1. The Problem

### What Happens as Data Grows?

**Current State**
- We are using `IndexFlat` for vector search.
- `IndexFlat` performs a brute-force comparison against every vector.
- This approach is simple and accurate, but does not scale.

**Why It Becomes a Problem**
- Search time grows linearly with the number of vectors (O(N)).
- As the dataset grows, latency increases beyond acceptable limits.
- Memory usage also increases linearly with vector count.

**Practical Limits**
- ~10,000 vectors → Fast and acceptable
- ~100,000 vectors → Noticeable slowdown
- ~1,000,000+ vectors → Search latency becomes impractical

**Memory Considerations**
- Each 1024-dim float32 vector ≈ 4 KB
- 1 million vectors ≈ 4 GB RAM (vectors only)
- Additional memory is required for metadata and indexing structures

---

## 2. Scaling on a Single Machine

Before moving to multiple machines, the first and most effective step is to **use a smarter FAISS index**.

---

### Option A: IVF (Inverted File Index)

#### What is IVF?
IVF clusters vectors into groups (called “lists”) and only searches the most relevant clusters instead of the entire dataset.

**Analogy**
- Instead of searching every book in a library, first select relevant sections, then search only those shelves.

#### How It Works
1. Vectors are grouped into `nlist` clusters.
2. During search, only `nprobe` clusters are scanned.
3. This dramatically reduces the number of comparisons.

#### Benefits
- 10–100× faster than `IndexFlat`
- Scales well up to tens of millions of vectors
- Good balance between speed and accuracy

#### Trade-offs
- Slightly lower recall (typically 95–99%)
- Requires training before use

#### Example Code
```python
import faiss

dimension = 1024
nlist = 4096  # Number of clusters (≈ sqrt(total_vectors))
quantizer = faiss.IndexFlatIP(dimension)

base_index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

index = faiss.IndexIDMap2(base_index)

# Train using a random subset of data
index.train(training_vectors)

# Add vectors
index.add_with_ids(vectors, ids)

# Number of clusters to probe during search
index.nprobe = 32
````

#### Parameter Guidelines

* `nlist`: ≈ sqrt(number of vectors)
* `nprobe`: 16–64 (higher = better recall, slower search)

---

### Option B: HNSW (Hierarchical Navigable Small World)

#### What is HNSW?

HNSW builds a graph where each vector is connected to its nearest neighbors. Searches traverse the graph instead of scanning clusters.

#### Benefits

* Extremely fast search
* Excellent recall
* No training required

#### Trade-offs

* Higher memory usage than IVF
* More complex internal structure

#### Example Code

```python
import faiss

dimension = 1024
M = 32  # Graph connectivity

index = faiss.IndexHNSWFlat(dimension, M)
index.add(vectors)

index.hnsw.efSearch = 100
```

#### Parameter Guidelines

* `M`: 16–64 (higher = better recall, more memory)
* `efSearch`: 50–200 (higher = better recall, slower)

---

### Index Selection Guide

| Data Size      | Recommended Index         |
| -------------- | ------------------------- |
| < 100K vectors | IndexFlat                 |
| 100K – 1M      | IVF                       |
| 1M – 10M       | HNSW or IVF               |
| 10M+           | HNSW or IVF + compression |

---

## 3. Scaling Across Multiple Machines

When a single machine reaches CPU, memory, or disk limits, horizontal scaling is required.

### Core Idea

Split data across multiple machines (shards) and search them in parallel.

---

### Architecture Overview

```
                   ┌─────────────┐
                   │   Router    │
                   │   Service   │
                   └──────┬──────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐        ┌───▼─────┐        ┌───▼─────┐
   │ Shard 1 │        │ Shard 2 │        │ Shard 3 │
   │ (FAISS) │        │ (FAISS) │        │ (FAISS) │
   └─────────┘        └─────────┘        └─────────┘
```

---

### Sharding Strategy (Recommended)

#### Hash-Based Sharding

* Each document is assigned to a shard using a **stable hash** of `doc_id`.
* Ensures even data distribution.
* Simple and reliable.

**Important Note**
Python’s built-in `hash()` is not stable across processes or restarts.

**Use a stable hash instead**

```python
import xxhash

def shard_for_doc(doc_id, num_shards):
    return xxhash.xxh64(doc_id).intdigest() % num_shards
```

---

### Router Responsibilities

* Accept ingest/search requests
* Route inserts to the correct shard
* Broadcast search queries to all shards
* Merge and rank results
* Skip unhealthy shards

---

### Search Flow

1. Router sends query to all healthy shards (in parallel).
2. Each shard returns top-K results.
3. Router merges and re-ranks results globally.
4. Router returns final top-K to the client.

---

### Failure Handling

* If a shard is down, it is skipped.
* Partial results are still returned.
* System remains available.

---

### Adding More Machines

* Add new shard URL to router configuration.
* New data automatically flows to new shard.
* Old data remains on existing shards.
* Optional rebalancing can be added later if needed.

---

## 4. Step-by-Step Implementation Plan

### Phase 1: Optimize Single Machine (1–2 days)

1. Measure search latency.
2. Switch from `IndexFlat` to IVF.
3. Tune `nlist` and `nprobe`.
4. Persist trained index to disk.
5. Reload index on startup.

---

### Phase 2: Multi-Machine Setup (3–5 days)

1. Deploy FAISS service on multiple machines.
2. Introduce a router service.
3. Implement hash-based sharding.
4. Parallelize shard search requests.
5. Add health checks.

---

### Phase 3: Production Hardening

1. Introduce Redis caching.
2. Add monitoring and alerts.
3. Implement snapshot + restore.
4. Support deletions using `IndexIDMap2`.
5. Periodically compact/rebuild indexes.

---

## 5. Key Design Considerations

* Always use stable vector IDs (`IndexIDMap2`)
* Do not rely on vector position for metadata
* Persist indexes atomically (temp file + rename)
* Use namespaces for logical isolation
* Optimize before scaling horizontally

---

## 6. Summary

* `IndexFlat` is simple but does not scale.
* IVF and HNSW make single machines significantly faster.
* Horizontal scaling is achieved via sharding + routing.
* This design mirrors how production vector databases work.
* Start simple, measure, and scale incrementally.

---

**Recommended Path**

1. Upgrade to IVF/HNSW
2. Add persistence and deletion
3. Introduce sharding with a router
4. Harden for production
