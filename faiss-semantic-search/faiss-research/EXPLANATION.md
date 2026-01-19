# FAISS Research - Complete Explanation

## Overview

This is a production-ready FAISS-based vector database for semantic search. It provides stable vector IDs, SQLite metadata storage, deletion support, and scaling capabilities.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How Index is Stored and Persisted](#how-index-is-stored-and-persisted)
3. [How to Remove Data](#how-to-remove-data)
4. [How to Scale with Large Data](#how-to-scale-with-large-data)
5. [API Endpoints](#api-endpoints)
6. [Architecture](#architecture)

---

## Quick Start

### 1. Start the Server

```bash
cd faiss-research
python app_v2.py
```

Server runs on `http://localhost:5001`

### 2. Ingest Documents

```bash
curl -X POST http://localhost:5001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "docs_path": "../docs",
    "chunk_size": 800,
    "overlap": 120,
    "namespace": "my_project"
  }'
```

### 3. Search

```bash
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 5,
    "namespace": "my_project"
  }'
```

---

## How Index is Stored and Persisted

### Storage Architecture

The system uses a **dual-storage architecture**:

1. **FAISS Index Files** - Store the actual vectors
2. **SQLite Database** - Stores metadata and mappings

### Directory Structure

```
data/
├── manifest.json              # Current version + metadata
├── indices/
│   ├── text_index.faiss       # FAISS IndexIDMap2 (text vectors)
│   └── image_index.faiss     # FAISS IndexIDMap2 (image vectors)
├── metadata/
│   └── metadata.db            # SQLite database (metadata)
└── tmp/                       # Temporary files (atomic writes)
```

### How Persistence Works

#### 1. **FAISS Index Storage**

- **Format**: Binary FAISS index files (`.faiss`)
- **Type**: `IndexIDMap2` - Allows stable int64 IDs for each vector
- **Location**: `data/indices/text_index.faiss` and `data/indices/image_index.faiss`
- **Persistence**: Saved to disk after every operation using atomic writes

**Key Points:**
- Each vector has a **stable int64 ID** that persists across restarts
- IDs are auto-incremented (1, 2, 3, ...)
- Vectors are stored in memory during operation, saved to disk on `store.save()`

#### 2. **SQLite Metadata Storage**

- **Database**: `data/metadata/metadata.db`
- **Schema**:
  ```sql
  CREATE TABLE vectors (
      vector_id INTEGER PRIMARY KEY,    -- Stable ID (matches FAISS)
      doc_id TEXT NOT NULL,            -- Document identifier
      chunk_id TEXT NOT NULL,          -- Chunk identifier
      namespace TEXT NOT NULL,         -- Namespace for isolation
      deleted INTEGER NOT NULL,        -- Soft delete flag (0/1)
      vector_type TEXT NOT NULL,       -- 'text' or 'image'
      metadata_json TEXT,              -- JSON metadata (chunk text, etc.)
      created_at TIMESTAMP
  )
  ```

**Key Points:**
- **Not aligned by position** - Uses `vector_id` as primary key
- Survives deletions - Metadata remains even after vector deletion
- Indexed for fast lookups (doc_id, chunk_id, namespace, deleted)

#### 3. **Atomic Writes (Crash-Safe)**

All writes use atomic operations:

```python
# 1. Write to temporary file
tmp_path = "data/tmp/text_index.faiss.tmp"
faiss.write_index(index, tmp_path)

# 2. Atomic rename (crash-safe)
os.rename(tmp_path, final_path)
```

**Benefits:**
- If server crashes during write, old index remains intact
- No partial/corrupted files
- Atomic rename is guaranteed by filesystem

#### 4. **Manifest File**

`data/manifest.json` tracks current state:

```json
{
  "version": "1.0.0",
  "text_vector_count": 1000,
  "image_vector_count": 500,
  "next_vector_id": 1501,
  "indices": {
    "text": {"path": "indices/text_index.faiss", "dimension": 1024},
    "image": {"path": "indices/image_index.faiss", "dimension": 1024}
  }
}
```

**Purpose:**
- Single source of truth for current state
- Version tracking

### Loading Process

On server startup:

1. Load `manifest.json` to get current state
2. Load FAISS indices from `data/indices/*.faiss`
3. Connect to SQLite database
4. Migrate old `IndexFlatIP` to `IndexIDMap2` if needed
5. Ready for operations

---

## How to Remove Data

### Method 1: Delete by Document ID

Deletes all vectors for a specific document:

```bash
POST /delete
{
  "doc_id": "doc1.txt",
  "namespace": "my_project"
}
```

**What happens:**
1. Finds all vectors with `doc_id = "doc1.txt"` in SQLite
2. Calls `index.remove_ids()` to remove from FAISS index
3. Marks rows as `deleted = 1` in SQLite (soft delete)
4. Saves changes

**Result:**
- Vectors removed from FAISS index immediately
- Metadata marked as deleted (for audit trail)
- Search will not return deleted vectors

### Method 2: Delete by Chunk ID

Deletes a specific chunk:

```bash
POST /delete
{
  "chunk_id": "doc1.txt::chunk_0",
  "namespace": "my_project"
}
```

### Method 3: Reset (Clear All Data)

Completely clears all data and starts fresh:

```bash
POST /reset
```

**What happens:**
1. Deletes all vectors from FAISS indices
2. Clears SQLite database (removes all metadata)
3. Deletes index files from disk
4. Resets vector ID counter

**Warning:** This permanently deletes ALL data! Use only when you want to start completely fresh.

---

## How to Scale with Large Data

### Current Setup (IndexFlatIP)

**Best for:** < 100K vectors
- **Type**: Exact search (100% recall)
- **Latency**: ~1-5ms per query
- **Memory**: ~4MB per 1000 vectors (1024-dim)
- **Limitation**: Linear scan - gets slower as data grows

### Scaling Strategy

#### Stage 1: 100K - 1M Vectors → Use IndexIVFFlat

**When to upgrade:**
- Search latency > 10ms
- More than 100K vectors
- Need faster searches

**How to upgrade:**

Modify `src/store_v2.py`:

```python
def initialize_text_index(self, dimension: int = 1024):
    # Create quantizer (flat index for centroids)
    quantizer = faiss.IndexFlatIP(dimension)
    
    # Create IVF index
    nlist = 4096  # Number of clusters (sqrt(num_vectors) or 4096)
    self.text_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Set search parameters
    self.text_index.nprobe = 64  # Search in 64 clusters (higher = better recall, slower)
    
    # Wrap with IDMap for stable IDs
    self.text_index = faiss.IndexIDMap2(self.text_index)
```

**Training required:**
```python
# Before adding vectors, train on sample data
training_vectors = ...  # Sample of your vectors
self.text_index.train(training_vectors)
```

**Tradeoffs:**
- **Recall**: 98-99% (vs 100% exact)
- **Latency**: ~5-10ms (vs 1-5ms)
- **Memory**: Same as IndexFlatIP
- **Speed**: 10-100x faster for large datasets

#### Stage 2: 1M - 10M Vectors → Use IndexIVFPQ

**When to upgrade:**
- Memory constrained
- More than 1M vectors
- Acceptable to trade 2-3% recall for 4x memory reduction

**How to upgrade:**

```python
def initialize_text_index(self, dimension: int = 1024):
    quantizer = faiss.IndexFlatIP(dimension)
    nlist = 4096
    m = 64  # Number of sub-vectors (dimension / 16)
    nbits = 8  # Bits per sub-vector
    
    self.text_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
    self.text_index.nprobe = 64
    self.text_index = faiss.IndexIDMap2(self.text_index)
```

**Tradeoffs:**
- **Recall**: 95-98% (vs 100%)
- **Latency**: ~10-20ms
- **Memory**: 4x reduction (1MB per 1000 vectors)
- **Speed**: Still fast for large datasets

#### Stage 3: 10M+ Vectors → Use IndexHNSW

**When to upgrade:**
- Very large datasets (10M+)
- Need sub-20ms latency
- Can afford higher memory

**How to upgrade:**

```python
def initialize_text_index(self, dimension: int = 1024):
    # HNSW parameters
    M = 32  # Number of connections (higher = better recall, more memory)
    efConstruction = 200  # Construction time vs quality
    
    base_index = faiss.IndexHNSWFlat(dimension, M)
    base_index.hnsw.efConstruction = efConstruction
    base_index.hnsw.efSearch = 64  # Search parameter
    
    self.text_index = faiss.IndexIDMap2(base_index)
```

**Tradeoffs:**
- **Recall**: 95-97%
- **Latency**: ~15-20ms (consistent)
- **Memory**: Higher (but scales well)
- **Speed**: Very fast for very large datasets

### Multi-Machine Sharding

For datasets that don't fit on a single machine:

#### Architecture

```
┌─────────────────┐
│  Router Service │  (Flask API)
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
┌───▼───┐ ┌──▼───┐  ┌───▼───┐  ┌───▼───┐
│Shard 0│ │Shard 1│  │Shard 2│  │Shard 3│
│(0-25%)│ │(26-50%)│ │(51-75%)│ │(76-100%)│
└───────┘ └───────┘  └───────┘  └───────┘
```

#### Implementation

**Router Service:**
```python
def search_sharded(query, k, shard_urls):
    results = []
    
    # Fan out to all shards
    for shard_url in shard_urls:
        resp = requests.post(f"{shard_url}/search", json={
            "query": query,
            "k": k * 2  # Get more results for merging
        })
        results.extend(resp.json()["text_results"])
    
    # Merge by score, return top-k
    return sorted(results, key=lambda x: x["score"], reverse=True)[:k]
```

**Sharding Strategy:**
- Hash `doc_id` → `shard_id = hash(doc_id) % num_shards`
- All chunks of same document go to same shard (good for locality)
- Even distribution across shards

**Benefits:**
- Horizontal scaling
- Each shard handles subset of data
- Can add more shards as data grows

### Best Practices for Scaling

1. **Monitor Performance**
   - Track search latency
   - Monitor memory usage
   - Check recall (compare results to exact search)

2. **Upgrade Gradually**
   - Start with IndexFlatIP (exact search)
   - Upgrade to IVF when latency > 10ms
   - Use PQ if memory constrained
   - Use HNSW for very large datasets

3. **Tune Parameters**
   - **nprobe** (IVF): Higher = better recall, slower (start with 64)
   - **nlist** (IVF): sqrt(num_vectors) or 4096
   - **M** (HNSW): 32-64 (higher = better recall, more memory)

4. **Batch Operations**
   - Ingest in batches (1000-10000 vectors)

---

## Image Description Feature (Ollama)

### How It Works

The system uses **Ollama** to generate detailed text descriptions of images, then creates text embeddings from those descriptions. This enables **dual search capabilities**:

1. **Image-to-Image Search**: Using CLIP embeddings (image vectors) - stored in `image_index`
2. **Text-to-Image Search**: Using text embeddings of image descriptions - stored in `text_index`

### What Gets Stored

For each image (regular images and PDF pages):
- ✅ **Image Embedding (CLIP)**: Stored in `image_index` - for image-to-image similarity search
- ✅ **Description Text Embeddings**: Stored in `text_index` - for text queries to find images

**Result:** When you search with text, you'll get:
- Regular text documents
- **Images (matched via their descriptions)** ← This is the key feature!

### Setup

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.com
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Pull a Vision Model:**
   ```bash
   ollama pull llava
   # or
   ollama pull gemma3:4b
   ```

4. **Set Model (Optional):**
   ```bash
   export OLLAMA_VISION_MODEL="llava"  # or gemma3:4b
   ```

### Example Workflow

1. **Image**: `photo.png` shows "a red car on a highway"
2. **Ollama describes it**: "A red sedan car driving on a highway with blue sky in the background. The car appears to be moving at high speed..."
3. **Description is chunked** (if long) and embedded as text
4. **Both stored**:
   - CLIP embedding → `image_index` (for image similarity)
   - Description text embeddings → `text_index` (for text search)
5. **Search for "red car"** → finds the image via description embeddings!

### How It's Implemented

**For Regular Images:**
- Image loaded → Ollama describes → Description chunked → Text embeddings created
- **Both** CLIP image embeddings AND text embeddings stored

**For PDF Pages:**
- PDF pages converted to images → Ollama describes each page → Descriptions chunked → Text embeddings created
- **Both** CLIP image embeddings (for pages) AND text embeddings (for descriptions) stored

### Benefits

- **Text queries find images**: Search "red car" and get images of red cars
- **Dual search modes**: Image-to-image (CLIP) + text-to-image (descriptions)
- **No breaking changes**: Existing functionality preserved
- **Automatic**: Works when Ollama is available, gracefully degrades if not

---

## API Endpoints

### 1. Ingest Documents

**POST** `/ingest`

Ingest documents from a folder (handles embedding automatically).

**Note:** If Ollama is available, images will be described and their descriptions will be embedded as text, enabling text-to-image search.

```json
{
  "docs_path": "../docs",
  "chunk_size": 800,
  "overlap": 120,
  "extract_images": true,
  "namespace": "my_project"
}
```

**Note:** PDFs are automatically processed with **BOTH** text and image embeddings:
- **Text extraction** → chunking → text embeddings
- **Page images** → CLIP embeddings  
- **Image descriptions** (Ollama) → text embeddings (if Ollama available)

This gives you the best search results - text queries find text content, and image queries find visual content!

### 2. Search

**POST** `/search`

Search for similar documents.

```json
{
  "query": "machine learning",
  "k": 5,
  "namespace": "my_project",
  "vector_type": "both"
}
```

### 3. Delete

**POST** `/delete`

Delete by document ID or chunk ID.

```json
{
  "doc_id": "doc1.txt",
  "namespace": "my_project"
}
```

### 4. Get Statistics

**GET** `/stats`

Returns index statistics.

### 5. Reset (Clear All Data)

**POST** `/reset`

⚠️ **Warning:** Permanently deletes ALL data and resets the database!

Use this to completely start fresh.

---

## Architecture

### Components

1. **VectorStoreV2** (`src/store_v2.py`)
   - Manages FAISS indices
   - Handles SQLite metadata
   - Provides stable IDs

2. **Flask API** (`app_v2.py`)
   - REST API endpoints
   - Handles requests
   - Orchestrates operations

3. **Embedding Pipeline**
   - Loads documents
   - Chunks text
   - Generates embeddings
   - Stores in VectorStoreV2

### Data Flow

```
Documents → Load → Chunk → Embed → Store
                                    ↓
                            FAISS Index (vectors)
                            SQLite DB (metadata)
                                    ↓
                            Search → Results
```

### Key Features

- **Stable IDs**: Each vector has persistent int64 ID
- **SQLite Metadata**: Not aligned by position, survives deletions
- **Atomic Writes**: Crash-safe persistence
- **Namespace Isolation**: Multi-tenancy support

---

## Summary

### Storage
- **FAISS indices**: Binary files in `data/indices/`
- **SQLite database**: Metadata in `data/metadata/metadata.db`
- **Atomic writes**: Crash-safe using temp files + rename
- **Manifest**: Tracks current state

### Deletion
- **Soft delete**: Mark as deleted, filter in search
- **Hard delete**: Use `remove_ids()` to remove from index

### Scaling
- **< 100K**: IndexFlatIP (exact search)
- **100K-1M**: IndexIVFFlat (98-99% recall, 10x faster)
- **1M-10M**: IndexIVFPQ (95-98% recall, 4x memory reduction)
- **10M+**: IndexHNSW (95-97% recall, consistent latency)
- **Multi-machine**: Shard by hash(doc_id)

---

For Postman examples, see `POSTMAN_COLLECTION.json` (import into Postman).
