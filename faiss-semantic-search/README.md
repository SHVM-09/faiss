# FAISS Semantic Search API

A simple, beginner-friendly semantic search API using FAISS (Facebook AI Similarity Search). Search through text documents and images using natural language queries.

**Features:**
- 🔍 Semantic search for text documents (`.txt`, `.md`)
- 🖼️ Image search using text queries (CLIP model)
- 📊 Fast similarity search with FAISS
- 💾 Persistent storage (indices saved to disk)
- 🚀 Simple REST API

**Flow:** `source -> load -> transform -> embed -> store -> retrieve`

---

## Table of Contents

- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n faiss-search python=3.11 -y
conda activate faiss-search

# Install FAISS (required first - must use conda)
conda install -c conda-forge faiss-cpu -y

# Install Python packages
cd faiss-semantic-search
pip install -r requirements.txt
```

**Note:** FAISS must be installed via conda on macOS. The `requirements.txt` does not include FAISS.

### 2. Add Documents

Create a `docs/` folder and add your files:

```bash
mkdir docs
# Add your files here
```

**Supported file types:**
- **Text files**: `.txt`, `.md`
- **Image files**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.tiff`, `.tif`

### 3. Run the API

```bash
python app.py
```

The server will start on `http://localhost:5001`

---

## API Endpoints

### 1. Health Check

Check if the API is running.

**Request:**
```bash
GET http://localhost:5001/health
```

**Response:**
```json
{
  "ok": true
}
```

**cURL:**
```bash
curl http://localhost:5001/health
```

---

### 2. Ingest Documents

Load and index documents and images from a folder.

**Request:**
```bash
POST http://localhost:5001/ingest
Content-Type: application/json
```

**Body:**
```json
{
  "docs_path": "./docs",
  "chunk_size": 800,
  "overlap": 120,
  "extract_images": true
}
```

**Parameters:**
- `docs_path` (string, required): Path to folder containing documents
- `chunk_size` (int, optional): Max characters per chunk (default: 800)
- `overlap` (int, optional): Characters to overlap between chunks (default: 120)
- `extract_images` (bool, optional): Load and index image files (default: true)

**Response:**
```json
{
  "success": true,
  "message": "Documents ingested successfully",
  "stats": {
    "files_loaded": 5,
    "text_files_loaded": 2,
    "image_files_loaded": 3,
    "chunks_created": 3,
    "text_vectors_stored": 3,
    "images_extracted": 3,
    "image_vectors_stored": 3
  }
}
```

**cURL:**
```bash
curl -X POST http://localhost:5001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "docs_path": "./docs",
    "chunk_size": 800,
    "overlap": 120,
    "extract_images": true
  }'
```

**Postman:**
1. Method: `POST`
2. URL: `http://localhost:5001/ingest`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "docs_path": "./docs",
  "chunk_size": 800,
  "overlap": 120,
  "extract_images": true
}
```

---

### 3. Search

Search for documents and images similar to your query.

**Request:**
```bash
POST http://localhost:5001/search
Content-Type: application/json
```

**Body:**
```json
{
  "query": "machine learning",
  "k": 5
}
```

**Parameters:**
- `query` (string, required): Your search query
- `k` (int, optional): Number of results to return (default: 5)

**Response:**
```json
{
  "query": "machine learning",
  "k": 5,
  "text_results": [
    {
      "score": 0.85,
      "source": "sample1.txt",
      "chunk_id": "sample1.txt::chunk_0",
      "chunk_text": "Machine learning is a subset of artificial intelligence...",
      "preview": "Machine learning is a subset of artificial intelligence...",
      "type": "text"
    }
  ],
  "image_results": [
    {
      "score": 0.78,
      "source": "diagram.png",
      "image_id": "diagram.png::img_0",
      "image_index": 0,
      "image_path": "./data/images/diagram.png",
      "type": "image"
    }
  ],
  "total_text_results": 3,
  "total_image_results": 2
}
```

**cURL:**
```bash
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 5
  }'
```

**Postman:**
1. Method: `POST`
2. URL: `http://localhost:5001/search`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "query": "machine learning",
  "k": 5
}
```

---

### 4. Get Stats

Get statistics about the indexed data.

**Request:**
```bash
GET http://localhost:5001/stats
```

**Response:**
```json
{
  "text_vector_count": 10,
  "text_dimension": 384,
  "image_vector_count": 5,
  "image_dimension": 512,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**cURL:**
```bash
curl http://localhost:5001/stats
```

---

### 5. Clear Index

Clear all indexed data and start fresh.

**Request:**
```bash
POST http://localhost:5001/clear
```

**Response:**
```json
{
  "success": true,
  "message": "Index cleared successfully"
}
```

**cURL:**
```bash
curl -X POST http://localhost:5001/clear
```

---

## Usage Examples

### Example 1: Basic Text Search

```bash
# 1. Ingest documents
curl -X POST http://localhost:5001/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "./docs"}'

# 2. Search
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "k": 3}'
```

### Example 2: Image Search

```bash
# 1. Ingest with images
curl -X POST http://localhost:5001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "docs_path": "./docs",
    "extract_images": true
  }'

# 2. Search for images
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "logo", "k": 5}'
```

### Example 3: Python Client

```python
import requests

BASE_URL = "http://localhost:5001"

# Ingest documents
response = requests.post(f"{BASE_URL}/ingest", json={
    "docs_path": "./docs",
    "chunk_size": 800,
    "overlap": 120,
    "extract_images": True
})
print(response.json())

# Search
response = requests.post(f"{BASE_URL}/search", json={
    "query": "machine learning",
    "k": 5
})
results = response.json()
print(f"Found {len(results['text_results'])} text results")
print(f"Found {len(results['image_results'])} image results")
```

---

## Project Structure

```
faiss-semantic-search/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── docs/                 # Your documents go here
│   ├── sample1.txt
│   ├── sample2.md
│   └── images/
├── data/                 # Auto-created (indices and metadata)
│   ├── text_index.faiss
│   ├── text_metadata.json
│   ├── image_index.faiss
│   ├── image_metadata.json
│   └── images/          # Saved copies of indexed images
└── src/
    ├── __init__.py
    ├── load.py          # STEP 1: Load documents and images
    ├── transform.py     # STEP 2: Split text into chunks
    ├── embed.py         # STEP 3: Convert text to vectors (384-dim)
    ├── image_embed.py   # STEP 3: Convert images to vectors (512-dim, CLIP)
    ├── store.py         # STEP 4: Save to FAISS indices
    ├── retrieve.py      # STEP 5: Search and retrieve results
    └── pipeline.py      # Complete pipeline orchestration
```

---

## How It Works

### Pipeline Flow

```
SOURCE (files on disk)
  ↓
LOAD (read .txt, .md files and image files)
  ↓
TRANSFORM (split text into chunks with overlap)
  ↓
EMBED TEXT (convert chunks to 384-dim vectors using sentence-transformers)
EMBED IMAGES (convert images to 512-dim vectors using CLIP)
  ↓
STORE (save to separate FAISS indices)
  ↓
RETRIEVE (search both indices, return results)
```

### Step-by-Step

1. **LOAD** (`src/load.py`)
   - Reads `.txt` and `.md` files
   - Loads image files (`.png`, `.jpg`, etc.)
   - Returns documents and images

2. **TRANSFORM** (`src/transform.py`)
   - Splits large text documents into smaller chunks
   - Adds overlap between chunks to preserve context
   - Example: 2000-char document → 3 chunks of ~800 chars each

3. **EMBED** (`src/embed.py` + `src/image_embed.py`)
   - **Text**: Converts chunks to 384-dimensional vectors using `sentence-transformers/all-MiniLM-L6-v2`
   - **Images**: Converts images to 512-dimensional vectors using CLIP (`openai/clip-vit-base-patch32`)
   - Vectors are normalized for cosine similarity

4. **STORE** (`src/store.py`)
   - Saves text vectors to `text_index.faiss`
   - Saves image vectors to `image_index.faiss`
   - Stores metadata (text, image paths) in JSON files
   - Indices persist to disk (survive server restarts)

5. **RETRIEVE** (`src/retrieve.py`)
   - Converts query to vectors (text + CLIP)
   - Searches both indices using FAISS
   - Returns most similar chunks and images
   - Automatically deduplicates results

### Why Separate Indices?

- Text vectors: 384 dimensions (sentence-transformers)
- Image vectors: 512 dimensions (CLIP)
- FAISS requires consistent dimensions, so we use separate indices

---

## Troubleshooting

### FAISS Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'faiss'`

**Solution:**
```bash
# FAISS must be installed via conda (not pip)
conda install -c conda-forge faiss-cpu -y
```

### Port Already in Use

**Problem:** `Address already in use`

**Solution:** Change port in `app.py`:
```python
app.run(host='0.0.0.0', port=5002, debug=True)  # Change 5001 to 5002
```

### Images Not Loading

**Problem:** Images not being indexed

**Solution:**
```bash
# Install Pillow
pip install Pillow

# Or use conda
conda install pillow -y
```

### Duplicate Results

**Problem:** Same chunk/image appears multiple times in search results

**Solution:** 
- The deduplication logic should handle this automatically
- If you see duplicates, clear the index and re-ingest:
```bash
curl -X POST http://localhost:5001/clear
curl -X POST http://localhost:5001/ingest -H "Content-Type: application/json" -d '{"docs_path": "./docs"}'
```

### Model Download Issues

**Problem:** Models not downloading (first time use)

**Solution:**
- Models are downloaded automatically on first use
- Text model: ~80MB (`sentence-transformers/all-MiniLM-L6-v2`)
- Image model: ~500MB (`openai/clip-vit-base-patch32`)
- Ensure you have internet connection and disk space

---

## Model Information

The API uses two models:

1. **Text Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
   - Fast and efficient
   - Good for general semantic search
   - ~80MB model size

2. **Image Embedding:** `openai/clip-vit-base-patch32` (512 dimensions)
   - Industry standard for image-text search
   - Good balance of performance and speed
   - ~500MB model size

**Want better accuracy?** See [MODELS.md](MODELS.md) for better alternatives and how to upgrade.

---

## License

MIT

---

## Support

For issues or questions, check the code comments in each file - they explain the flow and logic in detail.

For model alternatives and upgrades, see [MODELS.md](MODELS.md).
