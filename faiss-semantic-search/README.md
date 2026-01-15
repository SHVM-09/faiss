# FAISS Semantic Search API

A simple, beginner-friendly semantic search API using FAISS (Facebook AI Similarity Search). Search through text documents and images using natural language queries.

**Features:**
- 🔍 Semantic search for text documents (`.txt`, `.md`)
- 📄 **PDF support with flexible processing modes** (text extraction or image conversion)
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
- **PDF files**: `.pdf` (with text extraction or image conversion)
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
  "extract_images": true,
  "pdfType": "with_photo"
}
```

**Parameters:**
- `docs_path` (string, required): Path to folder containing documents
- `chunk_size` (int, optional): Max characters per chunk (default: 800)
- `overlap` (int, optional): Characters to overlap between chunks (default: 120)
- `extract_images` (bool, optional): Load and index image files (default: true)
- `pdfType` (string, optional): PDF processing mode - `"plain"` (text only) or `"with_photo"` (images only) (default: `"with_photo"`)

**Response:**
```json
{
  "success": true,
  "message": "Documents ingested successfully",
  "stats": {
    "files_loaded": 5,
    "text_files_loaded": 2,
    "image_files_loaded": 3,
    "pdf_files_loaded": 1,
    "chunks_created": 3,
    "text_vectors_stored": 3,
    "images_extracted": 3,
    "image_vectors_stored": 3,
    "pdf_pages_processed": 0,
    "pdf_image_vectors_stored": 0,
    "pdf_text_chunks_created": 513,
    "pdf_text_vectors_stored": 513
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

### 3. Ingest PDF

Process a single PDF file with flexible processing modes.

**Request:**
```bash
POST http://localhost:5001/ingest_pdf
Content-Type: application/json
```

**Body:**
```json
{
  "pdf_path": "./docs/document.pdf",
  "pdfType": "plain",
  "chunk_size": 800,
  "overlap": 120,
  "dpi": 200
}
```

**Parameters:**
- `pdf_path` (string, required): Path to PDF file
- `pdfType` (string, optional): Processing mode - `"plain"` (text only) or `"with_photo"` (images only) (default: `"with_photo"`)
  - `"plain"`: Extract text only → chunk → embed (text embeddings only, NO images)
  - `"with_photo"`: Convert pages to images → embed images (image embeddings only, NO text)
- `chunk_size` (int, optional): Max characters per chunk (default: 800, only used if `pdfType="plain"`)
- `overlap` (int, optional): Characters to overlap between chunks (default: 120, only used if `pdfType="plain"`)
- `dpi` (int, optional): DPI for image conversion (default: 200, only used if `pdfType="with_photo"`)

**Response:**
```json
{
  "success": true,
  "message": "PDF processed and stored successfully",
  "stats": {
    "pdf_file": "document.pdf",
    "pdf_type": "plain",
    "text_chunks_created": 513,
    "text_vectors_stored": 513,
    "pages_processed": 0,
    "image_vectors_stored": 0
  }
}
```

**cURL:**
```bash
# Plain mode (text only)
curl -X POST http://localhost:5001/ingest_pdf \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "./docs/document.pdf",
    "pdfType": "plain",
    "chunk_size": 800,
    "overlap": 120
  }'

# With photo mode (images only)
curl -X POST http://localhost:5001/ingest_pdf \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "./docs/document.pdf",
    "pdfType": "with_photo",
    "dpi": 200
  }'
```

---

### 4. Search

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

### 5. Get Stats

Get statistics about the indexed data.

**Request:**
```bash
GET http://localhost:5001/stats
```

**Response:**
```json
{
  "text_vector_count": 515,
  "text_dimension": 1024,
  "image_vector_count": 2,
  "image_dimension": 1024,
  "text_model": "BAAI/bge-large-en-v1.5",
  "image_model": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
}
```

**cURL:**
```bash
curl http://localhost:5001/stats
```

---

### 6. Clear Index

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

### Example 3: PDF Processing

```bash
# Process PDF with text extraction only
curl -X POST http://localhost:5001/ingest_pdf \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "./docs/report.pdf",
    "pdfType": "plain",
    "chunk_size": 800,
    "overlap": 120
  }'

# Process PDF with image conversion only
curl -X POST http://localhost:5001/ingest_pdf \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "./docs/presentation.pdf",
    "pdfType": "with_photo",
    "dpi": 200
  }'

# Search PDF content
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "amazon revenue", "k": 5}'
```

### Example 4: Python Client

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
    ├── embed.py         # STEP 3: Convert text to vectors (1024-dim)
    ├── image_embed.py   # STEP 3: Convert images to vectors (1024-dim, CLIP)
    ├── pdf_process.py   # PDF processing: text extraction and image conversion
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
PROCESS PDFs (extract text OR convert to images based on pdfType)
  ↓
TRANSFORM (split text into chunks with overlap)
  ↓
EMBED TEXT (convert chunks to 1024-dim vectors using BAAI/bge-large-en-v1.5)
EMBED IMAGES (convert images to 1024-dim vectors using CLIP)
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
   - **Text**: Converts chunks to 1024-dimensional vectors using `BAAI/bge-large-en-v1.5`
   - **Images**: Converts images to 1024-dimensional vectors using CLIP (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)
   - Vectors are normalized for cosine similarity

4. **PDF PROCESSING** (`src/pdf_process.py`)
   - **Plain mode**: Extracts text from PDF using PyMuPDF → chunks → embeds with text embedder
   - **With photo mode**: Converts PDF pages to images using pdf2image → embeds with CLIP
   - Supports large PDFs with progress tracking

5. **STORE** (`src/store.py`)
   - Saves text vectors to `text_index.faiss`
   - Saves image vectors to `image_index.faiss`
   - Stores metadata (text, image paths) in JSON files
   - Indices persist to disk (survive server restarts)

6. **RETRIEVE** (`src/retrieve.py`)
   - Converts query to vectors (text + CLIP)
   - Searches both indices using FAISS
   - Returns most similar chunks and images
   - Automatically deduplicates results

### Why Separate Indices?

- Text vectors: 1024 dimensions (BAAI/bge-large-en-v1.5)
- Image vectors: 1024 dimensions (CLIP-ViT-H-14)
- FAISS requires consistent dimensions, so we use separate indices
- Both currently use 1024 dimensions but kept separate for flexibility

### PDF Processing Modes

The API supports two PDF processing modes:

1. **Plain Mode** (`pdfType: "plain"`):
   - Extracts text from PDF using PyMuPDF
   - Chunks the extracted text
   - Embeds chunks using text embedder (BAAI/bge-large-en-v1.5)
   - Stores in text index
   - **Use case**: Text-based PDFs, documents, reports

2. **With Photo Mode** (`pdfType: "with_photo"`):
   - Converts PDF pages to images using pdf2image (requires poppler)
   - Embeds page images using CLIP
   - Stores in image index
   - **Use case**: Scanned PDFs, presentations with images, visual documents

**Note**: You can process the same PDF with both modes separately if you need both text and image search.

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

### PDF Processing Issues

**Problem:** PDF processing not working

**Solution:**
```bash
# Install required dependencies
pip install pymupdf pdf2image

# Install poppler (required for pdf2image)
# macOS:
brew install poppler

# Linux:
sudo apt-get install poppler-utils
# or
sudo yum install poppler-utils

# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
```

**Problem:** PDF text extraction is slow

**Solution:**
- Large PDFs (>100K characters) show progress during chunking
- The process is optimized for large documents
- Consider using `"with_photo"` mode for scanned PDFs instead

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

1. **Text Embedding:** `BAAI/bge-large-en-v1.5` (1024 dimensions)
   - State-of-the-art accuracy for English text
   - Best accuracy on benchmarks
   - ~1.3GB model size
   - Used for: text files (.txt, .md) and PDF text extraction

2. **Image Embedding:** `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (1024 dimensions)
   - High-accuracy CLIP model for multimodal search
   - Excellent for image-text search
   - ~2.5GB model size
   - Used for: image files and PDF page images

**Want different models?** See [MODELS.md](MODELS.md) for alternatives and how to change models.

---

## License

MIT

---

## Support

For issues or questions, check the code comments in each file - they explain the flow and logic in detail.

For model alternatives and upgrades, see [MODELS.md](MODELS.md).
