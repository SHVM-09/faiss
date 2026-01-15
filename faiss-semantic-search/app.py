"""
FAISS Semantic Search API
==========================
A simple Flask API for semantic document search.

Flow: source -> load -> transform -> embed -> store -> retrieve

This is the main application file that:
1. Sets up Flask web server
2. Defines API endpoints
3. Connects the pipeline steps together
"""

import os
import sys
import warnings

# Disable multiprocessing parallelism to avoid semaphore leaks
# This prevents resource_tracker warnings from sentence-transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Limit NumExpr threads

# Suppress multiprocessing warnings more aggressively
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*semaphore.*")

# Monkey-patch warnings.warn to suppress multiprocessing resource_tracker warnings
_original_warn = warnings.warn
def _suppress_multiprocessing_warnings(message, *args, **kwargs):
    if "resource_tracker" in str(message) or "semaphore" in str(message):
        return  # Suppress these warnings
    return _original_warn(message, *args, **kwargs)
warnings.warn = _suppress_multiprocessing_warnings

from flask import Flask, request, jsonify
from src.pipeline import run_pipeline
from src.retrieve import retrieve_similar
from src.embed import Embedder
from src.store import VectorStore

# Lazy import for image embedder (only needed if using image search)
try:
    from src.image_embed import ImageEmbedder
except ImportError:
    ImageEmbedder = None

# Lazy import for PDF processing
try:
    from src.pdf_process import process_pdf, store_pdf_data
except ImportError:
    process_pdf = None
    store_pdf_data = None

# Create Flask application
app = Flask(__name__)

# Get the directory where this file is located
# This helps us resolve relative paths correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global instances (singletons - created once, reused)
# These are loaded once and reused for all requests
embedder = None
image_embedder = None
store = None


def get_embedder():
    """
    Get or create the text embedder instance.
    
    Singleton pattern: Only create once, reuse for all requests.
    This avoids reloading the model on every request (which is slow).
    """
    global embedder
    if embedder is None:
        embedder = Embedder()
        embedder.load()  # Load the embedding model
    return embedder


def get_image_embedder():
    """
    Get or create the image embedder instance (CLIP).
    
    Singleton pattern: Only create once, reuse for all requests.
    CLIP model is larger, so loading takes time - we cache it.
    """
    global image_embedder
    if ImageEmbedder is None:
        raise ImportError("Image embedding not available. Install: pip install transformers Pillow")
    if image_embedder is None:
        image_embedder = ImageEmbedder()
        image_embedder.load()  # Load the CLIP model
    return image_embedder


def get_store():
    """
    Get or create the vector store instance.
    
    Singleton pattern: Only create once, reuse for all requests.
    Tries to load existing index from disk if available.
    """
    global store
    if store is None:
        store = VectorStore()
        store.load()  # Try to load existing index from disk
    return store


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint - API information.
    
    Returns information about available endpoints and the flow.
    """
    return jsonify({
        "message": "FAISS Semantic Search API",
        "version": "1.0.0",
        "description": "Semantic search API for text documents and images using FAISS",
        "flow": "source -> load -> transform -> embed -> store -> retrieve",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /ingest": "Ingest documents and images (load -> transform -> embed -> store)",
            "POST /ingest_pdf": "Ingest PDF files (extract text/images -> chunk -> embed -> store)",
            "POST /search": "Search documents and images (retrieve)",
            "GET /stats": "Get statistics about indexed data",
            "POST /clear": "Clear all indexed data"
        },
        "supported_files": {
            "text": [".txt", ".md"],
            "images": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"],
            "pdfs": [".pdf"]
        },
        "features": {
            "text_search": "Semantic search for text content from .txt and .md files",
            "image_search": "Search image files using text queries (CLIP model)",
            "multi_modal": "Single query searches both text and images simultaneously",
            "deduplication": "Automatically removes duplicate results"
        },
        "documentation": "See README.md for detailed usage examples"
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Simple endpoint to verify the API is running.
    """
    return jsonify({"ok": True})


@app.route('/ingest', methods=['POST'])
def ingest():
    """
    Ingest documents endpoint.
    
    This runs the complete pipeline:
    SOURCE -> LOAD -> TRANSFORM -> EMBED -> STORE
    
    Request body (JSON):
    {
        "docs_path": "./docs",      # Path to folder with documents and images
        "chunk_size": 800,          # Max characters per chunk
        "overlap": 120,             # Characters to overlap between chunks
        "extract_images": true,     # Load and index image files (default: true)
        "pdfType": "with_photo"     # PDF processing mode: "plain" (text only) or "with_photo" (images only) (default: "with_photo")
    }
    
    Supported files:
    - Text: .txt, .md
    - Images: .png, .jpg, .jpeg, .gif, .bmp, .webp, .tiff, .tif
    
    Returns:
    {
        "success": true,
        "message": "Documents ingested successfully",
        "stats": {
            "files_loaded": 3,
            "text_files_loaded": 2,
            "image_files_loaded": 1,
            "chunks_created": 15,
            "text_vectors_stored": 15,
            "images_extracted": 1,
            "image_vectors_stored": 1,
            "text_dimension": 1024,
            "image_dimension": 1024,
            "text_model": "BAAI/bge-large-en-v1.5",
            "image_model": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        }
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        # Extract parameters with defaults
        docs_path = data.get("docs_path", "./docs")
        chunk_size = data.get("chunk_size", 800)
        overlap = data.get("overlap", 120)
        pdf_type = data.get("pdfType", "with_photo")  # Default: "with_photo"
        
        # Convert relative path to absolute path
        # This ensures paths work regardless of where the script is run from
        if not os.path.isabs(docs_path):
            docs_path = os.path.join(BASE_DIR, docs_path)
        
        # Validate input parameters
        if chunk_size <= 0:
            return jsonify({"error": "chunk_size must be positive"}), 400
        if overlap < 0:
            return jsonify({"error": "overlap must be non-negative"}), 400
        if overlap >= chunk_size:
            return jsonify({"error": "overlap must be less than chunk_size"}), 400
        if pdf_type not in ["plain", "with_photo"]:
            return jsonify({"error": "pdfType must be 'plain' or 'with_photo'"}), 400
        
        # Check if user wants to extract images
        extract_images = data.get("extract_images", True)  # Default: True
        
        # Run the complete pipeline
        # This does: LOAD -> TRANSFORM -> EMBED TEXT -> EMBED IMAGES -> PROCESS PDFs -> STORE
        stats = run_pipeline(docs_path, chunk_size, overlap, extract_images=extract_images, pdf_type=pdf_type)
        
        # Reset global store so it reloads the updated indices
        global store
        store = None
        
        # Return success response with statistics
        return jsonify({
            "success": True,
            "message": "Documents ingested successfully",
            "stats": stats
        }), 200
    
    except ValueError as e:
        # User input error (e.g., folder not found)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Unexpected error
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/ingest_pdf', methods=['POST'])
def ingest_pdf():
    """
    Ingest PDF endpoint.
    
    This processes PDF files with two modes:
    1. "plain": Extract text ONLY → chunk it → embed it (text embeddings only, NO images)
    2. "with_photo": Convert pages to images ONLY → embed images (image embeddings only, NO text)
    
    Request body (JSON):
    {
        "pdf_path": "./docs/document.pdf",  # Path to PDF file
        "pdfType": "with_photo",            # "plain" (text only) or "with_photo" (images only) (default: "with_photo")
        "chunk_size": 800,                  # Optional: Max characters per chunk (default: 800, only used if pdfType="plain")
        "overlap": 120,                     # Optional: Characters to overlap between chunks (default: 120, only used if pdfType="plain")
        "dpi": 200                          # Optional: DPI for image conversion (default: 200, only used if pdfType="with_photo")
    }
    
    Returns:
    {
        "success": true,
        "message": "PDF processed successfully",
        "stats": {
            "pdf_file": "document.pdf",
            "pdf_type": "with_photo",
            "text_chunks_created": 15,      # Only if pdfType="plain"
            "text_vectors_stored": 15,      # Only if pdfType="plain"
            "pages_processed": 8,          # Only if pdfType="with_photo"
            "image_vectors_stored": 8      # Only if pdfType="with_photo"
        }
    }
    """
    try:
        global store  # Declare global at the start of the function
        
        # Check if PDF processing is available
        if process_pdf is None or store_pdf_data is None:
            return jsonify({
                "error": "PDF processing not available. Install: pip install pymupdf transformers Pillow"
            }), 400
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        # Extract parameters
        pdf_path = data.get("pdf_path", "")
        pdf_type = data.get("pdfType", "with_photo")  # Default: "with_photo"
        chunk_size = data.get("chunk_size", 800)
        overlap = data.get("overlap", 120)
        dpi = data.get("dpi", 200)  # Default DPI for image conversion
        
        # Validate input
        if not pdf_path:
            return jsonify({"error": "pdf_path is required"}), 400
        if pdf_type not in ["plain", "with_photo"]:
            return jsonify({"error": "pdfType must be 'plain' or 'with_photo'"}), 400
        if chunk_size <= 0:
            return jsonify({"error": "chunk_size must be positive"}), 400
        if overlap < 0:
            return jsonify({"error": "overlap must be non-negative"}), 400
        if overlap >= chunk_size:
            return jsonify({"error": "overlap must be less than chunk_size"}), 400
        if dpi <= 0:
            return jsonify({"error": "dpi must be positive"}), 400
        
        # Convert relative path to absolute path
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(BASE_DIR, pdf_path)
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            return jsonify({"error": f"PDF file not found: {pdf_path}"}), 400
        
        # Check if it's a PDF file
        if not pdf_path.lower().endswith('.pdf'):
            return jsonify({"error": "File must be a PDF (.pdf extension)"}), 400
        
        # Process PDF based on pdf_type
        pdf_data = process_pdf(pdf_path, dpi=dpi, pdf_type=pdf_type)
        
        # Get store instance
        store = get_store()
        
        # Process text ONLY if pdf_type is "plain"
        text_chunks = None
        text_vectors = None
        if pdf_type == "plain" and pdf_data.get("text_content"):
            from src.transform import transform_to_chunks
            from src.embed import Embedder
            
            # Create a document dict for chunking
            pdf_filename = os.path.basename(pdf_path)
            document = {
                "filename": pdf_filename,
                "filepath": pdf_path,
                "content": pdf_data["text_content"]
            }
            
            # Chunk the text
            text_len = len(document["content"])
            if text_len > 100000:
                print(f"Chunking PDF text ({text_len:,} characters, this may take a moment)...")
            else:
                print("Chunking PDF text...")
            text_chunks = transform_to_chunks([document], chunk_size=chunk_size, overlap=overlap)
            print(f"  ✓ Created {len(text_chunks)} text chunks")
            
            # Embed the chunks
            if text_chunks:
                num_chunks = len(text_chunks)
                print(f"Embedding PDF text chunks ({num_chunks} chunks)...")
                # Reuse global embedder if available, otherwise create new one
                try:
                    embedder = get_embedder()
                except:
                    embedder = Embedder()
                    embedder.load()
                
                chunk_texts = [chunk["chunk_text"] for chunk in text_chunks]
                # Use larger batch size for PDF chunks to speed up processing
                text_vectors = embedder.embed(chunk_texts, batch_size=128)
                print(f"  ✓ Generated {len(text_vectors)} text embeddings")
        
        # Store PDF data in indices
        # For "plain": store text chunks only
        # For "with_photo": store image vectors only
        pdf_filename = os.path.basename(pdf_path)
        store_pdf_data(store, pdf_data, pdf_filename, text_chunks=text_chunks, text_vectors=text_vectors)
        
        # Save to disk
        store.save()
        
        # Reset global store so it reloads the updated indices
        store = None
        
        # Return success response with statistics
        return jsonify({
            "success": True,
            "message": "PDF processed and stored successfully",
            "stats": {
                "pdf_file": pdf_filename,
                "pdf_type": pdf_type,
                "text_chunks_created": len(text_chunks) if text_chunks else 0,
                "text_vectors_stored": len(text_vectors) if text_vectors is not None else 0,
                "pages_processed": len(pdf_data.get("pages", [])),
                "image_vectors_stored": len(pdf_data["image_vectors"]) if pdf_data.get("image_vectors") is not None else 0
            }
        }), 200
    
    except ValueError as e:
        # User input error (e.g., file not found)
        return jsonify({"error": str(e)}), 400
    except ImportError as e:
        # Missing dependencies
        return jsonify({"error": f"Dependency error: {str(e)}"}), 500
    except Exception as e:
        # Unexpected error
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Search endpoint.
    
    This runs the RETRIEVE step to find similar documents.
    
    Request body (JSON):
    {
        "query": "machine learning",  # Search query
        "k": 5                        # Number of results to return
    }
    
    Returns:
    {
        "query": "machine learning",
        "k": 5,
        "results": [
            {
                "score": 0.85,
                "source": "doc1.txt",
                "chunk_id": "doc1.txt::chunk_0",
                "chunk_text": "...",
                "preview": "..."
            },
            ...
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        # Extract parameters
        query_text = data.get("query", "").strip()
        k = data.get("k", 5)
        
        # Validate input
        if not query_text:
            return jsonify({"error": "query cannot be empty"}), 400
        if k <= 0:
            return jsonify({"error": "k must be positive"}), 400
        
        # Get embedder and store instances
        embedder = get_embedder()
        store = get_store()
        
        # Try to get image embedder (optional - image search may not be available)
        image_embedder = None
        try:
            image_embedder = get_image_embedder()
        except (ImportError, AttributeError):
            # Image search not available, but text search will still work
            pass
        
        # Try to get PDF embedder (for PDF text/image search)
        pdf_embedder = None
        try:
            if process_pdf is not None:
                from src.pdf_process import PDFMultimodalEmbedder
                pdf_embedder = PDFMultimodalEmbedder()
                pdf_embedder.load()
        except (ImportError, AttributeError):
            # PDF search not available, but regular search will still work
            pass
        
        # Check if any index exists
        has_text = store.text_index is not None and store.text_index.ntotal > 0
        has_images = store.image_index is not None and store.image_index.ntotal > 0
        
        if not has_text and not has_images:
            return jsonify({"error": "Index is empty. Please ingest documents first."}), 400
        
        # Search for similar documents and images
        # This does: QUERY -> EMBED -> SEARCH (both indices) -> RETRIEVE
        results = retrieve_similar(query_text, k, embedder, image_embedder, store, pdf_embedder)
        
        # Return results
        return jsonify({
            "query": query_text,
            "k": k,
            "text_results": results["text_results"][:k],  # Limit to k results
            "image_results": results["image_results"][:k],  # Limit to k results
            "total_text_results": len(results["text_results"]),
            "total_image_results": len(results["image_results"])
        }), 200
    
    except ValueError as e:
        # User input error
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Unexpected error
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """
    Get statistics endpoint.
    
    Returns information about the current index:
    - vector_count: Number of vectors in index
    - dimension: Size of each vector (1024 for text, 1024 for images)
    """
    try:
        store = get_store()
        stats_dict = store.get_stats()
        return jsonify(stats_dict), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/clear', methods=['POST'])
def clear():
    """
    Clear index endpoint.
    
    Deletes all indexed documents and clears the index.
    Use this to start fresh.
    """
    try:
        global store
        store = get_store()
        store.clear()  # Clear index and delete files
        store = None   # Reset singleton
        
        return jsonify({
            "success": True,
            "message": "Index cleared successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Print startup banner
    print("="*60)
    print("FAISS Semantic Search API")
    print("Flow: source -> load -> transform -> embed -> store -> retrieve")
    print("="*60)
    print("\nInitializing...")
    
    # Pre-load components on startup
    # This makes the first request faster (models already loaded)
    try:
        get_embedder()
        print("✓ Text embedder ready")
    except Exception as e:
        print(f"⚠ Error loading text embedder: {str(e)}")
    
    try:
        if ImageEmbedder is not None:
            get_image_embedder()
            print("✓ Image embedder (CLIP) ready")
        else:
            print("⚠ Image embedder not available (install: pip install transformers Pillow)")
            print("  Text search will work, but image search will be disabled")
    except Exception as e:
        print(f"⚠ Error loading image embedder: {str(e)}")
        print("  Note: Image search will not work without CLIP model")
    
    try:
        get_store()
        print("✓ Vector store ready")
    except Exception as e:
        print(f"⚠ Error loading store: {str(e)}")
    
    # Print API information
    print("\n" + "="*60)
    print("API Endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  POST /ingest    - Ingest documents")
    print("  POST /ingest_pdf - Ingest PDF files")
    print("  POST /search    - Search documents")
    print("  GET  /stats     - Get statistics")
    print("  POST /clear     - Clear index")
    print("="*60)
    print("\nServer starting on http://localhost:5001\n")
    
    # Start Flask development server
    app.run(host='0.0.0.0', port=5001, debug=True)
