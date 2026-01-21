from __future__ import annotations

"""
PRODUCTION FAISS Semantic Search API
====================================
Production-ready Flask API with:
- Stable vector IDs (IndexIDMap2)
- SQLite metadata store
- Deletion support (soft/hard)
- IVF-PQ indexing with smart retraining
- Namespace isolation
"""

import os
import sys
import warnings
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Disable multiprocessing parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*semaphore.*")

_original_warn = warnings.warn
def _suppress_multiprocessing_warnings(message, *args, **kwargs):
    if "resource_tracker" in str(message) or "semaphore" in str(message):
        return
    return _original_warn(message, *args, **kwargs)
warnings.warn = _suppress_multiprocessing_warnings

from flask import Flask, request, jsonify

# Add parent directory to path to import from main src/
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Import from main src/ directory (shared modules)
from src.embed import Embedder
from src.transform import transform_to_chunks
from src.load import load_documents

# Import PDF processing components
try:
    from src.pdf_process import extract_text_from_pdf, convert_pdf_pages_to_images
except ImportError:
    extract_text_from_pdf = None
    convert_pdf_pages_to_images = None

# Import store_v2 from local faiss-research/src/
# Add current directory to path for local imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import store_v2 directly from the local src directory
import importlib.util
store_v2_path = os.path.join(current_dir, "src", "store_v2.py")
spec = importlib.util.spec_from_file_location("store_v2", store_v2_path)
store_v2_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store_v2_module)
VectorStoreV2 = store_v2_module.VectorStoreV2

# Import config
config_path = os.path.join(current_dir, "src", "config.py")
config_spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
get_config = config_module.get_config

import json

# Lazy imports
try:
    from src.image_embed import ImageEmbedder
except ImportError:
    ImageEmbedder = None

app = Flask(__name__)
# Base directory is parent of faiss-research (main project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load configuration
config = get_config()

# Global instances
embedder = None
image_embedder = None
store = None


def get_embedder():
    global embedder
    if embedder is None:
        embedder = Embedder()
        embedder.load()
    return embedder


def get_image_embedder():
    global image_embedder
    if ImageEmbedder is None:
        raise ImportError("Image embedding not available")
    if image_embedder is None:
        image_embedder = ImageEmbedder()
        image_embedder.load()
    return image_embedder


def get_store():
    global store
    if store is None:
        # Use shard-specific data directory from config
        data_dir = config.get_data_dir()
        store = VectorStoreV2(data_dir=data_dir)
        store.load()
    return store


def _describe_images_with_ollama(images: List[Dict], chunk_size: int, overlap: int, embedder) -> Tuple[Optional[List[Dict]], Optional[np.ndarray]]:
    """
    Generate image descriptions using Ollama and create text embeddings.
    
    Returns:
        Tuple of (image_description_chunks, image_description_vectors) or (None, None) if failed
    """
    try:
        import ollama
    except ImportError:
        print("  ⚠ Warning: ollama not installed. Run: pip install ollama")
        print("  Images will only have CLIP embeddings (no text descriptions)")
        return None, None
    
    try:
        models = ollama.list()
        print(f"  ✓ Ollama is available")
        
        vision_model = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b")
        print(f"  Using vision model: {vision_model}")
        
        def describe_image_with_ollama(image_path: str, model: str = None) -> str:
            """Use Ollama to describe an image in detail."""
            if model is None:
                model = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b")
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": "Describe this image in detail, including all text, objects, layout, colors, and any other relevant information. Be thorough and specific.",
                        "images": [image_path]
                    }]
                )
                return response["message"]["content"]
            except Exception as e:
                print(f"  ⚠ Warning: Failed to describe image {image_path}: {e}")
                return f"Image description unavailable: {str(e)}"
        
        # Describe each image
        image_descriptions = []
        for idx, img_data in enumerate(images):
            try:
                image_path = img_data.get("image_path")
                if not image_path:
                    import tempfile
                    from PIL import Image
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    img_data["image"].save(temp_file.name)
                    image_path = temp_file.name
                    img_data["image_path"] = image_path
                
                print(f"    Describing image {idx + 1}/{len(images)}: {img_data['source_file']}...")
                description = describe_image_with_ollama(image_path, model=vision_model)
                image_descriptions.append({
                    "source_file": img_data["source_file"],
                    "image_path": img_data.get("image_path"),
                    "description": description
                })
                print(f"    ✓ Described {img_data['source_file']} ({len(description)} characters)")
            except Exception as e:
                print(f"    ✗ ERROR: Failed to describe {img_data['source_file']}: {e}")
                continue
        
        if not image_descriptions:
            return None, None
        
        # Chunk the descriptions
        print("  Chunking image descriptions...")
        description_documents = []
        for desc_data in image_descriptions:
            description_documents.append({
                "filename": f"{desc_data['source_file']}_description",
                "filepath": desc_data.get("image_path", ""),
                "content": desc_data["description"]
            })
        
        image_description_chunks = transform_to_chunks(description_documents, chunk_size=chunk_size, overlap=overlap)
        
        # Add metadata to identify as image descriptions
        for chunk in image_description_chunks:
            chunk["type"] = "image_description"
            chunk["is_image_description"] = True
            for desc_data in image_descriptions:
                if desc_data["source_file"] in chunk.get("chunk_id", ""):
                    chunk["image_path"] = desc_data.get("image_path")
                    chunk["original_image_file"] = desc_data["source_file"]
                    break
        
        print(f"  ✓ Created {len(image_description_chunks)} description chunks")
        
        # Embed description chunks
        print("  Embedding image descriptions...")
        desc_chunk_texts = [chunk["chunk_text"] for chunk in image_description_chunks]
        image_description_vectors = embedder.embed(desc_chunk_texts, batch_size=128)
        print(f"  ✓ Generated {len(image_description_vectors)} description embeddings")
        
        return image_description_chunks, image_description_vectors
        
    except Exception as e:
        print(f"  ⚠ Warning: Could not connect to Ollama: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        print(f"  And pull a vision model: ollama pull gemma3:4b")
        return None, None


def _process_pdf_file(pdf_path: Path, embedder, image_embedder, chunk_size: int, overlap: int, 
                     namespace: str, store) -> Dict:
    """
    Process a single PDF file: extract text and convert pages to images.
    
    Returns:
        Dictionary with processing statistics
    """
    pdf_stats = {}
    pdf_filename = os.path.basename(str(pdf_path))
    
    try:
        print(f"\nProcessing PDF: {pdf_filename}")
        
        # Extract text from PDF
        if extract_text_from_pdf:
            print("  Step 1: Extracting text from PDF...")
            try:
                text_content = extract_text_from_pdf(str(pdf_path))
                if text_content:
                    document = {
                        "filename": pdf_filename,
                        "filepath": str(pdf_path),
                        "content": text_content
                    }
                    text_chunks_pdf = transform_to_chunks([document], chunk_size=chunk_size, overlap=overlap)
                    if text_chunks_pdf:
                        chunk_texts = [chunk["chunk_text"] for chunk in text_chunks_pdf]
                        text_vectors_pdf = embedder.embed(chunk_texts, batch_size=128)
                        store.store_text(text_vectors_pdf, text_chunks_pdf, namespace=namespace, skip_training_check=True)
                        pdf_stats["pdf_text_vectors_stored"] = len(text_vectors_pdf)
                        print(f"  ✓ Stored {len(text_vectors_pdf)} text embeddings from PDF")
            except Exception as e:
                print(f"  ⚠ Warning: Could not extract text from PDF: {e}")
        
        # Convert PDF pages to images and embed with CLIP
        if convert_pdf_pages_to_images and image_embedder:
            print("  Step 2: Converting PDF pages to images...")
            try:
                pages = convert_pdf_pages_to_images([str(pdf_path)], dpi=200)
                if pages:
                    from PIL import Image
                    pil_images_pdf = []
                    for page_data in pages:
                        if page_data.get("image_path") and os.path.exists(page_data["image_path"]):
                            pil_images_pdf.append(Image.open(page_data["image_path"]))
                    
                    if pil_images_pdf:
                        print(f"  Generating CLIP embeddings for {len(pil_images_pdf)} PDF pages...")
                        image_vectors_pdf = image_embedder.embed_images(pil_images_pdf)
                        
                        image_data_list = []
                        for idx, page_data in enumerate(pages):
                            if idx < len(image_vectors_pdf):
                                image_data_list.append({
                                    "source_file": page_data.get("source_file", pdf_filename),
                                    "image_index": page_data.get("page_num", 0),
                                    "image_path": page_data.get("image_path")
                                })
                        
                        if image_data_list:
                            store.store_images(image_vectors_pdf, image_data_list, namespace=namespace, skip_training_check=True)
                            pdf_stats["pdf_image_vectors_stored"] = len(image_vectors_pdf)
                            print(f"  ✓ Stored {len(image_vectors_pdf)} image embeddings from PDF pages")
                        
                        pdf_stats["pdf_pages_processed"] = len(pages)
            except Exception as e:
                print(f"  ⚠ Warning: Could not process PDF pages as images: {e}")
        
        pdf_stats["pdf_files_processed"] = 1
        return pdf_stats
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def root():
    """API information."""
    return jsonify({
        "message": "FAISS Semantic Search API (Production)",
        "version": "2.0.0",
        "features": [
            "Stable vector IDs (IndexIDMap2)",
            "SQLite metadata store",
            "Deletion support",
            "Namespace isolation",
            "Image descriptions (Ollama) - Text-to-image search"
        ],
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /whoami": "Get shard information",
            "POST /ingest": "Ingest documents from folder (full pipeline)",
            "POST /search": "Search vectors",
            "POST /delete": "Delete by doc_id or chunk_id (soft delete by default, use hard_delete=true for permanent)",
            "POST /restore": "Restore soft-deleted vectors (undo soft delete)",
            "GET /stats": "Get statistics",
            "GET /vectors": "Get all vectors (with optional namespace, limit, offset)",
            "POST /reset": "Reset - Clear all data and start fresh"
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({"ok": True}), 200


@app.route('/whoami', methods=['GET'])
def whoami():
    """Get shard information."""
    try:
        store = get_store()
        stats = store.get_stats()
        
        # Count namespaces
        namespaces_count = len(stats.get("namespaces", {}))
        
        return jsonify({
            "shard_id": config.shard_id,
            "shard_count": config.shard_count,
            "is_shard_mode": config.is_shard_mode(),
            "data_dir": str(config.shard_data_dir),
            "port": config.port,
            "namespaces_count": namespaces_count
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/ingest', methods=['POST'])
def ingest():
    """
    Ingest documents from a folder (full pipeline).
    
    This runs the complete pipeline: LOAD -> TRANSFORM -> EMBED -> STORE
    Uses the production VectorStoreV2 with stable IDs and SQLite metadata.
    
    Request body (JSON):
    {
        "docs_path": "./docs",      # Path to folder with documents and images
        "chunk_size": 800,          # Max characters per chunk (default: 800)
        "overlap": 120,             # Characters to overlap between chunks (default: 120)
        "extract_images": true,      # Load and index image files (default: true)
        "namespace": "default"      # Optional namespace (default: "default")
    }
    
    Note: PDFs are automatically processed with BOTH:
    - Text extraction → chunking → text embeddings
    - Page images → CLIP embeddings
    - Image descriptions (Ollama) → text embeddings (if Ollama available)
    """
    global store
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        # Extract parameters
        docs_path = data.get("docs_path", "./docs")
        chunk_size = data.get("chunk_size", 800)
        overlap = data.get("overlap", 120)
        extract_images = data.get("extract_images", True)
        namespace = data.get("namespace", "default")
        
        # Convert relative path to absolute
        if not os.path.isabs(docs_path):
            docs_path = os.path.join(BASE_DIR, docs_path)
        
        # Validate parameters
        if chunk_size <= 0:
            return jsonify({"error": "chunk_size must be positive"}), 400
        if overlap < 0:
            return jsonify({"error": "overlap must be non-negative"}), 400
        if overlap >= chunk_size:
            return jsonify({"error": "overlap must be less than chunk_size"}), 400
        
        # Load documents
        documents, images = load_documents(docs_path, extract_images=extract_images)
        
        # Transform text to chunks
        chunks = transform_to_chunks(documents, chunk_size, overlap)
        
        # Get embedders
        embedder = get_embedder()
        image_embedder = None
        if images and extract_images:
            try:
                image_embedder = get_image_embedder()
            except:
                pass
        
        # Embed text chunks
        text_vectors = None
        if chunks:
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            text_vectors = embedder.embed(chunk_texts, batch_size=128)
        
        # Embed images (CLIP embeddings for image-to-image search)
        image_vectors = None
        if images and image_embedder:
            pil_images = [img_data["image"] for img_data in images]
            image_vectors = image_embedder.embed_images(pil_images)
        
        # Describe images with Ollama and create text embeddings (for text-to-image search)
        image_description_chunks = None
        image_description_vectors = None
        if images and extract_images:
            image_description_chunks, image_description_vectors = _describe_images_with_ollama(
                images, chunk_size, overlap, embedder
            )
        
        # Store in production VectorStoreV2
        store = get_store()
        
        # PHASE 1: Collect ALL vectors first (store in SQLite without training)
        # Store text chunks (skip training check - will train at end)
        if text_vectors is not None and chunks:
            store.store_text(text_vectors, chunks, namespace=namespace, skip_training_check=True)
        
        # Store image embeddings (CLIP) (skip training check - will train at end)
        if image_vectors is not None and images:
            store.store_images(image_vectors, images, namespace=namespace, skip_training_check=True)
        
        # Store image description chunks as text embeddings (skip training check - will train at end)
        if image_description_vectors is not None and image_description_chunks:
            store.store_text(image_description_vectors, image_description_chunks, namespace=namespace, skip_training_check=True)
        
        # Process PDFs - ALWAYS do BOTH text and image embeddings
        pdf_stats = {}
        if extract_text_from_pdf or convert_pdf_pages_to_images:
            docs_path_obj = Path(docs_path)
            pdf_files = list(docs_path_obj.glob("*.pdf")) + list(docs_path_obj.glob("*.PDF"))
            
            for pdf_path in pdf_files:
                file_stats = _process_pdf_file(pdf_path, embedder, image_embedder, chunk_size, overlap, namespace, store)
                # Aggregate stats
                for key, value in file_stats.items():
                    if key == "error":
                        continue
                    pdf_stats[key] = pdf_stats.get(key, 0) + value
        
        # PHASE 2: Finalize ingestion - check 20% rule ONCE and train if needed
        print("\n" + "="*60)
        print("Finalizing ingestion: Checking 20% rule and training if needed...")
        print("="*60)
        
        final_training_info = store.finalize_ingestion(namespace=namespace)
        
        text_training_info = final_training_info.get("text_training_info")
        image_training_info = final_training_info.get("image_training_info")
        
        if text_training_info and text_training_info.get("was_trained"):
            print(f"\n✓ Text index trained (reason: {text_training_info.get('retrain_reason')})")
            print(f"  Existing: {text_training_info.get('existing_ntotal')}, New: {text_training_info.get('new_count')}, Final: {text_training_info.get('final_ntotal')}")
        
        if image_training_info and image_training_info.get("was_trained"):
            print(f"\n✓ Image index trained (reason: {image_training_info.get('retrain_reason')})")
            print(f"  Existing: {image_training_info.get('existing_ntotal')}, New: {image_training_info.get('new_count')}, Final: {image_training_info.get('final_ntotal')}")
        
        # Save to disk
        store.save()
        
        # Reset global store
        store = None
        
        # Determine overall training status
        primary_training_info = text_training_info or image_training_info
        
        # Return statistics with training info
        response = {
            "success": True,
            "message": "Documents ingested successfully",
            "index_type": "IVF_PQ",
            "was_trained": primary_training_info.get("was_trained", False) if primary_training_info else False,
            "retrain_reason": primary_training_info.get("retrain_reason") if primary_training_info else None,
            "existing_ntotal": primary_training_info.get("existing_ntotal", 0) if primary_training_info else 0,
            "new_count": primary_training_info.get("new_count", 0) if primary_training_info else 0,
            "final_ntotal": primary_training_info.get("final_ntotal", 0) if primary_training_info else 0,
            "nlist": primary_training_info.get("nlist") if primary_training_info else None,
            "m": primary_training_info.get("m") if primary_training_info else None,
            "nbits": primary_training_info.get("nbits") if primary_training_info else None,
            "nprobe": primary_training_info.get("nprobe") if primary_training_info else None,
            "stats": {
                "files_loaded": len(documents) + len(images),
                "text_files_loaded": len(documents),
                "image_files_loaded": len(images),
                "chunks_created": len(chunks),
                "text_vectors_stored": len(text_vectors) if text_vectors is not None else 0,
                "images_extracted": len(images),
                "image_vectors_stored": len(image_vectors) if image_vectors is not None else 0,
                "image_descriptions_created": len(image_descriptions) if 'image_descriptions' in locals() and image_descriptions else 0,
                "image_description_chunks": len(image_description_chunks) if 'image_description_chunks' in locals() and image_description_chunks else 0,
                "image_description_vectors_stored": len(image_description_vectors) if 'image_description_vectors' in locals() and image_description_vectors is not None else 0,
                "namespace": namespace,
                **pdf_stats
            }
        }
        
        # Add detailed training info per vector type if available
        if text_training_info:
            response["text_training_info"] = text_training_info
        if image_training_info:
            response["image_training_info"] = image_training_info
        
        return jsonify(response), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Search for similar vectors.
    
    Request body (JSON):
    {
        "query": "machine learning",  // Text query
        "k": 5,  // Number of results
        "namespace": "default",  // Optional namespace filter
        "vector_type": "text"  // "text", "image", or "both"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        query_text = data.get("query", "").strip()
        k = data.get("k", 5)
        namespace = data.get("namespace")
        vector_type = data.get("vector_type", "both")
        
        if not query_text:
            return jsonify({"error": "query cannot be empty"}), 400
        if k <= 0:
            return jsonify({"error": "k must be positive"}), 400
        
        store = get_store()
        embedder = get_embedder()
        
        # Get namespace (default if not provided)
        namespace = namespace or store.default_namespace
        
        results = {
            "query": query_text,
            "k": k,
            "namespace": namespace,
            "text_results": [],
            "image_results": []
        }
        
        # Search text
        if vector_type in ["text", "both"]:
            try:
                query_vector = embedder.embed([query_text])[0]
                query_vector = query_vector.reshape(1, -1).astype(np.float32)
                text_results = store.search_text(query_vector, k, namespace=namespace)
                results["text_results"] = text_results
            except Exception as e:
                print(f"Text search failed: {e}")
                results["text_results"] = []
        
        # Search images
        if vector_type in ["image", "both"]:
            try:
                image_embedder = get_image_embedder()
                query_vector = image_embedder.embed_text(query_text)
                query_vector = query_vector.reshape(1, -1).astype(np.float32)
                image_results = store.search_images(query_vector, k, namespace=namespace)
                results["image_results"] = image_results
            except Exception as e:
                print(f"Image search failed: {e}")
                results["image_results"] = []
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/delete', methods=['POST'])
def delete():
    """
    Delete vectors by doc_id or chunk_id.
    
    Request body (JSON):
    {
        "doc_id": "doc1.txt",  // Delete all vectors for this document
        // OR
        "chunk_id": "doc1.txt::chunk_0",  // Delete specific chunk
        "namespace": "default"  // Optional namespace filter
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        doc_id = data.get("doc_id")
        chunk_id = data.get("chunk_id")
        namespace = data.get("namespace")
        hard_delete = data.get("hard_delete", False)  # Default to soft delete
        
        if not doc_id and not chunk_id:
            return jsonify({"error": "Either doc_id or chunk_id must be provided"}), 400
        if doc_id and chunk_id:
            return jsonify({"error": "Provide either doc_id or chunk_id, not both"}), 400
        
        store = get_store()
        
        if doc_id:
            result = store.delete_by_doc_id(doc_id, namespace=namespace, hard_delete=hard_delete)
            store.save()
            
            # Build response message
            if result.get("message"):
                status_message = result["message"]
            elif result["index_updated"] and result["database_updated"]:
                status_message = "✓ Removed from FAISS index and marked as deleted in database"
            else:
                status_message = "⚠ Partial deletion - check details"
            
            return jsonify({
                "success": result["deleted_count"] > 0,
                "message": result.get("message", f"Deleted {result['deleted_count']} vectors for doc_id: {doc_id}"),
                "details": {
                    "doc_id": doc_id,
                    "namespace": namespace or "default",
                    "total_deleted": result["deleted_count"],
                    "text_vectors_deleted": result["text_vectors_deleted"],
                    "image_vectors_deleted": result["image_vectors_deleted"],
                    "index_updated": result["index_updated"],
                    "database_updated": result["database_updated"],
                    "deletion_type": result.get("deletion_type", "soft"),
                    "status": status_message,
                    **{k: v for k, v in result.items() if k not in ["deleted_count", "text_vectors_deleted", "image_vectors_deleted", "index_updated", "database_updated", "deletion_type"]}
                }
            }), 200
        
        if chunk_id:
            result = store.delete_by_chunk_id(chunk_id, namespace=namespace, hard_delete=hard_delete)
            store.save()
            
            # Build response message
            if result.get("message"):
                status_message = result["message"]
            elif result["index_updated"] and result["database_updated"]:
                status_message = "✓ Removed from FAISS index and marked as deleted in database"
            else:
                status_message = "⚠ Partial deletion - check details"
            
            return jsonify({
                "success": result["deleted_count"] > 0,
                "message": result.get("message", f"Deleted {result['deleted_count']} vector(s) for chunk_id: {chunk_id}"),
                "details": {
                    "chunk_id": chunk_id,
                    "namespace": namespace or "default",
                    "vector_id": result["vector_id"],
                    "vector_type": result["vector_type"],
                    "deleted_count": result["deleted_count"],
                    "index_updated": result["index_updated"],
                    "database_updated": result["database_updated"],
                    "deletion_type": result.get("deletion_type", "soft"),
                    "status": status_message,
                    **{k: v for k, v in result.items() if k not in ["deleted_count", "vector_id", "vector_type", "index_updated", "database_updated", "deletion_type"]}
                }
            }), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics about the index."""
    try:
        store = get_store()
        stats_dict = store.get_stats()
        return jsonify(stats_dict), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/restore", methods=["POST"])
def restore():
    """Restore (undo soft delete) vectors for a document or chunk."""
    try:
        data = request.get_json() or {}
        doc_id = data.get("doc_id")
        chunk_id = data.get("chunk_id")
        namespace = data.get("namespace")
        
        if not doc_id and not chunk_id:
            return jsonify({"error": "Either doc_id or chunk_id must be provided"}), 400
        if doc_id and chunk_id:
            return jsonify({"error": "Provide either doc_id or chunk_id, not both"}), 400
        
        store = get_store()
        
        if doc_id:
            result = store.restore_by_doc_id(doc_id, namespace=namespace)
            
            return jsonify({
                "success": result["restored_count"] > 0,
                "message": result.get("message", f"Restored {result['restored_count']} vectors for doc_id: {doc_id}"),
                "details": {
                    "doc_id": doc_id,
                    "namespace": namespace or "default",
                    "total_restored": result["restored_count"],
                    "text_vectors_restored": result["text_vectors_restored"],
                    "image_vectors_restored": result["image_vectors_restored"],
                    "database_updated": result["database_updated"],
                    "note": "Vectors are restored in database but not in FAISS index. Re-ingest document to add them back to index."
                }
            }), 200
        
        if chunk_id:
            result = store.restore_by_chunk_id(chunk_id, namespace=namespace)
            
            return jsonify({
                "success": result["restored_count"] > 0,
                "message": result.get("message", f"Restored {result['restored_count']} vector(s) for chunk_id: {chunk_id}"),
                "details": {
                    "chunk_id": chunk_id,
                    "namespace": namespace or "default",
                    "vector_id": result["vector_id"],
                    "vector_type": result["vector_type"],
                    "restored_count": result["restored_count"],
                    "database_updated": result["database_updated"],
                    "note": "Vector is restored in database but not in FAISS index. Re-ingest chunk to add it back to index."
                }
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/vectors', methods=['GET'])
def get_vectors():
    """
    Get all vectors from the database.
    
    Query parameters:
    - namespace: Optional namespace filter
    - limit: Optional limit (default: 1000)
    - offset: Optional offset for pagination (default: 0)
    """
    try:
        namespace = request.args.get('namespace')
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', type=int, default=0)
        
        store = get_store()
        vectors = store.get_all_vectors(namespace=namespace, limit=limit, offset=offset)
        
        return jsonify({
            "vectors": vectors,
            "count": len(vectors),
            "namespace": namespace,
            "limit": limit,
            "offset": offset
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """
    Reset - Clear all data and start fresh.
    
    This permanently deletes:
    - All vectors from FAISS indices
    - All metadata from SQLite database
    - All index files
    
    Use this to completely start over.
    """
    global store
    try:
        store = get_store()
        store.clear()
        
        # Reset global store
        store = None
        
        return jsonify({
            "success": True,
            "message": "Database and index reset successfully - all data cleared"
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("FAISS Semantic Search API (Production)")
    print("Version 2.0.0")
    print("="*60)
    print("\nInitializing...")
    
    try:
        get_embedder()
        print("✓ Text embedder ready")
    except Exception as e:
        print(f"⚠ Error loading text embedder: {str(e)}")
    
    try:
        if ImageEmbedder is not None:
            get_image_embedder()
            print("✓ Image embedder ready")
        else:
            print("⚠ Image embedder not available")
    except Exception as e:
        print(f"⚠ Error loading image embedder: {str(e)}")
    
    try:
        get_store()
        print("✓ Vector store ready")
    except Exception as e:
        print(f"⚠ Error loading store: {str(e)}")
    
    print("\n" + "="*60)
    print("API Endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health     - Health check")
    print("  POST /ingest     - Ingest documents (full pipeline)")
    print("  POST /search     - Search vectors")
    print("  POST /delete     - Delete by doc_id or chunk_id")
    print("  GET  /stats      - Get statistics")
    print("  POST /reset      - Reset (clear all data)")
    print("="*60)
    port = config.port
    shard_info = ""
    if config.is_shard_mode():
        shard_info = f" (Shard {config.shard_id}/{config.shard_count})"
    print(f"\nServer starting on http://localhost:{port}{shard_info}\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)
