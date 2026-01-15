"""
STEP 5: RETRIEVE
================
This step searches for documents and images similar to a query.

What happens:
1. Takes a search query (text)
2. Converts query to vectors (text embedding + CLIP text embedding)
3. Searches both text and image indices
4. Returns the most similar chunks and images

Flow: QUERY (text) -> EMBED (query vectors) -> SEARCH (both indices) -> RETRIEVE (results)

How it works:
- Text query is converted to text embedding (1024 dim) for text search
- Same text query is converted to CLIP embedding (1024 dim) for image search
- Both indices are searched separately
- Results are returned with similarity scores
"""

import numpy as np
from typing import List, Dict, Optional
from .embed import Embedder
from .store import VectorStore

# Optional image embedder import (for image search)
try:
    from .image_embed import ImageEmbedder
except ImportError:
    ImageEmbedder = None

# Optional PDF embedder import (for PDF text/image search)
try:
    from .pdf_process import PDFMultimodalEmbedder
except ImportError:
    PDFMultimodalEmbedder = None


def retrieve_similar(query: str, k: int, embedder: Embedder, image_embedder: Optional[object], store: VectorStore, pdf_embedder: Optional[object] = None) -> Dict:
    """
    Search for documents and images similar to the query.
    
    This function:
    1. Converts your text query to vectors
    2. Searches the text index for similar text chunks
    3. Searches the image index for similar images
    4. Returns both text and image results
    
    Args:
        query: Search query text (e.g., "machine learning")
        k: Number of results to return (e.g., 5)
        embedder: Embedder instance for text embeddings
        image_embedder: ImageEmbedder instance for image search (CLIP)
        store: VectorStore instance with indexed documents and images
    
    Returns:
        Dictionary with:
        - text_results: List of text search results
        - image_results: List of image search results
        
    Example:
        results = retrieve_similar("python tutorial", k=5, embedder, image_embedder, store)
        # Returns:
        # {
        #     "text_results": [
        #         {"score": 0.85, "source": "tutorial.txt", "chunk_text": "...", ...},
        #         ...
        #     ],
        #     "image_results": [
        #         {"score": 0.78, "source": "diagram.png", "image_path": "...", ...},
        #         ...
        #     ]
        # }
    """
    results = {
        "text_results": [],
        "image_results": []
    }
    
    # ========================================================================
    # STEP 1: Search text index
    # ========================================================================
    # Note: Text index contains both regular text (.txt, .md) and PDF text chunks.
    # Both are embedded with the same text embedder (BAAI/bge-large-en-v1.5),
    # so we can search them all together.
    if store.text_index is not None and store.text_index.ntotal > 0:
        # Search with regular text embedder (works for both regular text and PDF text)
        query_vector = embedder.embed([query])[0]
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # Search more results than requested to ensure we get good diversity
        num_results = min(k * 2, store.text_index.ntotal)
        scores, indices = store.text_index.search(query_vector, num_results)
        
        # Format text results (includes both regular text and PDF text)
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(store.text_metadata):
                try:
                    metadata = store.text_metadata[idx]
                    result = {
                        "score": float(score),
                        "source": metadata.get("source_file", "unknown"),
                        "chunk_id": metadata.get("chunk_id", f"unknown::{idx}"),
                        "chunk_text": metadata.get("chunk_text", ""),
                        "preview": (metadata.get("chunk_text", "")[:200] + "...") if len(metadata.get("chunk_text", "")) > 200 else metadata.get("chunk_text", ""),
                        "type": metadata.get("type", "text")
                    }
                    # Add page_num if it's PDF text (though PDF text chunks don't have page_num in current implementation)
                    if metadata.get("page_num") is not None:
                        result["page_num"] = metadata.get("page_num")
                    results["text_results"].append(result)
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Warning: Error accessing metadata at index {idx}: {e}")
                    continue
    
    # ========================================================================
    # STEP 2: Search image index using CLIP
    # ========================================================================
    if store.image_index is not None and store.image_index.ntotal > 0 and image_embedder is not None:
        try:
            # Convert query to CLIP text embedding (for image search)
            # CLIP can understand both images and text in the same space
            # So we can search images using text queries!
            query_vector = image_embedder.embed_text(query)
            query_vector = query_vector.reshape(1, -1).astype(np.float32)  # Shape: (1, 1024) for CLIP-ViT-H-14
            
            # Search image index
            scores, indices = store.image_index.search(query_vector, min(k, store.image_index.ntotal))
            
            # Format image results
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(store.image_metadata):
                    try:
                        metadata = store.image_metadata[idx]
                        result = {
                            "score": float(score),
                            "source": metadata.get("source_file", "unknown"),
                            "image_id": metadata.get("image_id", f"unknown::{idx}"),
                            "image_index": metadata.get("image_index", idx),
                            "image_path": metadata.get("image_path"),
                            "type": metadata.get("type", "image")
                        }
                        # Add page_num if it's a PDF image
                        if metadata.get("page_num") is not None:
                            result["page_num"] = metadata["page_num"]
                        results["image_results"].append(result)
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"Warning: Error accessing image metadata at index {idx}: {e}")
                        continue
        except Exception as e:
            print(f"Warning: Image search failed: {e}")
    
    # ========================================================================
    # STEP 3: Remove duplicates (keep highest scoring version)
    # ========================================================================
    # Why duplicates occur:
    # - If you run /ingest multiple times, the same documents/images get added again
    # - FAISS doesn't prevent duplicates, it just adds vectors to the index
    # - This deduplication ensures each unique chunk/image appears only once
    
    # Deduplicate text results by chunk_id
    # Use a dictionary to track unique chunks (key = chunk_id, value = result)
    seen_text = {}
    for result in results["text_results"]:
        chunk_id = result["chunk_id"]
        # If we haven't seen this chunk, or this version has a higher score, keep it
        if chunk_id not in seen_text or result["score"] > seen_text[chunk_id]["score"]:
            seen_text[chunk_id] = result
    
    # Convert back to list (only unique chunks, highest scoring version)
    results["text_results"] = list(seen_text.values())
    
    # Deduplicate image results by image_id
    seen_images = {}
    for result in results["image_results"]:
        image_id = result["image_id"]
        # If we haven't seen this image, or this version has a higher score, keep it
        if image_id not in seen_images or result["score"] > seen_images[image_id]["score"]:
            seen_images[image_id] = result
    
    # Convert back to list (only unique images, highest scoring version)
    results["image_results"] = list(seen_images.values())
    
    # ========================================================================
    # STEP 4: Sort results by score (highest first)
    # ========================================================================
    results["text_results"].sort(key=lambda x: x["score"], reverse=True)
    results["image_results"].sort(key=lambda x: x["score"], reverse=True)
    
    return results
