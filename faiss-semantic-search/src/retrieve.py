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


def retrieve_similar(query: str, k: int, embedder: Embedder, image_embedder: Optional[object], store: VectorStore) -> Dict:
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
    if store.text_index is not None and store.text_index.ntotal > 0:
        # Convert query text to vector using the same embedding model
        # This ensures the query is in the same space as the indexed documents
        query_vector = embedder.embed([query])[0]  # Get first (and only) vector
        query_vector = query_vector.reshape(1, -1).astype(np.float32)  # Shape: (1, 1024) for bge-large-en-v1.5
        
        # Search FAISS index for k most similar vectors
        # search() returns:
        #   - scores: Similarity scores (higher = more similar, range: 0-1)
        #   - indices: Vector IDs in the index
        scores, indices = store.text_index.search(query_vector, min(k, store.text_index.ntotal))
        
        # Format text results
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # Valid result (FAISS returns -1 for invalid)
                # Get metadata for this vector ID
                metadata = store.text_metadata[idx]
                
                # Create result dictionary
                results["text_results"].append({
                    "score": float(score),                    # Similarity score (0-1)
                    "source": metadata["source_file"],        # Original filename
                    "chunk_id": metadata["chunk_id"],        # Chunk identifier
                    "chunk_text": metadata["chunk_text"],    # Full chunk text
                    "preview": metadata["chunk_text"][:200] + "..." if len(metadata["chunk_text"]) > 200 else metadata["chunk_text"],
                    "type": "text"
                })
    
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
                if idx >= 0:
                    metadata = store.image_metadata[idx]
                    results["image_results"].append({
                        "score": float(score),
                        "source": metadata["source_file"],
                        "image_id": metadata["image_id"],
                        "image_index": metadata["image_index"],
                        "image_path": metadata.get("image_path"),  # Path to saved image file
                        "type": "image"
                    })
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
