"""
COMPLETE PIPELINE
=================
This runs all steps in sequence: SOURCE -> LOAD -> TRANSFORM -> EMBED -> STORE

This is the main function that orchestrates the entire ingestion process.
It supports both TEXT and IMAGES.

Each step feeds into the next step.

Flow:
    SOURCE (files on disk)
      ↓
    LOAD (read text files and image files)
      ↓
    TRANSFORM (split text into chunks)
      ↓
    EMBED TEXT (convert text chunks to vectors)
    EMBED IMAGES (convert images to vectors)
      ↓
    STORE (save to FAISS indices)
      ↓
    READY (both text and images searchable!)
"""

from .load import load_documents
from .transform import transform_to_chunks
from .embed import Embedder
from .store import VectorStore
from typing import Dict

# Lazy import for image embedder (only needed if extracting images)
try:
    from .image_embed import ImageEmbedder
except ImportError:
    ImageEmbedder = None


def run_pipeline(docs_path: str, chunk_size: int = 800, overlap: int = 120, extract_images: bool = True) -> Dict:
    """
    Complete ingestion pipeline with TEXT and IMAGE support.
    
    This function runs the complete pipeline:
    1. Loads text files (.txt, .md) and image files (.png, .jpg, etc.)
    2. Splits text into chunks
    3. Converts text chunks to vectors (embeddings)
    4. Converts images to vectors (embeddings)
    5. Stores everything in FAISS indices
    
    Args:
        docs_path: Path to folder containing .txt, .md files and image files
        chunk_size: Maximum characters per chunk (default: 800)
        overlap: Characters to overlap between chunks (default: 120)
        extract_images: Whether to load and index image files (default: True)
    
    Returns:
        Dictionary with statistics:
        - files_loaded: Total files processed (text + images)
        - text_files_loaded: Number of text files
        - image_files_loaded: Number of image files
        - chunks_created: Number of text chunks created
        - text_vectors_stored: Number of text vectors
        - images_extracted: Number of images loaded
        - image_vectors_stored: Number of image vectors
    """
    # ========================================================================
    # STEP 1: LOAD
    # ========================================================================
    # Read all text files and image files from the folder
    print("="*60)
    print("Step 1: Loading documents and images...")
    print(f"  extract_images parameter: {extract_images}")
    print(f"  docs_path: {docs_path}")
    
    documents, images = load_documents(docs_path, extract_images=extract_images)
    
    print(f"  ✓ Loaded {len(documents)} text documents")
    print(f"  ✓ Loaded {len(images)} image files")
    
    if images:
        print(f"    - Images ready for embedding")
        for i, img in enumerate(images, 1):
            print(f"      Image {i+1}: {img.get('source_file', 'unknown')}")
    else:
        if extract_images:
            print(f"    ⚠ WARNING: No images found!")
            print(f"    ⚠ This could mean:")
            print(f"      1. Pillow is not installed (run: pip install Pillow)")
            print(f"      2. No image files in {docs_path}")
    
    print("="*60)
    
    # ========================================================================
    # STEP 2: TRANSFORM
    # ========================================================================
    # Split large text documents into smaller chunks
    print("Step 2: Transforming text to chunks...")
    chunks = transform_to_chunks(documents, chunk_size, overlap)
    print(f"  ✓ Created {len(chunks)} text chunks")
    
    # ========================================================================
    # STEP 3: EMBED TEXT
    # ========================================================================
    # Convert text chunks to vector embeddings
    text_vectors = None
    if chunks:
        print("Step 3: Embedding text chunks...")
        embedder = Embedder()
        embedder.load()  # Load the embedding model (downloads on first use)
        
        # Extract just the text from chunks
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        
        # Convert all chunks to vectors at once (batch processing is faster)
        text_vectors = embedder.embed(chunk_texts)
        print(f"  ✓ Generated {len(text_vectors)} text embeddings")
    
    # ========================================================================
    # STEP 4: EMBED IMAGES
    # ========================================================================
    # Convert images to vector embeddings using CLIP
    image_vectors = None
    if images:
        if ImageEmbedder is None:
            print("  ⚠ Warning: Image embedding not available. Install transformers and Pillow.")
            print("  Images loaded but not indexed. Text search will still work.")
        else:
            print("Step 4: Embedding images...")
            image_embedder = ImageEmbedder()
            image_embedder.load()  # Load CLIP model (downloads on first use, ~2.5GB)
            
            # Extract PIL Images from image data
            pil_images = [img_data["image"] for img_data in images]
            
            # Convert all images to vectors at once (batch processing)
            image_vectors = image_embedder.embed_images(pil_images)
            print(f"  ✓ Generated {len(image_vectors)} image embeddings")
    
    # ========================================================================
    # STEP 5: STORE
    # ========================================================================
    # Save vectors to FAISS indices
    print("Step 5: Storing in FAISS indices...")
    store = VectorStore()
    
    # Try to load existing indices (if we're adding to existing data)
    store.load()
    
    # Store text vectors
    if text_vectors is not None and chunks:
        if store.text_index is None:
            store.initialize_text_index(text_vectors.shape[1])
        store.store_text(text_vectors, chunks)
        print(f"  ✓ Stored {len(text_vectors)} text vectors")
    
    # Store image vectors
    if image_vectors is not None and images:
        if store.image_index is None:
            store.initialize_image_index(image_vectors.shape[1])
        store.store_images(image_vectors, images)
        print(f"  ✓ Stored {len(image_vectors)} image vectors")
    
    # Save to disk (persist for next time)
    store.save()
    print(f"  ✓ Saved all indices to disk")
    
    # ========================================================================
    # Return statistics
    # ========================================================================
    total_files = len(documents) + len(images)
    
    return {
        "files_loaded": total_files,  # Total files including images
        "text_files_loaded": len(documents),  # Just text files
        "image_files_loaded": len(images),  # Just image files
        "chunks_created": len(chunks),
        "text_vectors_stored": len(text_vectors) if text_vectors is not None else 0,
        "images_extracted": len(images),
        "image_vectors_stored": len(image_vectors) if image_vectors is not None else 0
    }
