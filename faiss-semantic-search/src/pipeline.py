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
from pathlib import Path
import os

# Lazy import for image embedder (only needed if extracting images)
try:
    from .image_embed import ImageEmbedder
except ImportError:
    ImageEmbedder = None

# Lazy import for PDF processing
try:
    from .pdf_process import process_pdf, store_pdf_data, PDF2IMAGE_AVAILABLE
    # Check if pdf2image is actually available
    if not PDF2IMAGE_AVAILABLE:
        process_pdf = None
        store_pdf_data = None
        print("⚠ PDF processing not available: pdf2image is not installed")
        print("  Install: pip install pdf2image")
        print("  Also install poppler: brew install poppler (Mac) or apt-get install poppler-utils (Linux)")
except ImportError as e:
    process_pdf = None
    store_pdf_data = None
    print(f"⚠ PDF processing not available: {e}")
    print("  Install: pip install pdf2image")
    print("  Also install poppler: brew install poppler (Mac) or apt-get install poppler-utils (Linux)")
except AttributeError:
    # PDF2IMAGE_AVAILABLE might not exist in older versions
    try:
        from .pdf_process import process_pdf, store_pdf_data
        # Try to check if it will work by testing convert_from_path
        from .pdf_process import convert_from_path
        if convert_from_path is None:
            process_pdf = None
            store_pdf_data = None
            print("⚠ PDF processing not available: pdf2image is not installed")
    except:
        process_pdf = None
        store_pdf_data = None


def run_pipeline(docs_path: str, chunk_size: int = 800, overlap: int = 120, extract_images: bool = True, pdf_type: str = "with_photo") -> Dict:
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
        pdf_type: PDF processing mode - "plain" (text only) or "with_photo" (text + images) (default: "with_photo")
    
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
        num_chunks = len(chunks)
        print(f"Step 3: Embedding text chunks ({num_chunks} chunks)...")
        embedder = Embedder()
        embedder.load()  # Load the embedding model (downloads on first use)
        
        # Extract just the text from chunks
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        
        # Convert all chunks to vectors at once (batch processing is faster)
        # Use appropriate batch size based on number of chunks
        batch_size = 128 if num_chunks > 50 else 64
        text_vectors = embedder.embed(chunk_texts, batch_size=batch_size)
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
    # STEP 5: PROCESS PDFs
    # ========================================================================
    # Find and process PDF files - split into pages, render as images, embed
    pdf_files = []
    pdf_data_list = []  # Store processed PDF data
    pdf_pages_processed = 0
    pdf_image_vectors = 0
    
    # Check if PDF processing is available
    print("="*60)
    print("Step 5: Checking PDF processing availability...")
    print(f"  process_pdf is None: {process_pdf is None}")
    print(f"  store_pdf_data is None: {store_pdf_data is None}")
    
    if process_pdf is None or store_pdf_data is None:
        print("⚠ PDF Processing Module Status:")
        if process_pdf is None:
            print("  ✗ process_pdf function is not available")
        if store_pdf_data is None:
            print("  ✗ store_pdf_data function is not available")
        print("  This usually means pdf2image is not installed or poppler is missing")
        print("  Install: pip install pdf2image")
        print("  Also install poppler: brew install poppler (Mac) or apt-get install poppler-utils (Linux)")
    else:
        print("✓ PDF Processing Module is available")
    
    if process_pdf is not None and store_pdf_data is not None:
        docs_path_obj = Path(docs_path)
        pdf_files = list(docs_path_obj.glob("*.pdf")) + list(docs_path_obj.glob("*.PDF"))
        
        if pdf_files:
            print("="*60)
            print("Step 5: Processing PDF files...")
            print(f"  Found {len(pdf_files)} PDF file(s): {[f.name for f in pdf_files]}")
            if pdf_type == "plain":
                print(f"  Approach: Extract text -> Chunk -> Embed (text embeddings only)")
            else:
                print(f"  Approach: Split into single-page PDFs -> Convert to images (pdf2image) -> Embed with CLIP")
            
            for pdf_path in pdf_files:
                try:
                    print(f"\n  Processing: {os.path.basename(str(pdf_path))}")
                    print(f"  Full path: {pdf_path}")
                    print(f"  PDF Type: {pdf_type}")
                    
                    # Process PDF based on pdf_type
                    pdf_data = process_pdf(str(pdf_path), dpi=200, pdf_type=pdf_type)
                    
                    if pdf_data is None:
                        print(f"  ✗ ERROR: process_pdf returned None for {pdf_path}")
                        continue
                    
                    pdf_data["pdf_path"] = str(pdf_path)  # Store path for later
                    
                    # Process text ONLY if pdf_type is "plain"
                    text_chunks = None
                    text_vectors = None
                    if pdf_type == "plain" and pdf_data.get("text_content"):
                        # Create a document dict for chunking
                        pdf_filename = os.path.basename(str(pdf_path))
                        document = {
                            "filename": pdf_filename,
                            "filepath": str(pdf_path),
                            "content": pdf_data["text_content"]
                        }
                        
                        # Chunk the text
                        text_len = len(document["content"])
                        if text_len > 100000:
                            print(f"  Chunking PDF text ({text_len:,} characters, this may take a moment)...")
                        else:
                            print("  Chunking PDF text...")
                        text_chunks = transform_to_chunks([document], chunk_size=chunk_size, overlap=overlap)
                        print(f"    ✓ Created {len(text_chunks)} text chunks")
                        
                        # Embed the chunks
                        if text_chunks:
                            num_chunks = len(text_chunks)
                            print(f"  Embedding PDF text chunks ({num_chunks} chunks)...")
                            embedder = Embedder()
                            embedder.load()
                            chunk_texts = [chunk["chunk_text"] for chunk in text_chunks]
                            # Use larger batch size for PDF chunks to speed up processing
                            text_vectors = embedder.embed(chunk_texts, batch_size=128)
                            print(f"    ✓ Generated {len(text_vectors)} text embeddings")
                    
                    # Store text chunks and vectors in pdf_data for later storage
                    pdf_data["text_chunks"] = text_chunks
                    pdf_data["text_vectors"] = text_vectors
                    pdf_data_list.append(pdf_data)
                    
                    # Count PDF pages processed (for images)
                    pages_count = len(pdf_data.get("pages", []))
                    pdf_pages_processed += pages_count
                    
                    # Count image vectors
                    image_vectors_count = 0
                    if pdf_data.get("image_vectors") is not None:
                        image_vectors_count = len(pdf_data["image_vectors"])
                        pdf_image_vectors += image_vectors_count
                    
                    # Count text chunks
                    text_chunks_count = len(text_chunks) if text_chunks else 0
                    
                    print(f"  ✓ Summary:")
                    if text_chunks_count > 0:
                        print(f"    - Text chunks: {text_chunks_count}")
                    if pages_count > 0:
                        print(f"    - Pages processed as images: {pages_count}")
                    if image_vectors_count > 0:
                        print(f"    - Image embeddings: {image_vectors_count}")
                    
                    if pages_count == 0 and pdf_type == "with_photo":
                        print(f"  ⚠ WARNING: No pages were processed as images from {pdf_path}")
                        print(f"    This could mean:")
                        print(f"    1. PDF is corrupted or empty")
                        print(f"    2. pdf2image/poppler failed to convert pages")
                        print(f"    3. Check console output above for errors")
                    
                except Exception as e:
                    print(f"  ✗ ERROR: Failed to process {pdf_path}")
                    print(f"    Error type: {type(e).__name__}")
                    print(f"    Error message: {str(e)}")
                    import traceback
                    print("    Full traceback:")
                    traceback.print_exc()
                    continue
        else:
            print("="*60)
            print("Step 5: No PDF files found")
            print(f"  Searched in: {docs_path}")
    else:
        # Check if PDFs exist but processing is not available
        docs_path_obj = Path(docs_path)
        pdf_files = list(docs_path_obj.glob("*.pdf")) + list(docs_path_obj.glob("*.PDF"))
        if pdf_files:
            print("="*60)
            print("⚠ WARNING: PDF files found but PDF processing not available")
            print(f"  Found {len(pdf_files)} PDF file(s): {[f.name for f in pdf_files]}")
            print(f"  Install: pip install pdf2image")
            print(f"  Also install poppler:")
            print(f"    - Mac: brew install poppler")
            print(f"    - Linux: apt-get install poppler-utils or yum install poppler-utils")
            print(f"    - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
            print(f"  Or use /ingest_pdf endpoint to process PDFs individually")
    
    # ========================================================================
    # STEP 6: STORE
    # ========================================================================
    # Save vectors to FAISS indices
    print("="*60)
    print("Step 6: Storing in FAISS indices...")
    store = VectorStore()
    
    # Try to load existing indices (if we're adding to existing data)
    store.load()
    
    # Store text vectors (from .txt and .md files)
    if text_vectors is not None and chunks:
        if store.text_index is None:
            store.initialize_text_index(text_vectors.shape[1])
        store.store_text(text_vectors, chunks)
        print(f"  ✓ Stored {len(text_vectors)} text vectors (from text files)")
    
    # Store image vectors (from standalone image files)
    if image_vectors is not None and images:
        if store.image_index is None:
            store.initialize_image_index(image_vectors.shape[1])
        store.store_images(image_vectors, images)
        print(f"  ✓ Stored {len(image_vectors)} image vectors (from image files)")
    
    # Store PDF data (text chunks and images from PDFs)
    for pdf_data in pdf_data_list:
        try:
            pdf_filename = os.path.basename(pdf_data["pdf_path"])
            text_chunks = pdf_data.get("text_chunks")
            text_vectors = pdf_data.get("text_vectors")
            store_pdf_data(store, pdf_data, pdf_filename, text_chunks=text_chunks, text_vectors=text_vectors)
            print(f"  ✓ Stored PDF content from {pdf_filename}")
        except Exception as e:
            print(f"  ✗ ERROR: Failed to store PDF {pdf_data.get('pdf_path', 'unknown')}: {e}")
            continue
    
    # Save to disk (persist for next time)
    store.save()
    print(f"  ✓ Saved all indices to disk")
    
    # ========================================================================
    # Return statistics
    # ========================================================================
    total_files = len(documents) + len(images) + len(pdf_files)
    
    # Count PDF text chunks and vectors
    pdf_text_chunks_count = 0
    pdf_text_vectors_count = 0
    for pdf_data in pdf_data_list:
        if pdf_data.get("text_chunks"):
            pdf_text_chunks_count += len(pdf_data["text_chunks"])
        if pdf_data.get("text_vectors") is not None:
            pdf_text_vectors_count += len(pdf_data["text_vectors"])
    
    return {
        "files_loaded": total_files,  # Total files including images and PDFs
        "text_files_loaded": len(documents),  # Just text files (.txt, .md)
        "image_files_loaded": len(images),  # Just standalone image files
        "pdf_files_loaded": len(pdf_files),  # Just PDF files
        "chunks_created": len(chunks),  # Chunks from text files only
        "text_vectors_stored": len(text_vectors) if text_vectors is not None else 0,  # From text files only
        "images_extracted": len(images),  # Standalone images only
        "image_vectors_stored": len(image_vectors) if image_vectors is not None else 0,  # From image files only
        "pdf_pages_processed": pdf_pages_processed,  # Pages processed from PDFs as images
        "pdf_image_vectors_stored": pdf_image_vectors,  # Page image vectors from PDFs
        "pdf_text_chunks_created": pdf_text_chunks_count,  # Text chunks from PDFs
        "pdf_text_vectors_stored": pdf_text_vectors_count  # Text vectors from PDFs
    }
