"""
PDF PROCESSING MODULE
=====================
This module handles PDF semantic search by:
1. Splitting PDF into individual single-page PDFs
2. Converting each single-page PDF to an image using pdf2image
3. Embedding page images using CLIP

What happens:
1. Takes a PDF file path
2. Splits PDF into single-page PDFs
3. Converts each single-page PDF to an image using pdf2image (poppler)
4. Embeds page images using CLIP
5. Stores everything in FAISS indices

Flow: PDF FILE -> SPLIT INTO SINGLE-PAGE PDFs -> CONVERT TO IMAGES -> EMBED (CLIP) -> STORE

Why this approach?
- pdf2image uses poppler which is more reliable for rendering
- Works for all PDFs (scanned, text-based, mixed)
- CLIP can understand both text and images in rendered pages
"""

import os
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# PDF processing library for splitting
try:
    import fitz  # PyMuPDF for splitting PDFs
except ImportError:
    fitz = None
    raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

# PDF to image conversion
try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_path = None
    convert_from_bytes = None
    PDF2IMAGE_AVAILABLE = False
    # Don't raise here - let the functions handle it gracefully

# Image processing
try:
    from PIL import Image
except ImportError:
    Image = None
    raise ImportError("Pillow (PIL) not installed. Run: pip install Pillow")

# CLIP for multimodal embedding
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    torch = None
    CLIPProcessor = None
    CLIPModel = None
    raise ImportError("transformers not installed. Run: pip install transformers torch")


class PDFMultimodalEmbedder:
    """
    Multimodal embedder for PDFs using CLIP.
    
    Uses CLIP to embed page images from PDFs.
    CLIP can understand both text and images in the rendered page images.
    
    CURRENT MODEL: 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K' (BEST ACCURACY)
    - 1024-dimensional vectors
    - State-of-the-art accuracy for multimodal search
    """
    
    def __init__(self):
        """Initialize the PDF multimodal embedder (model loads lazily)."""
        self.model = None
        self.processor = None
        self.dimension = 1024  # CLIP-ViT-H-14 dimension
    
    def load(self):
        """
        Load the CLIP model from HuggingFace.
        
        Note: First time will download the model (~2.5GB).
        Subsequent calls reuse the cached model.
        """
        if CLIPModel is None:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        if self.model is None:
            print("="*60)
            print("Loading PDF Multimodal Embedder: CLIP-ViT-H-14")
            print("="*60)
            print("📥 Downloading model (~2.5GB) - this may take several minutes...")
            print("   (Model will be cached for future use)")
            print("-"*60)
            
            # Load CLIP model for multimodal embedding
            self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            
            # Set to evaluation mode (faster, no gradients)
            self.model.eval()
            
            print("-"*60)
            print("✓ PDF Multimodal Embedder loaded successfully!")
            print("="*60)
    
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Embed images using CLIP image encoder.
        
        Args:
            images: List of PIL Image objects (rendered PDF pages)
            
        Returns:
            numpy array of shape (num_images, 1024)
            Each row is a vector representing one page image
        """
        if self.model is None:
            self.load()
        
        with torch.no_grad():
            # Process images for CLIP
            inputs = self.processor(images=images, return_tensors="pt")
            
            # Get image embeddings
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embeddings = image_features.numpy().astype(np.float32)
        
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to a vector (for searching PDF pages with text).
        
        Args:
            text: Text query (e.g., "company overview")
            
        Returns:
            numpy array of shape (1024,)
            Text embedding in same space as page images
        """
        if self.model is None:
            self.load()
        
        with torch.no_grad():
            # Process text for CLIP
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            
            # Get text embedding
            text_features = self.model.get_text_features(**inputs)
            
            # Normalize for cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding = text_features[0].numpy().astype(np.float32)
        
        return embedding


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    if fitz is None:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")
    
    # Open PDF
    doc = fitz.open(str(pdf_path))
    text_parts = []
    
    print(f"  Extracting text from {len(doc)} pages...")
    
    # Extract text from each page
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
                print(f"    ✓ Extracted text from page {page_num + 1}/{len(doc)} ({len(page_text)} characters)")
        except Exception as e:
            print(f"    ✗ ERROR: Failed to extract text from page {page_num}: {e}")
            continue
    
    doc.close()
    
    # Combine all text
    full_text = "\n\n".join(text_parts)
    print(f"  ✓ Extracted {len(full_text)} total characters from PDF")
    
    return full_text


def split_pdf_into_pages(pdf_path: str, output_dir: str) -> List[str]:
    """
    Split a PDF into individual single-page PDFs.
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory to save single-page PDFs
        
    Returns:
        List of paths to single-page PDF files
    """
    if fitz is None:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF
    doc = fitz.open(str(pdf_path))
    filename = pdf_path_obj.stem  # Filename without extension
    single_page_paths = []
    
    print(f"  Splitting PDF into {len(doc)} single-page PDFs...")
    
    # Split into single pages
    for page_num in range(len(doc)):
        try:
            # Create a new PDF with just this page
            single_page_doc = fitz.open()  # New empty PDF
            single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Save single-page PDF
            single_page_filename = f"{filename}_page{page_num}.pdf"
            single_page_path = os.path.join(output_dir, single_page_filename)
            single_page_doc.save(single_page_path)
            single_page_doc.close()
            
            single_page_paths.append(single_page_path)
            print(f"    ✓ Created page {page_num + 1}/{len(doc)}: {single_page_filename}")
            
        except Exception as e:
            print(f"    ✗ ERROR: Failed to create single-page PDF for page {page_num}: {e}")
            continue
    
    doc.close()
    
    print(f"  ✓ Created {len(single_page_paths)} single-page PDFs")
    
    return single_page_paths


def convert_pdf_pages_to_images(pdf_paths: List[str], dpi: int = 200) -> List[Dict]:
    """
    Convert single-page PDFs to images using pdf2image.
    
    Args:
        pdf_paths: List of paths to single-page PDF files
        dpi: Resolution for image conversion (default: 200)
        
    Returns:
        List of page dictionaries, each containing:
        - page_num: Page number (0-indexed)
        - image: PIL Image object of the rendered page
        - source_file: Original PDF filename
        - image_path: Path where page image was saved
    """
    if convert_from_path is None:
        raise ImportError(
            "pdf2image not installed. Run: pip install pdf2image\n"
            "Also install poppler:\n"
            "  - Mac: brew install poppler\n"
            "  - Linux: apt-get install poppler-utils or yum install poppler-utils\n"
            "  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/"
        )
    
    if Image is None:
        raise ImportError("Pillow not installed. Run: pip install Pillow")
    
    # Create directory to save page images
    save_dir = "./data/pdf_images"
    os.makedirs(save_dir, exist_ok=True)
    
    pages = []
    
    print(f"  Converting {len(pdf_paths)} single-page PDFs to images...")
    
    for idx, single_page_pdf_path in enumerate(pdf_paths):
        try:
            # Extract page number and original filename from path
            pdf_path_obj = Path(single_page_pdf_path)
            filename_with_page = pdf_path_obj.stem  # e.g., "company_page0"
            
            # Extract original filename and page number
            if "_page" in filename_with_page:
                parts = filename_with_page.rsplit("_page", 1)
                original_filename = parts[0] + ".pdf"
                page_num = int(parts[1])
            else:
                original_filename = pdf_path_obj.name
                page_num = idx
            
            # Convert single-page PDF to image using pdf2image
            images = convert_from_path(
                single_page_pdf_path,
                dpi=dpi,
                first_page=1,
                last_page=1,
                fmt='png'
            )
            
            if not images:
                print(f"    ⚠ Warning: No image generated for {pdf_path_obj.name}")
                continue
            
            # Get the first (and only) image
            pil_image = images[0]
            
            # Ensure RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save page image for reference
            image_filename = f"{original_filename.replace('.pdf', '')}_page{page_num}.png"
            saved_path = os.path.join(save_dir, image_filename)
            pil_image.save(saved_path, quality=95)
            
            # Verify image has content (not all white/blank)
            img_array = np.array(pil_image)
            non_white_pixels = np.sum(np.any(img_array < 250, axis=2))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            white_ratio = 1.0 - (non_white_pixels / total_pixels)
            
            if white_ratio > 0.99:
                print(f"    ⚠ Warning: Page {page_num} appears to be blank ({(white_ratio*100):.1f}% white)")
            
            # Add to pages list
            pages.append({
                "page_num": page_num,
                "image": pil_image,
                "source_file": original_filename,
                "image_path": saved_path
            })
            
            print(f"    ✓ Converted page {page_num + 1} ({pil_image.size[0]}x{pil_image.size[1]}, {non_white_pixels}/{total_pixels} non-white pixels)")
            
        except Exception as e:
            print(f"    ✗ ERROR: Failed to convert {single_page_pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  ✓ Converted {len(pages)} pages to images")
    
    return pages


def process_pdf(pdf_path: str, dpi: int = 200, pdf_type: str = "with_photo") -> Dict:
    """
    Complete PDF processing pipeline with support for text extraction OR image conversion.
    
    This function supports two modes:
    1. "plain": Extract text only → chunk → embed (text embeddings only, NO images)
    2. "with_photo": Convert pages to images → embed images (image embeddings only, NO text)
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for image conversion (default: 200, only used if pdf_type="with_photo")
        pdf_type: Processing mode - "plain" (text only) or "with_photo" (images only)
    
    Returns:
        Dictionary with:
        - pages: List of page dictionaries with images (if pdf_type="with_photo")
        - image_vectors: numpy array of page image embeddings (if pdf_type="with_photo")
        - text_content: Extracted text content (if pdf_type="plain")
        - embedder: PDFMultimodalEmbedder instance (for reuse, if images were processed)
    """
    print("="*60)
    print("PDF Processing Pipeline")
    print("="*60)
    print(f"Input PDF: {pdf_path}")
    print(f"PDF Type: {pdf_type}")
    
    # Validate pdf_type
    if pdf_type not in ["plain", "with_photo"]:
        raise ValueError(f"pdf_type must be 'plain' or 'with_photo', got: {pdf_type}")
    
    result = {
        "pages": [],
        "image_vectors": None,
        "text_content": None,
        "embedder": None
    }
    
    # Process based on pdf_type
    if pdf_type == "plain":
        # PLAIN MODE: Extract text only (NO images)
        print("Mode: PLAIN (text extraction only)")
        print("Step 1: Extracting text from PDF...")
        try:
            text_content = extract_text_from_pdf(pdf_path)
            result["text_content"] = text_content
            print(f"  ✓ Extracted {len(text_content)} characters of text")
        except Exception as e:
            print(f"  ✗ ERROR: Failed to extract text: {e}")
            raise  # Re-raise since text is required for plain mode
    
    elif pdf_type == "with_photo":
        # WITH_PHOTO MODE: Convert to images only (NO text extraction)
        print("Mode: WITH_PHOTO (image conversion only)")
        
        # Check if pdf2image is available
        if convert_from_path is None:
            error_msg = (
                "pdf2image is not installed or poppler is missing.\n"
                "Install: pip install pdf2image\n"
                "Also install poppler:\n"
                "  - Mac: brew install poppler\n"
                "  - Linux: apt-get install poppler-utils or yum install poppler-utils\n"
                "  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/"
            )
            print(f"  ✗ ERROR: {error_msg}")
            raise ImportError(error_msg)
        
        # Step 1: Split PDF into single-page PDFs
        print("Step 1: Splitting PDF into single-page PDFs...")
        temp_dir = tempfile.mkdtemp(prefix="pdf_pages_")
        print(f"  Temporary directory: {temp_dir}")
        
        try:
            single_page_pdfs = split_pdf_into_pages(pdf_path, temp_dir)
            
            if not single_page_pdfs:
                print("  ✗ ERROR: No single-page PDFs could be created")
                print("    This might indicate the PDF is corrupted or empty")
                raise ValueError("Failed to split PDF into pages")
            
            print(f"  ✓ Created {len(single_page_pdfs)} single-page PDFs")
            
            # Step 2: Convert single-page PDFs to images
            print("Step 2: Converting single-page PDFs to images using pdf2image...")
            print(f"  Using DPI: {dpi}")
            pages = convert_pdf_pages_to_images(single_page_pdfs, dpi=dpi)
            
            if not pages:
                print("  ✗ ERROR: No pages could be converted to images")
                print("    This might indicate:")
                print("    1. pdf2image/poppler is not working correctly")
                print("    2. PDF pages are corrupted")
                print("    3. Check if poppler is installed: which pdftoppm")
                raise ValueError("Failed to convert PDF pages to images")
            
            print(f"  ✓ Converted {len(pages)} pages to images")
            result["pages"] = pages
            
            # Step 3: Initialize embedder
            print("Step 3: Loading multimodal embedder (CLIP)...")
            embedder = PDFMultimodalEmbedder()
            embedder.load()
            result["embedder"] = embedder
            
            # Step 4: Embed page images
            print("Step 4: Embedding page images...")
            pil_images = [page_data["image"] for page_data in pages]
            image_vectors = embedder.embed_images(pil_images)
            result["image_vectors"] = image_vectors
            print(f"  ✓ Generated {len(image_vectors)} page embeddings")
        
        except Exception as e:
            print(f"  ✗ ERROR in image processing: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise since image processing is required for with_photo mode
        
        finally:
            # Clean up temporary single-page PDFs
            import shutil
            try:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
                    print(f"  ✓ Cleaned up temporary files from {temp_dir}")
            except Exception as e:
                print(f"  ⚠ Warning: Could not clean up temp directory: {e}")
    
    print("="*60)
    print("✓ PDF processing complete!")
    if result["text_content"]:
        print(f"  Text extracted: {len(result['text_content'])} characters")
    if result["pages"]:
        print(f"  Pages processed as images: {len(result['pages'])}")
    if result["image_vectors"] is not None:
        print(f"  Image embeddings created: {len(result['image_vectors'])}")
    print("="*60)
    
    return result
    
    print("="*60)
    print("✓ PDF processing complete!")
    if result["text_content"]:
        print(f"  Text extracted: {len(result['text_content'])} characters")
    if result["pages"]:
        print(f"  Pages processed as images: {len(result['pages'])}")
    if result["image_vectors"] is not None:
        print(f"  Image embeddings created: {len(result['image_vectors'])}")
    print("="*60)
    
    return result
    


def store_pdf_data(store, pdf_data: Dict, pdf_filename: str, text_chunks: List[Dict] = None, text_vectors: np.ndarray = None):
    """
    Store PDF data (text chunks and/or page images) in the vector store.
    
    This function takes processed PDF data and stores it in the appropriate FAISS indices.
    - Text chunks go to text_index
    - Page images go to image_index
    
    Args:
        store: VectorStore instance
        pdf_data: Dictionary from process_pdf() containing pages, vectors, and text_content
        pdf_filename: Name of the PDF file (for metadata)
        text_chunks: Optional list of text chunks (if text was extracted and chunked)
        text_vectors: Optional numpy array of text embeddings (if text was embedded)
    """
    # Store text chunks and vectors if provided
    if text_chunks is not None and text_vectors is not None and len(text_chunks) > 0:
        # Initialize text index if needed
        if store.text_index is None:
            store.initialize_text_index(text_vectors.shape[1])
        
        # Add vectors to index
        store.text_index.add(text_vectors)
        
        # Store metadata for each chunk
        for chunk in text_chunks:
            store.text_metadata.append({
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "chunk_text": chunk["chunk_text"],
                "type": "pdf_text"
            })
        
        print(f"  ✓ Stored {len(text_vectors)} text vectors from PDF")
    
    # Store page image vectors if available
    if pdf_data.get("image_vectors") is not None and pdf_data.get("pages"):
        vectors = pdf_data["image_vectors"]
        pages = pdf_data["pages"]
        
        # Initialize image index if needed
        if store.image_index is None:
            store.initialize_image_index(vectors.shape[1])
        
        # Add vectors to index
        store.image_index.add(vectors)
        
        # Store metadata for each page
        for page_data in pages:
            store.image_metadata.append({
                "image_id": f"{page_data['source_file']}::page_{page_data['page_num']}",
                "source_file": page_data["source_file"],
                "page_num": page_data["page_num"],
                "image_index": page_data["page_num"],  # Use page_num as image_index
                "image_path": page_data.get("image_path"),
                "type": "pdf_page"
            })
        
        print(f"  ✓ Stored {len(vectors)} page image vectors from PDF")
