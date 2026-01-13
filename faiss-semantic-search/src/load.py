"""
STEP 1: LOAD
============
This step reads documents and images from files on disk.

What happens:
1. Takes a folder path as input
2. Finds all .txt and .md files (text documents)
3. Finds all image files (.png, .jpg, .jpeg, etc.)
4. Reads text content from text files
5. Loads images from image files
6. Returns both documents and images

Flow: SOURCE (files) -> LOAD (read files) -> returns documents + images

Supported file types:
- Text: .txt, .md
- Images: .png, .jpg, .jpeg, .gif, .bmp, .webp, .tiff, .tif
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple


def load_documents(docs_path: str, extract_images: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """
    Load text documents and image files from a folder.
    
    This function:
    1. Finds all text files (.txt, .md) and reads their content
    2. Finds all image files (.png, .jpg, etc.) and loads them
    3. Returns both as separate lists
    
    Args:
        docs_path: Path to folder containing documents and images
        extract_images: Whether to load and process image files (default: True)
        
    Returns:
        Tuple of (documents, images):
        - documents: List of text document dictionaries
        - images: List of image dictionaries with PIL Image objects
        
    Example:
        documents, images = load_documents("./docs")
        # documents: [
        #     {"filename": "doc1.txt", "filepath": "./docs/doc1.txt", "content": "..."},
        #     {"filename": "doc2.md", "filepath": "./docs/doc2.md", "content": "..."}
        # ]
        # images: [
        #     {"image": <PIL.Image>, "source_file": "logo.png", "image_path": "./data/images/logo.png", ...},
        #     ...
        # ]
    """
    # Convert string path to Path object for easier file operations
    docs_path_obj = Path(docs_path)
    
    # Check if folder exists
    if not docs_path_obj.exists():
        raise ValueError(f"Folder not found: {docs_path}")
    
    # ========================================================================
    # STEP 1: Find all text files (.txt and .md)
    # ========================================================================
    txt_files = list(docs_path_obj.glob("*.txt"))  # Find all .txt files
    md_files = list(docs_path_obj.glob("*.md"))     # Find all .md files
    text_files = txt_files + md_files
    
    print(f"  Found {len(txt_files)} .txt files, {len(md_files)} .md files")
    
    # ========================================================================
    # STEP 2: Initialize image loading support
    # ========================================================================
    # We need Pillow (PIL) to load images
    can_load_images = False
    PILImage = None
    
    if extract_images:
        try:
            from PIL import Image as PILImage
            can_load_images = True
            print(f"  ✓ Image support enabled (Pillow installed)")
        except ImportError as e:
            print(f"  ✗ ERROR: Image loading not available: {e}")
            print(f"  ✗ Install Pillow: pip install Pillow")
            print(f"  ✗ Images will NOT be processed!")
            can_load_images = False
    else:
        print(f"  ⚠ Image loading disabled (extract_images=False)")
        can_load_images = False
    
    # ========================================================================
    # STEP 3: Find all image files
    # ========================================================================
    image_files = []
    if extract_images:
        # Common image file extensions
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp", "*.tiff", "*.tif"]
        
        # Search for each extension (both lowercase and uppercase)
        for ext in image_extensions:
            found_lower = list(docs_path_obj.glob(ext))           # e.g., *.png
            found_upper = list(docs_path_obj.glob(ext.upper()))   # e.g., *.PNG
            image_files.extend(found_lower)
            image_files.extend(found_upper)
        
        if image_files:
            print(f"  Found {len(image_files)} image file(s): {[f.name for f in image_files]}")
            if not can_load_images:
                print(f"  ✗ WARNING: Cannot load images - Pillow not installed!")
        else:
            print(f"  No image files found in {docs_path}")
    
    # ========================================================================
    # STEP 4: Check if we found any files
    # ========================================================================
    if not text_files and not image_files:
        raise ValueError(f"No .txt, .md, or image files found in {docs_path}")
    
    # ========================================================================
    # STEP 5: Load text documents
    # ========================================================================
    documents = []
    
    for file_path in text_files:
        try:
            # Read text file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Store document information
            documents.append({
                "filename": os.path.basename(str(file_path)),  # Just the filename (e.g., "doc.txt")
                "filepath": str(file_path),                     # Full path (e.g., "./docs/doc.txt")
                "content": content                              # File content (the actual text)
            })
            
            print(f"  ✓ Loaded: {os.path.basename(str(file_path))} ({len(content)} characters)")
            
        except Exception as e:
            # If we can't read a file, print warning but continue with other files
            print(f"  ⚠ Warning: Could not read {file_path}: {e}")
            continue
    
    # ========================================================================
    # STEP 6: Load image files
    # ========================================================================
    images = []
    
    if extract_images and can_load_images and PILImage is not None:
        if image_files:
            print(f"\n  Processing {len(image_files)} image file(s)...")
        
        for image_path in image_files:
            try:
                # Load image using PIL (Pillow)
                pil_image = PILImage.open(str(image_path))
                
                # Convert to RGB if needed
                # Some formats like PNG have transparency (RGBA mode)
                # CLIP model expects RGB, so we convert
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Create directory to save images (for reference/viewing)
                save_dir = "./data/images"
                os.makedirs(save_dir, exist_ok=True)
                
                # Save a copy of the image for reference
                # This allows you to view the images that were indexed
                image_filename = os.path.basename(str(image_path))
                saved_path = os.path.join(save_dir, image_filename)
                pil_image.save(saved_path)
                
                # Add to images list
                # This image will be converted to a vector embedding later
                images.append({
                    "image": pil_image,                    # PIL Image object (for embedding)
                    "source_file": image_filename,         # Original filename (e.g., "logo.png")
                    "image_path": saved_path,              # Where we saved it (e.g., "./data/images/logo.png")
                    "image_index": 0                        # Index (always 0 for standalone images)
                })
                
                print(f"  ✓ Loaded and saved image: {image_filename} -> {saved_path}")
                
            except Exception as e:
                print(f"  ✗ ERROR: Could not load image {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    elif extract_images and not can_load_images:
        if image_files:
            print(f"\n  ✗ ERROR: Found {len(image_files)} image file(s) but Pillow is NOT installed!")
            print(f"  ✗ Image files found: {[f.name for f in image_files]}")
            print(f"  ✗ Install Pillow: pip install Pillow")
            print(f"  ✗ These images will NOT be loaded until Pillow is installed")
    
    # ========================================================================
    # STEP 7: Summary
    # ========================================================================
    print(f"\n  📊 LOAD SUMMARY:")
    print(f"    Text files loaded: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"      {i}. {doc['filename']}")
    
    print(f"    Image files loaded: {len(images)}")
    for i, img in enumerate(images, 1):
        print(f"      {i}. {img['source_file']}")
    
    print(f"    TOTAL FILES: {len(documents) + len(images)}")
    
    return documents, images
