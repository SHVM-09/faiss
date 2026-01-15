"""
STEP 2: TRANSFORM
=================
This step splits large documents into smaller chunks.

Why chunking?
- Large documents are too big to embed efficiently
- Chunks allow us to find specific parts of documents
- Overlap ensures we don't lose context at boundaries

What happens:
1. Takes a list of documents (text files)
2. Splits each document into smaller pieces (chunks)
3. Adds overlap between chunks to preserve context
4. Returns a list of chunks with metadata

Flow: LOAD (documents) -> TRANSFORM (split into chunks) -> returns chunks

Example:
    Document: "This is a very long document with 2000 characters..."
    Chunk size: 800, Overlap: 120
    
    Result:
    - Chunk 0: characters 0-800
    - Chunk 1: characters 680-1480 (starts at 800-120=680 due to overlap)
    - Chunk 2: characters 1360-2160
    - etc.
"""

import re
import sys
from typing import List, Dict

# Suppress warnings during chunking
import warnings
warnings.filterwarnings("ignore")


def transform_to_chunks(documents: List[Dict], chunk_size: int = 800, overlap: int = 120) -> List[Dict]:
    """
    Split documents into smaller chunks with overlap.
    
    This function takes large text documents and breaks them into smaller pieces.
    The overlap ensures that important context isn't lost at chunk boundaries.
    
    Args:
        documents: List of document dictionaries, each with:
                   - filename: Name of the file
                   - content: Text content of the file
        chunk_size: Maximum characters per chunk (default: 800)
                    Larger = more context per chunk, but fewer chunks
        overlap: Characters to overlap between chunks (default: 120)
                 This ensures context isn't lost at boundaries
                 Example: If chunk 1 ends at char 800, chunk 2 starts at char 680
    
    Returns:
        List of chunk dictionaries, each containing:
        - chunk_id: Unique identifier (e.g., "file.txt::chunk_0")
        - source_file: Original filename
        - chunk_text: Text content of the chunk
    
    Example:
        documents = [
            {"filename": "doc.txt", "content": "Very long text..."}
        ]
        chunks = transform_to_chunks(documents, chunk_size=800, overlap=120)
        # Returns: [
        #     {"chunk_id": "doc.txt::chunk_0", "source_file": "doc.txt", "chunk_text": "..."},
        #     {"chunk_id": "doc.txt::chunk_1", "source_file": "doc.txt", "chunk_text": "..."},
        #     ...
        # ]
    """
    chunks = []
    
    # Process each document
    for doc in documents:
        text = doc["content"]      # Get the text content
        filename = doc["filename"]  # Get the filename
        
        # For very large documents, show progress
        text_len = len(text)
        is_large = text_len > 100000  # More than 100K characters
        
        if is_large:
            print(f"    Processing large document ({text_len:,} characters)...")
        
        # Clean up text: replace multiple spaces/newlines with single space
        # This makes the text more uniform and easier to process
        # For large texts, do this more efficiently
        if is_large:
            # For large texts, use a more efficient approach
            text = ' '.join(text.split())
        else:
            text = re.sub(r'\s+', ' ', text.strip())
        
        # Skip empty documents
        if not text:
            continue
        
        # If document is small enough, use it as a single chunk
        if len(text) <= chunk_size:
            chunks.append({
                "chunk_id": f"{filename}::chunk_0",  # Unique ID: filename + chunk number
                "source_file": filename,              # Which file this came from
                "chunk_text": text                    # The actual text
            })
        else:
            # Document is large, split it into multiple chunks
            start = 0      # Start position in text
            chunk_num = 0  # Chunk counter
            total_chars = len(text)
            estimated_chunks = (total_chars // (chunk_size - overlap)) + 1
            
            if is_large:
                print(f"    Splitting into ~{estimated_chunks} chunks...")
            
            # Keep creating chunks until we've processed all text
            while start < len(text):
                # Calculate end position
                end = start + chunk_size
                
                # Try to break at sentence boundary (better than mid-sentence)
                # Look for period (.) near the end of chunk
                # For large documents, limit the search range for performance
                if end < len(text):
                    search_start = max(start, end - 200)  # Only search last 200 chars
                    sentence_end = text.rfind('.', search_start, end)
                    if sentence_end > start:
                        end = sentence_end + 1  # Include the period
                
                # Extract chunk text
                chunk_text = text[start:end].strip()
                
                # Only add non-empty chunks
                if chunk_text:
                    chunks.append({
                        "chunk_id": f"{filename}::chunk_{chunk_num}",
                        "source_file": filename,
                        "chunk_text": chunk_text
                    })
                    chunk_num += 1
                    
                    # Show progress for large documents
                    if is_large and chunk_num % 50 == 0:
                        progress = (end / total_chars) * 100
                        print(f"    Progress: {chunk_num} chunks created ({progress:.1f}%)")
                
                # Move start position forward, accounting for overlap
                # Overlap ensures context isn't lost between chunks
                # Example: If chunk ends at 800, next chunk starts at 800-120=680
                start = end - overlap
                
                # Safety check: don't go backwards
                if start <= 0:
                    start = end
        
        if is_large:
            print(f"    ✓ Created {len([c for c in chunks if c['source_file'] == filename])} chunks from {filename}")
    
    return chunks
