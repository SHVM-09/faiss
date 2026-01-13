"""
STEP 4: STORE
=============
This step saves vectors to a FAISS index for fast similarity search.

What is FAISS?
- FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
- It can search through millions of vectors in milliseconds
- IndexFlatIP uses Inner Product (cosine similarity for normalized vectors)

What happens:
1. Creates separate FAISS indices for text and images
2. Adds vectors to the appropriate index
3. Stores metadata (chunk text, image info) separately
4. Saves everything to disk for persistence

Flow: EMBED (vectors) -> STORE (save to FAISS) -> index ready for search

How it works:
- Text vectors (1024 dimensions) go to text_index
- Image vectors (1024 dimensions) go to image_index
- Metadata is stored separately in JSON files
- When searching, FAISS returns vector IDs, which we use to look up metadata
"""

import os
import json
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss not installed. Run: conda install -c conda-forge faiss-cpu")

from typing import List, Dict


class VectorStore:
    """
    Manages FAISS indices and metadata for both text and images.
    
    This class handles:
    - Creating FAISS indices for text and images
    - Storing vectors in the indices
    - Storing metadata (text, image info) separately
    - Loading and saving indices to disk
    
    Why separate indices?
    - Text vectors are 1024 dimensions (from BAAI/bge-large-en-v1.5)
    - Image vectors are 1024 dimensions (from CLIP-ViT-H-14)
    - FAISS indices must have consistent dimensions, so we need separate indices
    - Note: Both currently use 1024 dimensions, but kept separate for flexibility
    """
    
    def __init__(self):
        """Initialize the vector store with separate indices for text and images."""
        # Text index: stores text embeddings
        self.text_index = None
        self.text_metadata = []  # Stores actual text content for each vector
        # BEST ACCURACY MODEL (current)
        self.text_dimension = 1024  # BAAI/bge-large-en-v1.5 dimension
        
        # ALTERNATIVE: Fast model (commented out - uncomment if using fast model)
        # self.text_dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # ALTERNATIVE: Balanced model (commented out - uncomment if using balanced model)
        # self.text_dimension = 768  # all-mpnet-base-v2 dimension
        
        # Image index: stores image embeddings
        self.image_index = None
        self.image_metadata = []  # Stores image info for each vector
        # BEST ACCURACY MODEL (current)
        self.image_dimension = 1024  # CLIP-ViT-H-14 dimension
        
        # ALTERNATIVE: Fast model (commented out - uncomment if using fast model)
        # self.image_dimension = 512  # clip-vit-base-patch32 dimension
        
        # ALTERNATIVE: Balanced model (commented out - uncomment if using balanced model)
        # self.image_dimension = 768  # clip-vit-large-patch14 dimension
        
        # File paths for saving/loading
        self.text_index_path = "./data/text_index.faiss"
        self.image_index_path = "./data/image_index.faiss"
        self.text_metadata_path = "./data/text_metadata.json"
        self.image_metadata_path = "./data/image_metadata.json"
        
        # Create data directory if it doesn't exist
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./data/images", exist_ok=True)
    
    def initialize_text_index(self, dimension: int = 1024):
        """
        Create a new empty FAISS index for text vectors.
        
        Args:
            dimension: Size of each vector (1024 for BAAI/bge-large-en-v1.5)
            
        What is IndexFlatIP?
        - IndexFlatIP = Index Flat Inner Product
        - Inner Product = cosine similarity when vectors are normalized
        - "Flat" means exact search (no approximation, slower but accurate)
        """
        self.text_dimension = dimension
        self.text_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        self.text_metadata = []
    
    def initialize_image_index(self, dimension: int = 1024):
        """
        Create a new empty FAISS index for image vectors.
        
        Args:
            dimension: Size of each vector (1024 for CLIP-ViT-H-14)
        """
        self.image_dimension = dimension
        self.image_index = faiss.IndexFlatIP(dimension)
        self.image_metadata = []
    
    def load(self) -> bool:
        """
        Load existing indices and metadata from disk.
        
        This allows the index to persist between server restarts.
        You don't need to re-index everything each time!
        
        Returns:
            True if at least one index loaded, False if none exist
        """
        loaded = False
        
        # Load text index if it exists
        if os.path.exists(self.text_index_path):
            try:
                self.text_index = faiss.read_index(self.text_index_path)
                self.text_dimension = self.text_index.d  # Get dimension from loaded index
                with open(self.text_metadata_path, 'r', encoding='utf-8') as f:
                    self.text_metadata = json.load(f)
                loaded = True
            except Exception:
                pass
        
        # Load image index if it exists
        if os.path.exists(self.image_index_path):
            try:
                self.image_index = faiss.read_index(self.image_index_path)
                self.image_dimension = self.image_index.d
                with open(self.image_metadata_path, 'r', encoding='utf-8') as f:
                    self.image_metadata = json.load(f)
                loaded = True
            except Exception:
                pass
        
        return loaded
    
    def store_text(self, vectors: np.ndarray, chunks: List[Dict]):
        """
        Add text vectors and their metadata to the text index.
        
        Args:
            vectors: numpy array of shape (num_vectors, 1024) - text embeddings
            chunks: List of chunk dictionaries with text and metadata
            
        How it works:
        1. If index doesn't exist, create it
        2. Add vectors to FAISS index (fast similarity search)
        3. Store corresponding metadata (text, source file) in list
        4. Metadata index matches vector index (metadata[0] = vectors[0])
        """
        # Initialize text index if it doesn't exist
        if self.text_index is None:
            self.initialize_text_index(vectors.shape[1])
        
        # Add vectors to FAISS index
        # Vectors are already normalized from embedding step
        self.text_index.add(vectors)
        
        # Store metadata for each chunk
        # The order must match the vector order!
        for chunk in chunks:
            self.text_metadata.append({
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "chunk_text": chunk["chunk_text"],
                "type": "text"
            })
    
    def store_images(self, vectors: np.ndarray, images: List[Dict]):
        """
        Add image vectors and their metadata to the image index.
        
        Args:
            vectors: numpy array of shape (num_images, 1024) - image embeddings
            images: List of image dictionaries with metadata
            
        How it works:
        1. If index doesn't exist, create it
        2. Add vectors to FAISS index
        3. Store corresponding metadata (image path, source file) in list
        """
        # Initialize image index if it doesn't exist
        if self.image_index is None:
            self.initialize_image_index(vectors.shape[1])
        
        # Add vectors to FAISS index
        self.image_index.add(vectors)
        
        # Store metadata for each image
        for img_data in images:
            self.image_metadata.append({
                "image_id": f"{img_data['source_file']}::img_{img_data['image_index']}",
                "source_file": img_data["source_file"],
                "image_index": img_data["image_index"],
                "image_path": img_data.get("image_path"),  # Path to saved image file
                "type": "image"
            })
    
    def save(self):
        """
        Save all indices and metadata to disk.
        
        This allows the indices to persist between server restarts.
        Next time you start the server, it will load these indices automatically.
        """
        # Save text index
        if self.text_index is not None:
            faiss.write_index(self.text_index, self.text_index_path)
            with open(self.text_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.text_metadata, f, indent=2, ensure_ascii=False)
        
        # Save image index
        if self.image_index is not None:
            faiss.write_index(self.image_index, self.image_index_path)
            with open(self.image_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.image_metadata, f, indent=2, ensure_ascii=False)
    
    def clear(self):
        """
        Clear all indices and delete saved files.
        
        Use this to start fresh or remove all indexed documents and images.
        """
        # Clear in-memory data
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []
        
        # Delete files from disk
        for path in [self.text_index_path, self.image_index_path, 
                     self.text_metadata_path, self.image_metadata_path]:
            if os.path.exists(path):
                os.remove(path)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the stored vectors.
        
        Returns:
            Dictionary with:
            - text_vector_count: Number of text vectors in index
            - image_vector_count: Number of image vectors in index
            - text_dimension: Text vector dimension (1024 for bge-large-en-v1.5)
            - image_dimension: Image vector dimension (1024 for CLIP-ViT-H-14)
            - text_model: Current text model name
            - image_model: Current image model name
        """
        return {
            "text_vector_count": int(self.text_index.ntotal) if self.text_index else 0,
            "image_vector_count": int(self.image_index.ntotal) if self.image_index else 0,
            "text_dimension": int(self.text_dimension) if self.text_index else None,
            "image_dimension": int(self.image_dimension) if self.image_index else None,
            "text_model": "BAAI/bge-large-en-v1.5",  # Current text model
            "image_model": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  # Current image model
        }
