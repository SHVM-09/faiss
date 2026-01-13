"""
STEP 3: EMBED
=============
This step converts text chunks into numerical vectors (embeddings).

What are embeddings?
- Embeddings are arrays of numbers that represent text
- Similar texts have similar embeddings
- We can compare embeddings to find similar documents

What happens:
1. Takes a list of text chunks
2. Uses a pre-trained model to convert each chunk to a vector
3. Each vector is 1024 numbers (dimensions) for BAAI/bge-large-en-v1.5
4. Vectors are normalized for cosine similarity

Flow: TRANSFORM (chunks) -> EMBED (convert to vectors) -> returns vectors
"""

from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")


class Embedder:
    """
    Wrapper for the embedding model.
    
    CURRENT MODEL: 'BAAI/bge-large-en-v1.5' (BEST ACCURACY)
    - State-of-the-art accuracy for English text
    - Produces 1024-dimensional vectors
    - Best for semantic search accuracy
    
    ALTERNATIVES (commented below):
    - FAST: 'sentence-transformers/all-MiniLM-L6-v2' (384 dim, ~80MB) - fastest
    - BALANCED: 'sentence-transformers/all-mpnet-base-v2' (768 dim, ~420MB) - good balance
    - See MODELS.md for more alternatives
    """
    
    def __init__(self):
        """Initialize the embedder (model loads lazily)."""
        self.model = None
        # BEST ACCURACY MODEL (current)
        self.dimension = 1024  # Output dimension of BAAI/bge-large-en-v1.5
        
        # ALTERNATIVE: Fast model (commented out - uncomment to use)
        # self.dimension = 384  # Output dimension of all-MiniLM-L6-v2
        
        # ALTERNATIVE: Balanced model (commented out - uncomment to use)
        # self.dimension = 768  # Output dimension of all-mpnet-base-v2
    
    def load(self):
        """
        Load the embedding model from HuggingFace.
        
        Note: First time will download the model (~1.3GB for bge-large-en-v1.5).
        Subsequent calls reuse the cached model.
        
        Download tips:
        - Models are cached after first download (no re-download needed)
        - Download happens in background with progress bar
        """
        if self.model is None:
            print("="*60)
            print("Loading embedding model: BAAI/bge-large-en-v1.5")
            print("="*60)
            print("📥 Downloading model (~1.3GB) - this may take a few minutes...")
            print("   (Model will be cached for future use)")
            print("   Progress bar will show download status")
            print("-"*60)
            
            # ====================================================================
            # BEST ACCURACY MODEL (current)
            # ====================================================================
            # BAAI/bge-large-en-v1.5: State-of-the-art accuracy for English
            # - 1024 dimensions
            # - ~1.3GB model size
            # - Best accuracy on benchmarks
            # - Slower inference but most accurate
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            
            # ====================================================================
            # ALTERNATIVE: Fast model (commented out - uncomment to use)
            # ====================================================================
            # sentence-transformers/all-MiniLM-L6-v2: Fast and efficient
            # - 384 dimensions
            # - ~80MB model size
            # - Fast inference
            # - Good accuracy for general use
            # self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # ====================================================================
            # ALTERNATIVE: Balanced model (commented out - uncomment to use)
            # ====================================================================
            # sentence-transformers/all-mpnet-base-v2: Best balance
            # - 768 dimensions
            # - ~420MB model size
            # - Excellent accuracy, still reasonably fast
            # self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            print("-"*60)
            print("✓ Model loaded successfully!")
            print("="*60)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert text chunks to vector embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (num_texts, 1024) for bge-large-en-v1.5
            (or 384 for all-MiniLM-L6-v2, or 768 for all-mpnet-base-v2 if using those models)
            Each row is a vector representing one text
            
        Example:
            texts = ["Hello world", "Machine learning"]
            vectors = embedder.embed(texts)
            # Returns: array([[0.1, 0.2, ...], [0.3, 0.4, ...]], shape=(2, 1024))
        """
        # Load model if not already loaded
        if self.model is None:
            self.load()
        
        # Convert texts to embeddings
        # normalize_embeddings=True: Makes vectors unit length (for cosine similarity)
        # convert_to_numpy=True: Returns numpy array (not PyTorch tensor)
        # show_progress_bar=True: Shows progress for large batches
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True,       # Return numpy array
            show_progress_bar=True       # Show progress
        )
        
        # Ensure float32 dtype (FAISS requires this)
        return embeddings.astype(np.float32)
