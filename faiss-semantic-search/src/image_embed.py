"""
IMAGE EMBEDDING
===============
This module handles image embeddings using CLIP (Contrastive Language-Image Pre-training).

What is CLIP?
- CLIP can understand both images and text in the same space
- You can search images with text queries (e.g., "engine block")
- Images are converted to vectors just like text

What happens:
1. Takes an image (PIL Image or file path)
2. Uses CLIP to convert image to a vector
3. Returns a 1024-dimensional vector (for CLIP-ViT-H-14)
"""

from typing import List, Union
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None
    raise ImportError("Pillow (PIL) not installed. Run: pip install Pillow")

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    torch = None
    CLIPProcessor = None
    CLIPModel = None


class ImageEmbedder:
    """
    Wrapper for CLIP model to embed images.
    
    CURRENT MODEL: 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K' (BEST ACCURACY)
    - State-of-the-art accuracy for image-text search
    - Produces 1024-dimensional vectors
    - Trained on 2B image-text pairs
    - Best for image search accuracy
    
    ALTERNATIVES (commented below):
    - FAST: 'openai/clip-vit-base-patch32' (512 dim, ~500MB) - fastest
    - BALANCED: 'openai/clip-vit-large-patch14' (768 dim, ~890MB) - good balance
    - See MODELS.md for more alternatives
    """
    
    def __init__(self):
        """Initialize the image embedder (model loads lazily)."""
        self.model = None
        self.processor = None
        # BEST ACCURACY MODEL (current)
        self.dimension = 1024  # Output dimension of CLIP-ViT-H-14
        
        # ALTERNATIVE: Fast model (commented out - uncomment to use)
        # self.dimension = 512  # Output dimension of clip-vit-base-patch32
        
        # ALTERNATIVE: Balanced model (commented out - uncomment to use)
        # self.dimension = 768  # Output dimension of clip-vit-large-patch14
    
    def load(self):
        """
        Load the CLIP model from HuggingFace.
        
        Note: First time will download the model (~2.5GB for CLIP-ViT-H-14).
        Subsequent calls reuse the cached model.
        
        Download tips:
        - Models are cached after first download (no re-download needed)
        - Download happens in background with progress bar
        """
        if CLIPModel is None:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        if self.model is None:
            print("="*60)
            print("Loading CLIP model: CLIP-ViT-H-14-laion2B-s32B-b79K")
            print("="*60)
            print("📥 Downloading model (~2.5GB) - this may take several minutes...")
            print("   (Model will be cached for future use)")
            print("   Progress bar will show download status")
            print("-"*60)
            
            # ====================================================================
            # BEST ACCURACY MODEL (current)
            # ====================================================================
            # laion/CLIP-ViT-H-14-laion2B-s32B-b79K: State-of-the-art accuracy
            # - 1024 dimensions
            # - ~2.5GB model size
            # - Trained on 2B image-text pairs
            # - Best accuracy on benchmarks
            # - Slower inference but most accurate
            self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            
            # ====================================================================
            # ALTERNATIVE: Fast model (commented out - uncomment to use)
            # ====================================================================
            # openai/clip-vit-base-patch32: Fast and efficient
            # - 512 dimensions
            # - ~500MB model size
            # - Fast inference
            # - Good accuracy for general use
            # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # ====================================================================
            # ALTERNATIVE: Balanced model (commented out - uncomment to use)
            # ====================================================================
            # openai/clip-vit-large-patch14: Best balance
            # - 768 dimensions
            # - ~890MB model size
            # - Excellent accuracy, still reasonably fast
            # self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            # Set to evaluation mode (faster, no gradients)
            self.model.eval()
            print("-"*60)
            print("✓ CLIP model loaded successfully!")
            print("="*60)
    
    def embed_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Convert an image to a vector embedding.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            numpy array of shape (1024,) for CLIP-ViT-H-14
            (or 512 for clip-vit-base-patch32, or 768 for clip-vit-large-patch14 if using those models)
            The image embedding vector
            
        Example:
            img = Image.open("engine.png")
            vector = embedder.embed_image(img)
            # Returns: array([0.1, 0.2, ...], shape=(1024,))
        """
        if self.model is None:
            self.load()
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Process image and get embedding
        with torch.no_grad():  # No gradients needed (faster)
            # Process image for CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Get image embedding
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy and ensure float32
            embedding = image_features[0].numpy().astype(np.float32)
        
        return embedding
    
    def embed_images(self, images: List[Union[Image.Image, str]]) -> np.ndarray:
        """
        Convert multiple images to vectors (batch processing).
        
        Args:
            images: List of PIL Images or image file paths
            
        Returns:
            numpy array of shape (num_images, 1024) for CLIP-ViT-H-14
            (or (num_images, 512) for clip-vit-base-patch32, or (num_images, 768) for clip-vit-large-patch14)
        """
        if self.model is None:
            self.load()
        
        # Load images if paths provided
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img))
            else:
                pil_images.append(img)
        
        # Process all images at once (batch)
        with torch.no_grad():
            inputs = self.processor(images=pil_images, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embeddings = image_features.numpy().astype(np.float32)
        
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to a vector (for searching images with text).
        
        Args:
            text: Text query (e.g., "engine block")
            
        Returns:
            numpy array of shape (1024,) for CLIP-ViT-H-14
            (or 512 for clip-vit-base-patch32, or 768 for clip-vit-large-patch14 if using those models)
            Text embedding in same space as images
            
        This allows you to search images using text queries!
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
