"""
PRODUCTION VECTOR STORE WITH IVF-PQ
====================================
Production-ready vector store with:
- Per-namespace IVF-PQ indices (IndexIVFPQ wrapped in IndexIDMap2)
- SQLite metadata store with embedding storage
- Smart retraining (20% growth threshold)
- Atomic writes
- Deletion support
- Namespace isolation
"""

import os
import json
import sqlite3
import numpy as np
import shutil
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    import faiss
except ImportError:
    raise ImportError("faiss not installed. Run: conda install -c conda-forge faiss-cpu")


def _choose_pq_m(dimension: int, preferred_m: int = 64) -> int:
    """
    Choose PQ subquantizer count (m) that divides dimension evenly.
    
    Args:
        dimension: Vector dimension
        preferred_m: Preferred m value (default 64)
    
    Returns:
        Valid m value (32, 48, or 64)
    """
    valid_ms = [32, 48, 64]
    for m in [preferred_m] + [x for x in valid_ms if x != preferred_m]:
        if dimension % m == 0:
            return m
    # Fallback: use largest valid divisor
    return max([m for m in valid_ms if dimension % m == 0] or [32])


def _compute_nlist(ntotal: int, min_nlist: int = 256, max_nlist: int = 8192) -> int:
    """
    Compute nlist (number of clusters) based on dataset size.
    
    FAISS training requirement:
    - Minimum: need at least nlist training points (hard requirement)
    - Recommended: need at least 39*nlist training points for good quality
    
    So for a dataset of size N:
    - Maximum nlist = N (hard limit)
    - Ideal nlist ≈ N/39 (for good quality)
    - Balance: use sqrt(N) but ensure N >= 39*nlist
    
    Args:
        ntotal: Total number of vectors (or expected)
        min_nlist: Minimum nlist (default 256, but adjusted for smaller datasets)
        max_nlist: Maximum nlist (default 8192)
    
    Returns:
        nlist value (respects FAISS training requirements)
    """
    if ntotal == 0:
        return 256  # Default for empty index
    
    # Calculate based on sqrt (good balance for most cases)
    nlist_sqrt = int(np.sqrt(ntotal))
    
    # Calculate based on FAISS training requirement (39x rule for good quality)
    # Maximum nlist that allows 39x training points
    nlist_max_by_training = max(1, int(ntotal / 39))
    
    # Use the smaller of sqrt and training-based calculation
    # This ensures we have enough training points
    nlist = min(nlist_sqrt, nlist_max_by_training)
    
    # Apply min/max bounds intelligently
    # For smaller datasets (< 10K), allow smaller nlist (down to 16)
    # For larger datasets, use standard min_nlist
    if ntotal < 10000:
        # For smaller datasets, allow nlist as low as 16
        # This prevents the "need 9984 training points" warning
        nlist = max(16, min(max_nlist, nlist))
    else:
        # For larger datasets, use standard min_nlist
        nlist = max(min_nlist, min(max_nlist, nlist))
    
    # Ensure nlist doesn't exceed ntotal (FAISS hard requirement)
    nlist = min(nlist, ntotal)
    
    # Ensure at least 1
    nlist = max(1, nlist)
    
    return nlist


class VectorStoreV2:
    """
    Production-ready vector store with per-namespace IVF-PQ indices.
    """
    
    def __init__(self, data_dir: str = "./data", namespace: str = "default"):
        """
        Initialize the vector store.
        
        Args:
            data_dir: Base directory for all data files
            namespace: Default namespace for vectors
        """
        self.data_dir = Path(data_dir)
        self.default_namespace = namespace
        
        # Directory structure
        self.metadata_dir = self.data_dir / "metadata"
        self.tmp_dir = self.data_dir / "tmp"
        
        # Create directories
        for dir_path in [self.metadata_dir, self.tmp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Shared metadata database
        self.metadata_db_path = self.metadata_dir / "metadata.db"
        
        # Per-namespace indices cache (loaded on demand)
        self._namespace_indices: Dict[str, Dict] = {}
        
        # Default IVF-PQ parameters
        self.default_nlist = None  # Computed dynamically
        self.default_m = 64
        self.default_nbits = 8
        self.default_nprobe = 16
        
        # Load existing data
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema including embedding storage."""
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # Check if embedding_blob column exists
            cursor.execute("PRAGMA table_info(vectors)")
            columns = [col[1] for col in cursor.fetchall()]
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    deleted INTEGER NOT NULL DEFAULT 0,
                    vector_type TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add embedding_blob column if it doesn't exist
            if 'embedding_blob' not in columns:
                cursor.execute("ALTER TABLE vectors ADD COLUMN embedding_blob BLOB")
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON vectors(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON vectors(chunk_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_namespace ON vectors(namespace)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_deleted ON vectors(deleted)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_type ON vectors(vector_type)")
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
            # Ensure directory exists and retry
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    deleted INTEGER NOT NULL DEFAULT 0,
                    vector_type TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON vectors(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON vectors(chunk_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_namespace ON vectors(namespace)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_deleted ON vectors(deleted)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_type ON vectors(vector_type)")
            conn.commit()
            conn.close()
    
    def _get_namespace_dir(self, namespace: str) -> Path:
        """Get directory for a namespace."""
        return self.data_dir / namespace
    
    def _get_namespace_manifest_path(self, namespace: str) -> Path:
        """Get manifest path for a namespace."""
        return self._get_namespace_dir(namespace) / "manifest.json"
    
    def _get_namespace_index_path(self, namespace: str, vector_type: str) -> Path:
        """Get index path for a namespace and vector type."""
        return self._get_namespace_dir(namespace) / f"{vector_type}_index.faiss"
    
    def _load_namespace_indices(self, namespace: str) -> Dict:
        """
        Load indices for a namespace (cached).
        
        Returns:
            Dict with 'text_index', 'image_index', 'text_manifest', 'image_manifest'
        """
        if namespace in self._namespace_indices:
            return self._namespace_indices[namespace]
        
        namespace_dir = self._get_namespace_dir(namespace)
        namespace_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            'text_index': None,
            'image_index': None,
            'text_manifest': {},
            'image_manifest': {}
        }
        
        # Load text index
        text_index_path = self._get_namespace_index_path(namespace, "text")
        text_manifest_path = self._get_namespace_manifest_path(namespace)
        
        if text_index_path.exists():
            try:
                result['text_index'] = faiss.read_index(str(text_index_path))
            except Exception as e:
                print(f"Warning: Could not load text index for namespace '{namespace}': {e}")
        
        # Load image index
        image_index_path = self._get_namespace_index_path(namespace, "image")
        if image_index_path.exists():
            try:
                result['image_index'] = faiss.read_index(str(image_index_path))
            except Exception as e:
                print(f"Warning: Could not load image index for namespace '{namespace}': {e}")
        
        # Load manifest
        if text_manifest_path.exists():
            try:
                with open(text_manifest_path, 'r') as f:
                    manifest = json.load(f)
                    result['text_manifest'] = manifest.get('text', {})
                    result['image_manifest'] = manifest.get('image', {})
            except Exception as e:
                print(f"Warning: Could not load manifest for namespace '{namespace}': {e}")
        
        self._namespace_indices[namespace] = result
        return result
    
    def _get_next_vector_id(self) -> int:
        """Get next available vector ID."""
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(vector_id) FROM vectors")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] is not None:
                return result[0] + 1
            return 1
        except Exception as e:
            print(f"Warning: Could not get next vector ID: {e}")
            return 1
    
    def _create_flat_index(self, dimension: int):
        """Create a simple IndexFlatIP wrapped in IndexIDMap2 (for small datasets)."""
        base_index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap2(base_index)
        return index
    
    def _create_ivfpq_index(self, dimension: int, nlist: int, m: int, nbits: int, nprobe: int = 16):
        """
        Create a new IndexIVFPQ wrapped in IndexIDMap2.
        
        Args:
            dimension: Vector dimension
            nlist: Number of clusters
            m: Number of PQ subquantizers
            nbits: Bits per subquantizer
            nprobe: Number of clusters to probe during search
        
        Returns:
            IndexIDMap2 wrapping IndexIVFPQ
        """
        # Ensure dimension is divisible by m
        m = _choose_pq_m(dimension, m)
        
        # Ensure nlist is reasonable (FAISS requirement: nlist <= number of training vectors)
        # But we'll validate this at training time, not here
        
        # Create quantizer (flat index for centroids) - always create fresh
        quantizer = faiss.IndexFlatIP(dimension)
        
        # Create IVF-PQ index with specified nlist
        # IMPORTANT: nlist must be set correctly here - this is what FAISS uses for clustering
        ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        ivfpq.nprobe = nprobe
        ivfpq.metric_type = faiss.METRIC_INNER_PRODUCT
        
        # Verify nlist was set correctly
        if ivfpq.nlist != nlist:
            raise ValueError(f"Failed to create index with nlist={nlist}, got {ivfpq.nlist} instead")
        
        # Wrap with IDMap for stable IDs
        index = faiss.IndexIDMap2(ivfpq)
        
        return index
    
    def _get_existing_ntotal(self, namespace: str, vector_type: str) -> int:
        """Get existing vector count for a namespace and type."""
        indices = self._load_namespace_indices(namespace)
        index = indices.get(f'{vector_type}_index')
        if index is None:
            return 0
        return index.ntotal
    
    def _is_index_trained(self, index) -> bool:
        """Check if an IVF index is trained."""
        if index is None:
            return False
        # Unwrap IndexIDMap2 to get base index
        if isinstance(index, faiss.IndexIDMap2):
            base_index = faiss.downcast_index(index.index)
            if isinstance(base_index, faiss.IndexIVFPQ):
                return base_index.is_trained
        return False
    
    def _should_retrain(self, existing_ntotal: int, new_count: int, index, vector_type: str) -> Tuple[bool, str]:
        """
        Determine if retraining is needed.
        
        Returns:
            (should_retrain: bool, reason: str)
        """
        # First time or no index
        if index is None or existing_ntotal == 0:
            return True, "first_train"
        
        # Index not trained
        if not self._is_index_trained(index):
            return True, "untrained"
        
        # Growth >= 20%
        if existing_ntotal > 0:
            growth_ratio = new_count / existing_ntotal
            if growth_ratio >= 0.20:
                return True, "growth>=20%"
        
        return False, None
    
    def _sample_training_vectors(self, namespace: str, vector_type: str, new_vectors: np.ndarray, 
                                 target_count: int = 100000) -> np.ndarray:
        """
        Sample training vectors from existing (SQLite) + new vectors.
        
        Args:
            namespace: Namespace
            vector_type: 'text' or 'image'
            new_vectors: New vectors to add
            target_count: Target number of training vectors
        
        Returns:
            Array of training vectors
        """
        training_vectors = []
        
        # Get existing vectors from SQLite
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT embedding_blob FROM vectors 
            WHERE namespace = ? AND vector_type = ? AND deleted = 0 AND embedding_blob IS NOT NULL
        """, (namespace, vector_type))
        
        existing_vectors = []
        for row in cursor.fetchall():
            if row[0] is not None:
                vec = np.frombuffer(row[0], dtype=np.float32)
                existing_vectors.append(vec)
        
        conn.close()
        
        # Combine existing and new
        all_vectors = existing_vectors.copy()
        if len(new_vectors) > 0:
            # Convert new_vectors to list of arrays if it's a 2D array
            if isinstance(new_vectors, np.ndarray) and len(new_vectors.shape) == 2:
                for i in range(len(new_vectors)):
                    all_vectors.append(new_vectors[i])
            else:
                all_vectors.extend(new_vectors)
        
        if len(all_vectors) == 0:
            return np.array([])
        
        # Sample randomly
        sample_size = min(target_count, len(all_vectors))
        sampled = random.sample(all_vectors, sample_size)
        
        # Convert to numpy array
        if len(sampled) == 0:
            return np.array([])
        
        # Get dimension from first vector
        dimension = len(sampled[0]) if hasattr(sampled[0], '__len__') else (new_vectors.shape[1] if len(new_vectors) > 0 else 1024)
        
        # Ensure all vectors are numpy arrays
        sampled_arrays = []
        for vec in sampled:
            if isinstance(vec, np.ndarray):
                sampled_arrays.append(vec)
            else:
                sampled_arrays.append(np.array(vec, dtype=np.float32))
        
        training_array = np.array(sampled_arrays, dtype=np.float32)
        if len(training_array.shape) == 1:
            training_array = training_array.reshape(1, -1)
        
        return training_array
    
    def _rebuild_index(self, namespace: str, vector_type: str, dimension: int, 
                      nlist: int, m: int, nbits: int, nprobe: int):
        """
        Rebuild index by loading all active vectors from SQLite and re-adding them.
        
        Args:
            namespace: Namespace
            vector_type: 'text' or 'image'
            dimension: Vector dimension
            nlist, m, nbits, nprobe: IVF-PQ parameters (nlist will be recomputed based on actual data)
        
        Note: This method IGNORES the nlist parameter and recomputes it based on actual vector count.
        """
        # Clear any cached index for this namespace/type to ensure we create a fresh one
        indices = self._load_namespace_indices(namespace)
        if f'{vector_type}_index' in indices:
            del indices[f'{vector_type}_index']
        
        # Delete old index file from disk to ensure we create a completely fresh one
        index_path = self._get_namespace_index_path(namespace, vector_type)
        if index_path.exists():
            try:
                index_path.unlink()
                print(f"  Deleted old {vector_type} index file to ensure fresh creation")
            except Exception as e:
                print(f"  Warning: Could not delete old index file: {e}")
        
        # Get all active vectors from SQLite FIRST
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT vector_id, embedding_blob FROM vectors 
            WHERE namespace = ? AND vector_type = ? AND deleted = 0 AND embedding_blob IS NOT NULL
        """, (namespace, vector_type))
        
        vectors = []
        vector_ids = []
        
        for row in cursor.fetchall():
            vector_id, embedding_blob = row
            if embedding_blob is not None:
                vec = np.frombuffer(embedding_blob, dtype=np.float32)
                vectors.append(vec)
                vector_ids.append(int(vector_id))
        
        conn.close()
        
        if len(vectors) == 0:
            # No vectors to add, create empty flat index
            index = self._create_flat_index(dimension)
            indices = self._load_namespace_indices(namespace)
            indices[f'{vector_type}_index'] = index
            return
        
        # Recompute nlist based on ACTUAL vector count
        # Always use the smart _compute_nlist which respects FAISS training requirements
        actual_nlist = _compute_nlist(len(vectors))
        
        # FAISS recommendation: need at least 39*nlist training points for good quality
        # Minimum requirement: at least nlist training points
        min_training_points = actual_nlist
        recommended_training_points = 39 * actual_nlist
        
        # Decision: Use IndexFlatIP only for very small datasets (< 500 vectors)
        # For medium datasets (500+), use IVF-PQ even if warnings appear (they're harmless)
        # IVF-PQ provides memory efficiency benefits even for smaller datasets
        min_vectors_for_ivfpq = 500  # Minimum vectors to attempt IVF-PQ
        
        if len(vectors) < min_vectors_for_ivfpq:
            # Use IndexFlatIP for very small datasets (< 500 vectors)
            # This avoids FAISS training warnings and provides better quality (100% recall)
            print(f"  Very small dataset ({len(vectors)} vectors) - using IndexFlatIP (exact search, 100% recall, no warnings)")
            print(f"  IVF-PQ will be used automatically when you have {min_vectors_for_ivfpq}+ vectors")
            index = self._create_flat_index(dimension)
            index_type = "IndexFlatIP"
            nlist = None  # Not applicable for Flat index
        elif len(vectors) < min_training_points:
            # Use IndexFlatIP if we don't have enough vectors for even minimum training
            print(f"  Insufficient vectors ({len(vectors)} < {min_training_points} required for {actual_nlist} clusters), using IndexFlatIP instead of IndexIVFPQ")
            index = self._create_flat_index(dimension)
            index_type = "IndexFlatIP"
            nlist = None  # Not applicable for Flat index
        else:
            # Use IVF-PQ (even if below recommended training size)
            # Warnings may appear but are harmless - index will still work correctly
            if len(vectors) < recommended_training_points:
                print(f"  Using IVF-PQ with {len(vectors)} vectors (nlist={actual_nlist}, recommended training: {recommended_training_points})")
                print(f"  Note: FAISS may show warnings about training points - these are informational and can be safely ignored")
            
            # Initialize index variable to None to ensure it's always defined
            index = None
            index_type = None
            nlist = None
            
            # Convert vectors list to numpy array for training
            vectors_array = np.array(vectors, dtype=np.float32)
            if len(vectors_array.shape) == 1:
                vectors_array = vectors_array.reshape(1, -1)
            
            # Sample for training (use all vectors if we have fewer than recommended)
            target_training = min(recommended_training_points, len(vectors))
            training_vectors = self._sample_training_vectors(namespace, vector_type, vectors_array, target_count=target_training)
            
            if len(training_vectors) < min_training_points:
                # Not enough training vectors, use IndexFlatIP
                print(f"  Not enough training vectors ({len(training_vectors)} < {min_training_points} required), using IndexFlatIP instead")
                index = self._create_flat_index(dimension)
                index_type = "IndexFlatIP"
                nlist = None
            else:
                # Update nlist to match actual computed value
                nlist = actual_nlist
                # Recreate index with correct nlist - IMPORTANT: use actual_nlist, not the parameter
                print(f"  Creating IVF-PQ index with nlist={nlist} (computed from {len(vectors)} vectors)")
                index = self._create_ivfpq_index(dimension, nlist, m, nbits, nprobe)
                
                # Verify the index was created with correct nlist
                base_index = faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap2) else index
                if isinstance(base_index, faiss.IndexIVFPQ):
                    actual_index_nlist = base_index.nlist
                    print(f"  ✓ Verified: Index created with nlist={actual_index_nlist} (expected {nlist})")
                    if actual_index_nlist != nlist:
                        print(f"  ⚠ ERROR: Index nlist mismatch! Expected {nlist}, got {actual_index_nlist}. This should not happen!")
                        raise ValueError(f"Index nlist mismatch: expected {nlist}, got {actual_index_nlist}")
                else:
                    print(f"  ⚠ WARNING: Index is not IndexIVFPQ, it's {type(base_index)}")
                
                # Train index
                if len(training_vectors) < recommended_training_points:
                    print(f"  Training {vector_type} index with {len(training_vectors)} vectors (nlist={nlist}, recommended: {recommended_training_points})...")
                    print(f"  ⚠ Note: Fewer training points than recommended - quality may be reduced (this is OK for smaller datasets)")
                else:
                    print(f"  Training {vector_type} index with {len(training_vectors)} vectors (nlist={nlist})...")
                
                try:
                    # Suppress FAISS clustering warnings - they come from C++ code
                    # Use os.dup2 to redirect stderr at file descriptor level (catches C++ output)
                    import sys
                    import os
                    
                    # Verify the index is correct before training
                    base_index = faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap2) else index
                    if isinstance(base_index, faiss.IndexIVFPQ):
                        print(f"  Training with actual nlist={base_index.nlist} (training vectors: {len(training_vectors)})")
                        if base_index.nlist != nlist:
                            raise ValueError(f"Index nlist mismatch before training: expected {nlist}, got {base_index.nlist}")
                    
                    # Redirect stderr to /dev/null at file descriptor level
                    # This catches C++ warnings that bypass Python's stderr
                    original_stderr_fd = sys.stderr.fileno()
                    saved_stderr_fd = os.dup(original_stderr_fd)  # Save original
                    
                    try:
                        # Open /dev/null and redirect stderr to it
                        with open(os.devnull, 'w') as null_file:
                            null_fd = null_file.fileno()
                            os.dup2(null_fd, original_stderr_fd)  # Redirect stderr
                            
                            try:
                                # Train index (all stderr output goes to /dev/null)
                                index.train(training_vectors)
                            finally:
                                # Restore original stderr
                                os.dup2(saved_stderr_fd, original_stderr_fd)
                    finally:
                        # Close saved fd
                        os.close(saved_stderr_fd)
                    index_type = "IVF_PQ"
                except Exception as e:
                    print(f"  Training failed: {e}, falling back to IndexFlatIP")
                    index = self._create_flat_index(dimension)
                    index_type = "IndexFlatIP"
                    nlist = None
        
        # Ensure index is defined before using it
        if index is None:
            raise RuntimeError(f"Index was not created for {vector_type} vectors (count: {len(vectors)})")
        
        # Add all vectors
        vectors_array = np.array(vectors, dtype=np.float32).reshape(-1, dimension)
        vector_ids_array = np.array(vector_ids, dtype=np.int64)
        index.add_with_ids(vectors_array, vector_ids_array)
        
        # Update cache
        indices = self._load_namespace_indices(namespace)
        indices[f'{vector_type}_index'] = index
        
        # Update manifest with correct index type
        old_manifest = indices.get(f'{vector_type}_manifest', {})
        manifest_info = old_manifest.copy()
        manifest_info.update({
            "index_type": index_type,
            "dimension": dimension,
            "trained": index_type == "IVF_PQ",
            "ntotal": int(index.ntotal),
            "last_train_ntotal": int(index.ntotal) if index_type == "IVF_PQ" else 0,
            "updated_at": datetime.now().isoformat()
        })
        if index_type == "IVF_PQ" and nlist is not None:
            manifest_info.update({
                "nlist": nlist,
                "m": m,
                "nbits": nbits,
                "nprobe": nprobe
            })
        if "created_at" not in manifest_info:
            manifest_info["created_at"] = datetime.now().isoformat()
        
        self._save_namespace_index(namespace, vector_type, index, manifest_info)
    
    def _get_index_type(self, index) -> str:
        """Detect index type (IndexFlatIP or IndexIVFPQ)."""
        if index is None:
            return "None"
        # Unwrap IndexIDMap2 to get base index
        if isinstance(index, faiss.IndexIDMap2):
            base_index = faiss.downcast_index(index.index)
            if isinstance(base_index, faiss.IndexIVFPQ):
                return "IVF_PQ"
            elif isinstance(base_index, faiss.IndexFlatIP):
                return "IndexFlatIP"
        return "Unknown"
    
    def _save_namespace_index(self, namespace: str, vector_type: str, index, manifest_info: Dict):
        """Save index and manifest for a namespace with atomic writes."""
        namespace_dir = self._get_namespace_dir(namespace)
        namespace_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = self._get_namespace_index_path(namespace, vector_type)
        manifest_path = self._get_namespace_manifest_path(namespace)
        
        # Save index atomically
        if index is not None:
            tmp_path = self.tmp_dir / f"{namespace}_{vector_type}_index.faiss.tmp"
            faiss.write_index(index, str(tmp_path))
            os.rename(str(tmp_path), str(index_path))
        
        # Load existing manifest or create new
        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            except:
                pass
        
        # Update manifest for this vector type
        # Detect actual index type if not specified
        if "index_type" not in manifest_info:
            manifest_info["index_type"] = self._get_index_type(index)
        manifest[vector_type] = manifest_info
        
        # Save manifest atomically
        tmp_manifest = self.tmp_dir / f"{namespace}_manifest.json.tmp"
        with open(tmp_manifest, 'w') as f:
            json.dump(manifest, f, indent=2)
        os.rename(str(tmp_manifest), str(manifest_path))
    
    def ensure_index(self, namespace: str, vector_type: str, dimension: int, 
                    nlist: Optional[int] = None, m: Optional[int] = None, 
                    nbits: Optional[int] = None, nprobe: Optional[int] = None) -> Dict:
        """
        Ensure index exists for namespace/type, creating or loading as needed.
        
        Returns:
            Dict with index info and whether it was created
        """
        indices = self._load_namespace_indices(namespace)
        index = indices.get(f'{vector_type}_index')
        
        # Get or compute parameters
        nlist = nlist or self.default_nlist or _compute_nlist(0)
        m = m or self.default_m
        nbits = nbits or self.default_nbits
        nprobe = nprobe or self.default_nprobe
        
        # Ensure m divides dimension
        m = _choose_pq_m(dimension, m)
        
        if index is None:
            # For new index, we'll decide on IVF-PQ vs Flat based on expected size
            # Default to IVF-PQ, but will fall back to Flat if needed during training
            index = self._create_ivfpq_index(dimension, nlist, m, nbits, nprobe)
            indices[f'{vector_type}_index'] = index
            return {
                'index': index,
                'created': True,
                'nlist': nlist,
                'm': m,
                'nbits': nbits,
                'nprobe': nprobe
            }
        
        # Check if we need to migrate from old index type
        base_index = faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap2) else index
        
        if not isinstance(base_index, faiss.IndexIVFPQ):
            # Migration needed - rebuild as IVF-PQ
            print(f"  Migrating {vector_type} index from {type(base_index).__name__} to IndexIVFPQ...")
            existing_ntotal = index.ntotal
            nlist = _compute_nlist(existing_ntotal, min_nlist=256, max_nlist=8192)
            self._rebuild_index(namespace, vector_type, dimension, nlist, m, nbits, nprobe)
            indices = self._load_namespace_indices(namespace)
            index = indices[f'{vector_type}_index']
            return {
                'index': index,
                'created': False,
                'migrated': True,
                'nlist': nlist,
                'm': m,
                'nbits': nbits,
                'nprobe': nprobe
            }
        
        return {
            'index': index,
            'created': False,
            'migrated': False,
            'nlist': nlist,
            'm': m,
            'nbits': nbits,
            'nprobe': nprobe
        }
    
    def store_text(self, vectors: np.ndarray, chunks: List[Dict], namespace: Optional[str] = None,
                  nlist: Optional[int] = None, m: Optional[int] = None, 
                  nbits: Optional[int] = None, nprobe: Optional[int] = None,
                  skip_training_check: bool = False) -> Dict:
        """
        Add text vectors with stable IDs, handling training/retraining as needed.
        
        Returns:
            Dict with training info: was_trained, retrain_reason, etc.
        """
        namespace = namespace or self.default_namespace
        dimension = vectors.shape[1]
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        vectors = vectors / norms
        
        # Ensure index exists
        index_info = self.ensure_index(namespace, "text", dimension, nlist, m, nbits, nprobe)
        index = index_info['index']
        
        # Get existing count
        existing_ntotal = index.ntotal
        new_count = len(vectors)
        
        # If skip_training_check is True, just store in SQLite without adding to index
        if skip_training_check:
            vector_ids = np.arange(self._get_next_vector_id(), self._get_next_vector_id() + new_count, dtype=np.int64)
            
            # Store in SQLite only (will be added to index later)
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            for i, (vector_id, chunk) in enumerate(zip(vector_ids, chunks)):
                metadata_json = json.dumps({
                    "chunk_text": chunk.get("chunk_text", ""),
                    "source_file": chunk.get("source_file", ""),
                    "type": "text"
                })
                embedding_blob = vectors[i].astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json, embedding_blob)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, (
                    int(vector_id),
                    chunk.get("source_file", ""),
                    chunk.get("chunk_id", f"chunk_{vector_id}"),
                    namespace,
                    "text",
                    metadata_json,
                    embedding_blob
                ))
            conn.commit()
            conn.close()
            
            return {
                "was_trained": False,
                "retrain_reason": None,
                "existing_ntotal": existing_ntotal,
                "new_count": new_count,
                "final_ntotal": existing_ntotal,  # Not added to index yet
                "index_type": "pending",
                "nlist": None,
                "m": None,
                "nbits": None,
                "nprobe": None
            }
        
        # Check if retraining is needed
        should_retrain, retrain_reason = self._should_retrain(existing_ntotal, new_count, index, "text")
        
        was_trained = False
        
        if should_retrain:
            # Rebuild index with training
            nlist = index_info['nlist'] or _compute_nlist(existing_ntotal + new_count)
            m = index_info['m']
            nbits = index_info['nbits']
            nprobe = index_info['nprobe']
            
            # Store new vectors temporarily in SQLite for rebuild
            vector_ids = np.arange(self._get_next_vector_id(), self._get_next_vector_id() + new_count, dtype=np.int64)
            
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            for i, (vector_id, chunk) in enumerate(zip(vector_ids, chunks)):
                metadata_json = json.dumps({
                    "chunk_text": chunk.get("chunk_text", ""),
                    "source_file": chunk.get("source_file", ""),
                    "type": "text"
                })
                embedding_blob = vectors[i].astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json, embedding_blob)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, (
                    int(vector_id),
                    chunk.get("source_file", ""),
                    chunk.get("chunk_id", f"chunk_{vector_id}"),
                    namespace,
                    "text",
                    metadata_json,
                    embedding_blob
                ))
            conn.commit()
            conn.close()
            
            # Rebuild index (will update manifest automatically)
            self._rebuild_index(namespace, "text", dimension, nlist, m, nbits, nprobe)
            indices = self._load_namespace_indices(namespace)
            index = indices['text_index']
            was_trained = True
        else:
            # Incremental add
            vector_ids = np.arange(self._get_next_vector_id(), self._get_next_vector_id() + new_count, dtype=np.int64)
            
            # Add to index
            index.add_with_ids(vectors, vector_ids)
            
            # Store in SQLite
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            for i, (vector_id, chunk) in enumerate(zip(vector_ids, chunks)):
                metadata_json = json.dumps({
                    "chunk_text": chunk.get("chunk_text", ""),
                    "source_file": chunk.get("source_file", ""),
                    "type": "text"
                })
                embedding_blob = vectors[i].astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json, embedding_blob)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, (
                    int(vector_id),
                    chunk.get("source_file", ""),
                    chunk.get("chunk_id", f"chunk_{vector_id}"),
                    namespace,
                    "text",
                    metadata_json,
                    embedding_blob
                ))
            conn.commit()
            conn.close()
            
            # Update cache
            indices = self._load_namespace_indices(namespace)
            indices['text_index'] = index
            
            # Update manifest
            old_manifest = indices.get('text_manifest', {})
            manifest_info = old_manifest.copy()
            manifest_info.update({
                "ntotal": int(index.ntotal),
                "updated_at": datetime.now().isoformat()
            })
            self._save_namespace_index(namespace, "text", index, manifest_info)
        
        # Get actual index type from manifest
        indices = self._load_namespace_indices(namespace)
        manifest = indices.get('text_manifest', {})
        actual_index_type = manifest.get('index_type', 'IVF_PQ')
        
        return {
            "was_trained": was_trained,
            "retrain_reason": retrain_reason if should_retrain else None,
            "existing_ntotal": existing_ntotal,
            "new_count": new_count,
            "final_ntotal": int(index.ntotal),
            "index_type": actual_index_type,
            "nlist": manifest.get('nlist') if actual_index_type == "IVF_PQ" else None,
            "m": manifest.get('m') if actual_index_type == "IVF_PQ" else None,
            "nbits": manifest.get('nbits') if actual_index_type == "IVF_PQ" else None,
            "nprobe": manifest.get('nprobe', 16) if actual_index_type == "IVF_PQ" else None
        }
    
    def store_images(self, vectors: np.ndarray, images: List[Dict], namespace: Optional[str] = None,
                    nlist: Optional[int] = None, m: Optional[int] = None,
                    nbits: Optional[int] = None, nprobe: Optional[int] = None,
                    skip_training_check: bool = False) -> Dict:
        """
        Add image vectors with stable IDs, handling training/retraining as needed.
        
        Returns:
            Dict with training info: was_trained, retrain_reason, etc.
        """
        namespace = namespace or self.default_namespace
        dimension = vectors.shape[1]
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        
        # Ensure index exists
        index_info = self.ensure_index(namespace, "image", dimension, nlist, m, nbits, nprobe)
        index = index_info['index']
        
        # Get existing count
        existing_ntotal = index.ntotal
        new_count = len(vectors)
        
        # If skip_training_check is True, just store in SQLite without adding to index
        if skip_training_check:
            vector_ids = np.arange(self._get_next_vector_id(), self._get_next_vector_id() + new_count, dtype=np.int64)
            
            # Store in SQLite only (will be added to index later)
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            for i, (vector_id, img_data) in enumerate(zip(vector_ids, images)):
                image_id = f"{img_data.get('source_file', 'unknown')}::img_{img_data.get('image_index', i)}"
                metadata_json = json.dumps({
                    "image_id": image_id,
                    "source_file": img_data.get("source_file", ""),
                    "image_index": img_data.get("image_index", i),
                    "image_path": img_data.get("image_path"),
                    "type": "image"
                })
                embedding_blob = vectors[i].astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json, embedding_blob)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, (
                    int(vector_id),
                    img_data.get("source_file", ""),
                    image_id,
                    namespace,
                    "image",
                    metadata_json,
                    embedding_blob
                ))
            conn.commit()
            conn.close()
            
            return {
                "was_trained": False,
                "retrain_reason": None,
                "existing_ntotal": existing_ntotal,
                "new_count": new_count,
                "final_ntotal": existing_ntotal,  # Not added to index yet
                "index_type": "pending",
                "nlist": None,
                "m": None,
                "nbits": None,
                "nprobe": None
            }
        
        # Check if retraining is needed
        should_retrain, retrain_reason = self._should_retrain(existing_ntotal, new_count, index, "image")
        
        was_trained = False
        
        if should_retrain:
            # Rebuild index with training
            nlist = index_info['nlist'] or _compute_nlist(existing_ntotal + new_count)
            m = index_info['m']
            nbits = index_info['nbits']
            nprobe = index_info['nprobe']
            
            # Store new vectors temporarily in SQLite for rebuild
            vector_ids = np.arange(self._get_next_vector_id(), self._get_next_vector_id() + new_count, dtype=np.int64)
            
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            for i, (vector_id, img_data) in enumerate(zip(vector_ids, images)):
                image_id = f"{img_data.get('source_file', 'unknown')}::img_{img_data.get('image_index', i)}"
                metadata_json = json.dumps({
                    "image_id": image_id,
                    "source_file": img_data.get("source_file", ""),
                    "image_index": img_data.get("image_index", i),
                    "image_path": img_data.get("image_path"),
                    "type": "image"
                })
                embedding_blob = vectors[i].astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json, embedding_blob)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, (
                    int(vector_id),
                    img_data.get("source_file", ""),
                    image_id,
                    namespace,
                    "image",
                    metadata_json,
                    embedding_blob
                ))
            conn.commit()
            conn.close()
            
            # Rebuild index (will update manifest automatically)
            self._rebuild_index(namespace, "image", dimension, nlist, m, nbits, nprobe)
            indices = self._load_namespace_indices(namespace)
            index = indices['image_index']
            was_trained = True
        else:
            # Incremental add
            vector_ids = np.arange(self._get_next_vector_id(), self._get_next_vector_id() + new_count, dtype=np.int64)
            
            # Add to index
            index.add_with_ids(vectors, vector_ids)
            
            # Store in SQLite
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            for i, (vector_id, img_data) in enumerate(zip(vector_ids, images)):
                image_id = f"{img_data.get('source_file', 'unknown')}::img_{img_data.get('image_index', i)}"
                metadata_json = json.dumps({
                    "image_id": image_id,
                    "source_file": img_data.get("source_file", ""),
                    "image_index": img_data.get("image_index", i),
                    "image_path": img_data.get("image_path"),
                    "type": "image"
                })
                embedding_blob = vectors[i].astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json, embedding_blob)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, (
                    int(vector_id),
                    img_data.get("source_file", ""),
                    image_id,
                    namespace,
                    "image",
                    metadata_json,
                    embedding_blob
                ))
            conn.commit()
            conn.close()
            
            # Update cache
            indices = self._load_namespace_indices(namespace)
            indices['image_index'] = index
            
            # Update manifest
            old_manifest = indices.get('image_manifest', {})
            manifest_info = old_manifest.copy()
            manifest_info.update({
                "ntotal": int(index.ntotal),
                "updated_at": datetime.now().isoformat()
            })
            self._save_namespace_index(namespace, "image", index, manifest_info)
        
        # Get actual index type from manifest
        indices = self._load_namespace_indices(namespace)
        manifest = indices.get('image_manifest', {})
        actual_index_type = manifest.get('index_type', 'IVF_PQ')
        
        return {
            "was_trained": was_trained,
            "retrain_reason": retrain_reason if should_retrain else None,
            "existing_ntotal": existing_ntotal,
            "new_count": new_count,
            "final_ntotal": int(index.ntotal),
            "index_type": actual_index_type,
            "nlist": manifest.get('nlist') if actual_index_type == "IVF_PQ" else None,
            "m": manifest.get('m') if actual_index_type == "IVF_PQ" else None,
            "nbits": manifest.get('nbits') if actual_index_type == "IVF_PQ" else None,
            "nprobe": manifest.get('nprobe', 16) if actual_index_type == "IVF_PQ" else None
        }
    
    def finalize_ingestion(self, namespace: Optional[str] = None,
                          nlist: Optional[int] = None, m: Optional[int] = None,
                          nbits: Optional[int] = None, nprobe: Optional[int] = None) -> Dict:
        """
        Finalize ingestion by checking 20% rule once for all new vectors and training if needed.
        This should be called after all vectors have been stored with skip_training_check=True.
        
        Returns:
            Dict with training info for text and image indices
        """
        namespace = namespace or self.default_namespace
        
        results = {
            "text_training_info": None,
            "image_training_info": None
        }
        
        # Get counts of new vectors (stored but not in index)
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Count new text vectors
        cursor.execute("""
            SELECT COUNT(*) FROM vectors 
            WHERE namespace = ? AND vector_type = 'text' AND deleted = 0
        """, (namespace,))
        total_text_vectors = cursor.fetchone()[0]
        
        # Count new image vectors
        cursor.execute("""
            SELECT COUNT(*) FROM vectors 
            WHERE namespace = ? AND vector_type = 'image' AND deleted = 0
        """, (namespace,))
        total_image_vectors = cursor.fetchone()[0]
        
        conn.close()
        
        # Process text vectors
        indices = self._load_namespace_indices(namespace)
        text_index = indices.get('text_index')
        
        if text_index is not None:
            existing_text_ntotal = text_index.ntotal
            new_text_count = total_text_vectors - existing_text_ntotal
        else:
            existing_text_ntotal = 0
            new_text_count = total_text_vectors
        
        if new_text_count > 0:
            # Load all new text vectors from SQLite
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT vector_id, embedding_blob, metadata_json FROM vectors 
                WHERE namespace = ? AND vector_type = 'text' AND deleted = 0
                ORDER BY vector_id
            """, (namespace,))
            
            new_text_vectors = []
            new_text_ids = []
            new_text_chunks = []
            
            for row in cursor.fetchall():
                vector_id, embedding_blob, metadata_json = row
                if embedding_blob is not None:
                    vec = np.frombuffer(embedding_blob, dtype=np.float32)
                    new_text_vectors.append(vec)
                    new_text_ids.append(int(vector_id))
                    new_text_chunks.append(json.loads(metadata_json))
            
            conn.close()
            
            if new_text_vectors:
                new_text_vectors = np.array(new_text_vectors, dtype=np.float32)
                new_text_ids = np.array(new_text_ids, dtype=np.int64)
                
                # Normalize
                norms = np.linalg.norm(new_text_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_text_vectors = new_text_vectors / norms
                
                # Ensure index exists
                dimension = new_text_vectors.shape[1]
                index_info = self.ensure_index(namespace, "text", dimension, nlist, m, nbits, nprobe)
                text_index = index_info['index']
                
                # Check 20% rule
                should_retrain, retrain_reason = self._should_retrain(existing_text_ntotal, new_text_count, text_index, "text")
                
                if should_retrain:
                    # Rebuild index - recompute nlist based on total vectors (existing + new)
                    total_vectors = existing_text_ntotal + new_text_count
                    nlist_val = _compute_nlist(total_vectors)  # Always recompute, don't use old manifest value
                    m_val = index_info['m']
                    nbits_val = index_info['nbits']
                    nprobe_val = index_info['nprobe']
                    self._rebuild_index(namespace, "text", dimension, nlist_val, m_val, nbits_val, nprobe_val)
                    indices = self._load_namespace_indices(namespace)
                    text_index = indices['text_index']
                    was_trained = True
                else:
                    # Add incrementally - only add vectors that aren't already in index
                    # Since we used skip_training_check, all vectors in SQLite should be new
                    # But to be safe, we'll add all of them (they shouldn't be in index yet)
                    text_index.add_with_ids(new_text_vectors, new_text_ids)
                    indices = self._load_namespace_indices(namespace)
                    indices['text_index'] = text_index
                    
                    # Update manifest
                    old_manifest = indices.get('text_manifest', {})
                    manifest_info = old_manifest.copy()
                    manifest_info.update({
                        "ntotal": int(text_index.ntotal),
                        "updated_at": datetime.now().isoformat()
                    })
                    self._save_namespace_index(namespace, "text", text_index, manifest_info)
                    was_trained = False
                
                # Get final stats
                indices = self._load_namespace_indices(namespace)
                manifest = indices.get('text_manifest', {})
                actual_index_type = manifest.get('index_type', 'IVF_PQ')
                
                results["text_training_info"] = {
                    "was_trained": was_trained,
                    "retrain_reason": retrain_reason if should_retrain else None,
                    "existing_ntotal": existing_text_ntotal,
                    "new_count": new_text_count,
                    "final_ntotal": int(text_index.ntotal),
                    "index_type": actual_index_type,
                    "nlist": manifest.get('nlist') if actual_index_type == "IVF_PQ" else None,
                    "m": manifest.get('m') if actual_index_type == "IVF_PQ" else None,
                    "nbits": manifest.get('nbits') if actual_index_type == "IVF_PQ" else None,
                    "nprobe": manifest.get('nprobe', 16) if actual_index_type == "IVF_PQ" else None
                }
        
        # Process image vectors (similar logic)
        indices = self._load_namespace_indices(namespace)
        image_index = indices.get('image_index')
        
        if image_index is not None:
            existing_image_ntotal = image_index.ntotal
            new_image_count = total_image_vectors - existing_image_ntotal
        else:
            existing_image_ntotal = 0
            new_image_count = total_image_vectors
        
        if new_image_count > 0:
            # Load all new image vectors from SQLite
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT vector_id, embedding_blob, metadata_json FROM vectors 
                WHERE namespace = ? AND vector_type = 'image' AND deleted = 0
                ORDER BY vector_id
            """, (namespace,))
            
            new_image_vectors = []
            new_image_ids = []
            
            for row in cursor.fetchall():
                vector_id, embedding_blob, metadata_json = row
                if embedding_blob is not None:
                    vec = np.frombuffer(embedding_blob, dtype=np.float32)
                    new_image_vectors.append(vec)
                    new_image_ids.append(int(vector_id))
            
            conn.close()
            
            if new_image_vectors:
                new_image_vectors = np.array(new_image_vectors, dtype=np.float32)
                new_image_ids = np.array(new_image_ids, dtype=np.int64)
                
                # Normalize
                norms = np.linalg.norm(new_image_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_image_vectors = new_image_vectors / norms
                
                # Ensure index exists
                dimension = new_image_vectors.shape[1]
                index_info = self.ensure_index(namespace, "image", dimension, nlist, m, nbits, nprobe)
                image_index = index_info['index']
                
                # Check 20% rule
                should_retrain, retrain_reason = self._should_retrain(existing_image_ntotal, new_image_count, image_index, "image")
                
                if should_retrain:
                    # Rebuild index - recompute nlist based on total vectors (existing + new)
                    total_vectors = existing_image_ntotal + new_image_count
                    nlist_val = _compute_nlist(total_vectors)  # Always recompute, don't use old manifest value
                    m_val = index_info['m']
                    nbits_val = index_info['nbits']
                    nprobe_val = index_info['nprobe']
                    self._rebuild_index(namespace, "image", dimension, nlist_val, m_val, nbits_val, nprobe_val)
                    indices = self._load_namespace_indices(namespace)
                    image_index = indices['image_index']
                    was_trained = True
                else:
                    # Add incrementally - only add vectors that aren't already in index
                    # Since we used skip_training_check, all vectors in SQLite should be new
                    # But to be safe, we'll add all of them (they shouldn't be in index yet)
                    image_index.add_with_ids(new_image_vectors, new_image_ids)
                    indices = self._load_namespace_indices(namespace)
                    indices['image_index'] = image_index
                    
                    # Update manifest
                    old_manifest = indices.get('image_manifest', {})
                    manifest_info = old_manifest.copy()
                    manifest_info.update({
                        "ntotal": int(image_index.ntotal),
                        "updated_at": datetime.now().isoformat()
                    })
                    self._save_namespace_index(namespace, "image", image_index, manifest_info)
                    was_trained = False
                
                # Get final stats
                indices = self._load_namespace_indices(namespace)
                manifest = indices.get('image_manifest', {})
                actual_index_type = manifest.get('index_type', 'IVF_PQ')
                
                results["image_training_info"] = {
                    "was_trained": was_trained,
                    "retrain_reason": retrain_reason if should_retrain else None,
                    "existing_ntotal": existing_image_ntotal,
                    "new_count": new_image_count,
                    "final_ntotal": int(image_index.ntotal),
                    "index_type": actual_index_type,
                    "nlist": manifest.get('nlist') if actual_index_type == "IVF_PQ" else None,
                    "m": manifest.get('m') if actual_index_type == "IVF_PQ" else None,
                    "nbits": manifest.get('nbits') if actual_index_type == "IVF_PQ" else None,
                    "nprobe": manifest.get('nprobe', 16) if actual_index_type == "IVF_PQ" else None
                }
        
        return results
    
    def save(self):
        """Save all namespace indices and manifests (indices are saved automatically during operations)."""
        pass
    
    def load(self) -> bool:
        """
        Load existing indices (lazy loading on demand).
        
        Returns:
            True if any indices were loaded
        """
        # Lazy loading - indices are loaded on demand
        return True
    
    
    def delete_by_doc_id(self, doc_id: str, namespace: Optional[str] = None, hard_delete: bool = False) -> dict:
        """Delete all vectors for a document (works with per-namespace indices)."""
        namespace = namespace or self.default_namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Find vector IDs to delete
        query = "SELECT vector_id, vector_type FROM vectors WHERE doc_id = ? AND deleted = 0"
        params = [doc_id]
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return {
                "deleted_count": 0,
                "text_vectors_deleted": 0,
                "image_vectors_deleted": 0,
                "index_updated": False,
                "database_updated": False,
                "message": f"Document '{doc_id}' not found"
            }
        
        # Separate by type
        text_ids = []
        image_ids = []
        
        for vector_id, vector_type in rows:
            if vector_type == "text":
                text_ids.append(int(vector_id))
            else:
                image_ids.append(int(vector_id))
        
        index_updated = False
        
        # Remove from FAISS indices
        indices = self._load_namespace_indices(namespace)
        if text_ids and indices.get('text_index') is not None:
            indices['text_index'].remove_ids(np.array(text_ids, dtype=np.int64))
            index_updated = True
            self._save_namespace_index(namespace, "text", indices['text_index'], indices.get('text_manifest', {}))
        
        if image_ids and indices.get('image_index') is not None:
            indices['image_index'].remove_ids(np.array(image_ids, dtype=np.int64))
            index_updated = True
            self._save_namespace_index(namespace, "image", indices['image_index'], indices.get('image_manifest', {}))
        
        # Update database
        if hard_delete:
            cursor.execute("""
                DELETE FROM vectors WHERE doc_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "hard"
        else:
            cursor.execute("""
                UPDATE vectors SET deleted = 1 WHERE doc_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "soft"
        
        conn.commit()
        conn.close()
        
        return {
            "deleted_count": deleted_count,
            "text_vectors_deleted": len(text_ids),
            "image_vectors_deleted": len(image_ids),
            "index_updated": index_updated,
            "database_updated": deleted_count > 0,
            "deletion_type": deletion_type,
            "message": f"{deletion_type.capitalize()} deleted {deleted_count} vector(s)"
        }
    
    def delete_by_chunk_id(self, chunk_id: str, namespace: Optional[str] = None, hard_delete: bool = False) -> dict:
        """Delete a specific chunk (works with per-namespace indices)."""
        namespace = namespace or self.default_namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        query = "SELECT vector_id, vector_type FROM vectors WHERE chunk_id = ? AND deleted = 0"
        params = [chunk_id]
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {
                "deleted_count": 0,
                "vector_type": None,
                "vector_id": None,
                "index_updated": False,
                "database_updated": False,
                "message": f"Chunk '{chunk_id}' not found"
            }
        
        vector_id, vector_type = row
        vector_id = int(vector_id)
        
        index_updated = False
        
        # Remove from FAISS index
        indices = self._load_namespace_indices(namespace)
        if vector_type == "text" and indices.get('text_index') is not None:
            indices['text_index'].remove_ids(np.array([vector_id], dtype=np.int64))
            index_updated = True
            self._save_namespace_index(namespace, "text", indices['text_index'], indices.get('text_manifest', {}))
        elif vector_type == "image" and indices.get('image_index') is not None:
            indices['image_index'].remove_ids(np.array([vector_id], dtype=np.int64))
            index_updated = True
            self._save_namespace_index(namespace, "image", indices['image_index'], indices.get('image_manifest', {}))
        
        # Update database
        if hard_delete:
            cursor.execute("""
                DELETE FROM vectors WHERE chunk_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "hard"
        else:
            cursor.execute("""
                UPDATE vectors SET deleted = 1 WHERE chunk_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "soft"
        
        conn.commit()
        conn.close()
        
        return {
            "deleted_count": deleted_count,
            "vector_type": vector_type,
            "vector_id": vector_id,
            "index_updated": index_updated,
            "database_updated": deleted_count > 0,
            "deletion_type": deletion_type,
            "message": f"{deletion_type.capitalize()} deleted chunk '{chunk_id}'"
        }
    
    def restore_by_doc_id(self, doc_id: str, namespace: Optional[str] = None) -> dict:
        """Restore soft-deleted vectors for a document."""
        namespace = namespace or self.default_namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        query = "SELECT vector_id, vector_type FROM vectors WHERE doc_id = ? AND deleted = 1"
        params = [doc_id]
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return {
                "restored_count": 0,
                "text_vectors_restored": 0,
                "image_vectors_restored": 0,
                "database_updated": False,
                "message": f"No soft-deleted vectors found for doc_id: {doc_id}"
            }
        
        # Restore in database
        cursor.execute("""
            UPDATE vectors SET deleted = 0 WHERE doc_id = ? AND deleted = 1
        """ + (" AND namespace = ?" if namespace else ""), params)
        
        restored_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        text_count = sum(1 for _, vt in rows if vt == "text")
        image_count = sum(1 for _, vt in rows if vt == "image")
        
        return {
            "restored_count": restored_count,
            "text_vectors_restored": text_count,
            "image_vectors_restored": image_count,
            "database_updated": restored_count > 0,
            "message": f"Restored {restored_count} vector(s) in database. Re-ingest to add back to index."
        }
    
    def restore_by_chunk_id(self, chunk_id: str, namespace: Optional[str] = None) -> dict:
        """Restore soft-deleted chunk."""
        namespace = namespace or self.default_namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        query = "SELECT vector_id, vector_type FROM vectors WHERE chunk_id = ? AND deleted = 1"
        params = [chunk_id]
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {
                "restored_count": 0,
                "vector_type": None,
                "vector_id": None,
                "database_updated": False,
                "message": f"No soft-deleted vector found for chunk_id: {chunk_id}"
            }
        
        vector_id, vector_type = row
        
        cursor.execute("""
            UPDATE vectors SET deleted = 0 WHERE chunk_id = ? AND deleted = 1
        """ + (" AND namespace = ?" if namespace else ""), params)
        
        restored_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "restored_count": restored_count,
            "vector_type": vector_type,
            "vector_id": int(vector_id),
            "database_updated": restored_count > 0,
            "message": f"Restored chunk '{chunk_id}' in database. Re-ingest to add back to index."
        }
    
    def search_text(self, query_vector: np.ndarray, k: int, namespace: Optional[str] = None, exclude_deleted: bool = True) -> List[Dict]:
        """Search text index for a namespace."""
        namespace = namespace or self.default_namespace
        
        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        indices = self._load_namespace_indices(namespace)
        index = indices.get('text_index')
        
        if index is None or index.ntotal == 0:
            return []
        
        # Set nprobe from manifest if available
        manifest = indices.get('text_manifest', {})
        nprobe = manifest.get('nprobe', 16)
        base_index = faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap2) else index
        if isinstance(base_index, faiss.IndexIVFPQ):
            base_index.nprobe = nprobe
        
        # Search
        scores, indices_array = index.search(query_vector, min(k * 2, index.ntotal))
        
        # Filter by namespace and deleted status
        results = []
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        for idx, score in zip(indices_array[0], scores[0]):
            if idx < 0:
                continue
            
            vector_id = int(idx)
            
            query = "SELECT namespace, deleted, metadata_json FROM vectors WHERE vector_id = ?"
            cursor.execute(query, (vector_id,))
            row = cursor.fetchone()
            
            if not row:
                continue
            
            vec_namespace, deleted, metadata_json = row
            
            if exclude_deleted and deleted:
                continue
            if namespace and vec_namespace != namespace:
                continue
            
            results.append({
                "vector_id": vector_id,
                "score": float(score),
                "namespace": vec_namespace,
                "metadata": json.loads(metadata_json) if metadata_json else {}
            })
            
            if len(results) >= k:
                break
        
        conn.close()
        return results
    
    def search_images(self, query_vector: np.ndarray, k: int, namespace: Optional[str] = None, exclude_deleted: bool = True) -> List[Dict]:
        """Search image index for a namespace."""
        namespace = namespace or self.default_namespace
        
        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        indices = self._load_namespace_indices(namespace)
        index = indices.get('image_index')
        
        if index is None or index.ntotal == 0:
            return []
        
        # Set nprobe from manifest
        manifest = indices.get('image_manifest', {})
        nprobe = manifest.get('nprobe', 16)
        base_index = faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap2) else index
        if isinstance(base_index, faiss.IndexIVFPQ):
            base_index.nprobe = nprobe
        
        scores, indices_array = index.search(query_vector, min(k * 2, index.ntotal))
        
        results = []
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        for idx, score in zip(indices_array[0], scores[0]):
            if idx < 0:
                continue
            
            vector_id = int(idx)
            
            query = "SELECT namespace, deleted, metadata_json FROM vectors WHERE vector_id = ?"
            cursor.execute(query, (vector_id,))
            row = cursor.fetchone()
            
            if not row:
                continue
            
            vec_namespace, deleted, metadata_json = row
            
            if exclude_deleted and deleted:
                continue
            if namespace and vec_namespace != namespace:
                continue
            
            results.append({
                "vector_id": vector_id,
                "score": float(score),
                "namespace": vec_namespace,
                "metadata": json.loads(metadata_json) if metadata_json else {}
            })
            
            if len(results) >= k:
                break
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the store, including per-namespace info."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Total vectors
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE deleted = 0")
        total_active = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE deleted = 1")
        total_deleted = cursor.fetchone()[0]
        
        # By namespace
        cursor.execute("SELECT namespace, COUNT(*) FROM vectors WHERE deleted = 0 GROUP BY namespace")
        namespace_counts = dict(cursor.fetchall())
        
        # Per-namespace detailed stats
        namespaces_info = {}
        for ns in namespace_counts.keys():
            indices = self._load_namespace_indices(ns)
            text_manifest = indices.get('text_manifest', {})
            image_manifest = indices.get('image_manifest', {})
            
            text_index = indices.get('text_index')
            image_index = indices.get('image_index')
            
            namespaces_info[ns] = {
                "index_type": text_manifest.get("index_type", "IVF_PQ") if text_index else None,
                "trained": text_manifest.get("trained", False) or image_manifest.get("trained", False),
                "text_vectors": int(text_index.ntotal) if text_index else 0,
                "image_vectors": int(image_index.ntotal) if image_index else 0,
                "nlist": text_manifest.get("nlist") or image_manifest.get("nlist"),
                "nprobe": text_manifest.get("nprobe") or image_manifest.get("nprobe", 16)
            }
        
        conn.close()
        
        return {
            "total_active_vectors": total_active,
            "total_deleted_vectors": total_deleted,
            "namespace_counts": namespace_counts,
            "namespaces": namespaces_info
        }
    
    def get_all_vectors(self, namespace: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        """
        Get all vectors from the database.
        
        Args:
            namespace: Optional namespace filter
            limit: Optional limit on number of vectors to return
            offset: Offset for pagination
        
        Returns:
            List of vector dictionaries with metadata
        """
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        query = "SELECT vector_id, doc_id, chunk_id, namespace, vector_type, metadata_json, created_at FROM vectors WHERE deleted = 0"
        params = []
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        query += " ORDER BY vector_id"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        vectors = []
        for row in rows:
            vector_id, doc_id, chunk_id, ns, vector_type, metadata_json, created_at = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            vectors.append({
                "vector_id": vector_id,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "namespace": ns,
                "vector_type": vector_type,
                "metadata": metadata,
                "created_at": created_at
            })
        
        return vectors
    
    def clear(self):
        """Clear all data."""
        # Clear all namespace indices
        self._namespace_indices = {}
        
        # Delete all namespace directories
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name not in ['metadata', 'tmp', 'snapshots', 'indices']:
                shutil.rmtree(item)
        
        # Clear database
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vectors")
        conn.commit()
        conn.close()
