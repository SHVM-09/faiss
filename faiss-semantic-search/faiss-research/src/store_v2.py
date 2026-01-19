"""
PRODUCTION VECTOR STORE
=======================
Production-ready vector store with:
- IndexIDMap2 for stable vector IDs
- SQLite metadata store
- Atomic writes
- Deletion support
- Snapshot system
- Namespace isolation
"""

import os
import json
import sqlite3
import numpy as np
import shutil
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    import faiss
except ImportError:
    raise ImportError("faiss not installed. Run: conda install -c conda-forge faiss-cpu")


class VectorStoreV2:
    """
    Production-ready vector store with stable IDs, SQLite metadata, and deletion support.
    """
    
    def __init__(self, data_dir: str = "./data", namespace: str = "default"):
        """
        Initialize the vector store.
        
        Args:
            data_dir: Base directory for all data files
            namespace: Default namespace for vectors
        """
        self.data_dir = Path(data_dir)
        self.namespace = namespace
        
        # Directory structure
        self.indices_dir = self.data_dir / "indices"
        self.metadata_dir = self.data_dir / "metadata"
        self.snapshots_dir = self.data_dir / "snapshots"
        self.tmp_dir = self.data_dir / "tmp"
        self.manifest_path = self.data_dir / "manifest.json"
        
        # Create directories
        for dir_path in [self.indices_dir, self.metadata_dir, self.snapshots_dir, self.tmp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.text_index_path = self.indices_dir / "text_index.faiss"
        self.image_index_path = self.indices_dir / "image_index.faiss"
        self.metadata_db_path = self.metadata_dir / "metadata.db"
        
        # FAISS indices
        self.text_index = None
        self.image_index = None
        self.text_dimension = 1024
        self.image_dimension = 1024
        
        # Next vector ID (auto-increment)
        self.next_vector_id = 1
        
        # Load existing data
        self._init_database()
        self.load()
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        try:
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
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
    
    def initialize_text_index(self, dimension: int = 1024):
        """Create a new IndexIDMap2 for text vectors."""
        self.text_dimension = dimension
        base_index = faiss.IndexFlatIP(dimension)
        self.text_index = faiss.IndexIDMap2(base_index)
        self.text_metadata = []
    
    def initialize_image_index(self, dimension: int = 1024):
        """Create a new IndexIDMap2 for image vectors."""
        self.image_dimension = dimension
        base_index = faiss.IndexFlatIP(dimension)
        self.image_index = faiss.IndexIDMap2(base_index)
        self.image_metadata = []
    
    def load(self) -> bool:
        """
        Load existing indices and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        loaded = False
        
        # Load manifest
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
                    self.next_vector_id = manifest.get("next_vector_id", 1)
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}")
        
        # Load text index
        if self.text_index_path.exists():
            try:
                self.text_index = faiss.read_index(str(self.text_index_path))
                if isinstance(self.text_index, faiss.IndexIDMap2):
                    self.text_dimension = self.text_index.d
                    loaded = True
                else:
                    # Migrate old IndexFlatIP to IndexIDMap2
                    print("Migrating text index to IndexIDMap2...")
                    base_index = self.text_index
                    self.text_index = faiss.IndexIDMap2(base_index)
                    # Assign IDs 0, 1, 2, ... to existing vectors
                    if base_index.ntotal > 0:
                        ids = np.arange(base_index.ntotal, dtype=np.int64)
                        self.text_index.add_with_ids(base_index.reconstruct_n(0, base_index.ntotal), ids)
                    loaded = True
            except Exception as e:
                print(f"Warning: Could not load text index: {e}")
                # Reset to None if load failed
                self.text_index = None
        
        # Load image index
        if self.image_index_path.exists():
            try:
                self.image_index = faiss.read_index(str(self.image_index_path))
                if isinstance(self.image_index, faiss.IndexIDMap2):
                    self.image_dimension = self.image_index.d
                    loaded = True
                else:
                    # Migrate old IndexFlatIP to IndexIDMap2
                    print("Migrating image index to IndexIDMap2...")
                    base_index = self.image_index
                    self.image_index = faiss.IndexIDMap2(base_index)
                    if base_index.ntotal > 0:
                        ids = np.arange(base_index.ntotal, dtype=np.int64)
                        self.image_index.add_with_ids(base_index.reconstruct_n(0, base_index.ntotal), ids)
                    loaded = True
            except Exception as e:
                print(f"Warning: Could not load image index: {e}")
                # Reset to None if load failed
                self.image_index = None
        
        # Update next_vector_id from database
        self.next_vector_id = max(self.next_vector_id, self._get_next_vector_id())
        
        return loaded
    
    def store_text(self, vectors: np.ndarray, chunks: List[Dict], namespace: Optional[str] = None):
        """
        Add text vectors with stable IDs.
        
        Args:
            vectors: numpy array of shape (num_vectors, dimension)
            chunks: List of chunk dictionaries
            namespace: Optional namespace (defaults to self.namespace)
        """
        if self.text_index is None:
            self.initialize_text_index(vectors.shape[1])
        
        namespace = namespace or self.namespace
        
        # Generate stable IDs
        num_vectors = len(vectors)
        vector_ids = np.arange(self.next_vector_id, self.next_vector_id + num_vectors, dtype=np.int64)
        
        # Add to FAISS with IDs
        self.text_index.add_with_ids(vectors, vector_ids)
        
        # Store metadata in SQLite
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        for i, (vector_id, chunk) in enumerate(zip(vector_ids, chunks)):
            metadata_json = json.dumps({
                "chunk_text": chunk.get("chunk_text", ""),
                "source_file": chunk.get("source_file", ""),
                "type": "text"
            })
            
            cursor.execute("""
                INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json)
                VALUES (?, ?, ?, ?, 0, ?, ?)
            """, (
                int(vector_id),
                chunk.get("source_file", ""),
                chunk.get("chunk_id", f"chunk_{vector_id}"),
                namespace,
                "text",
                metadata_json
            ))
        
        conn.commit()
        conn.close()
        
        # Update next_vector_id
        self.next_vector_id += num_vectors
    
    def store_images(self, vectors: np.ndarray, images: List[Dict], namespace: Optional[str] = None):
        """
        Add image vectors with stable IDs.
        
        Args:
            vectors: numpy array of shape (num_images, dimension)
            images: List of image dictionaries
            namespace: Optional namespace (defaults to self.namespace)
        """
        if self.image_index is None:
            self.initialize_image_index(vectors.shape[1])
        
        namespace = namespace or self.namespace
        
        # Generate stable IDs
        num_vectors = len(vectors)
        vector_ids = np.arange(self.next_vector_id, self.next_vector_id + num_vectors, dtype=np.int64)
        
        # Add to FAISS with IDs
        self.image_index.add_with_ids(vectors, vector_ids)
        
        # Store metadata in SQLite
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
            
            cursor.execute("""
                INSERT INTO vectors (vector_id, doc_id, chunk_id, namespace, deleted, vector_type, metadata_json)
                VALUES (?, ?, ?, ?, 0, ?, ?)
            """, (
                int(vector_id),
                img_data.get("source_file", ""),
                image_id,
                namespace,
                "image",
                metadata_json
            ))
        
        conn.commit()
        conn.close()
        
        # Update next_vector_id
        self.next_vector_id += num_vectors
    
    def delete_by_doc_id(self, doc_id: str, namespace: Optional[str] = None, hard_delete: bool = False) -> dict:
        """
        Delete all vectors for a document.
        
        Args:
            doc_id: Document ID to delete
            namespace: Optional namespace filter
            hard_delete: If True, permanently delete from index and DB. If False, soft delete (mark as deleted).
        
        Returns:
            Dictionary with deletion details:
            {
                "deleted_count": int,
                "text_vectors_deleted": int,
                "image_vectors_deleted": int,
                "index_updated": bool,
                "database_updated": bool,
                "deletion_type": "soft" or "hard"
            }
        """
        namespace = namespace or self.namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # First, check if doc_id exists at all (even if already deleted)
        check_query = "SELECT COUNT(*) FROM vectors WHERE doc_id = ?"
        check_params = [doc_id]
        if namespace:
            check_query += " AND namespace = ?"
            check_params.append(namespace)
        
        cursor.execute(check_query, check_params)
        total_exists = cursor.fetchone()[0]
        
        # Find vector IDs to delete (only non-deleted ones)
        query = "SELECT vector_id, vector_type FROM vectors WHERE doc_id = ? AND deleted = 0"
        params = [doc_id]
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            # Check if doc_id exists but is already deleted
            if total_exists > 0:
                return {
                    "deleted_count": 0,
                    "text_vectors_deleted": 0,
                    "image_vectors_deleted": 0,
                    "index_updated": False,
                    "database_updated": False,
                    "message": f"Document '{doc_id}' exists but all vectors are already marked as deleted",
                    "total_vectors_for_doc": total_exists
                }
            else:
                return {
                    "deleted_count": 0,
                    "text_vectors_deleted": 0,
                    "image_vectors_deleted": 0,
                    "index_updated": False,
                    "database_updated": False,
                    "message": f"Document '{doc_id}' not found in database",
                    "total_vectors_for_doc": 0
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
        if text_ids and self.text_index is not None:
            self.text_index.remove_ids(np.array(text_ids, dtype=np.int64))
            index_updated = True
        
        if image_ids and self.image_index is not None:
            self.image_index.remove_ids(np.array(image_ids, dtype=np.int64))
            index_updated = True
        
        # Update database based on deletion type
        if hard_delete:
            # Permanently delete from database
            cursor.execute("""
                DELETE FROM vectors WHERE doc_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "hard"
            message = f"Permanently deleted {deleted_count} vector(s) from index and database (cannot be undone)"
        else:
            # Soft delete: mark as deleted
            cursor.execute("""
                UPDATE vectors SET deleted = 1 WHERE doc_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "soft"
            message = f"Soft deleted {deleted_count} vector(s) (can be restored)"
        
        database_updated = deleted_count > 0
        
        conn.commit()
        conn.close()
        
        return {
            "deleted_count": deleted_count,
            "text_vectors_deleted": len(text_ids),
            "image_vectors_deleted": len(image_ids),
            "index_updated": index_updated,
            "database_updated": database_updated,
            "deletion_type": deletion_type,
            "message": message
        }
    
    def delete_by_chunk_id(self, chunk_id: str, namespace: Optional[str] = None, hard_delete: bool = False) -> dict:
        """
        Delete a specific chunk.
        
        Args:
            chunk_id: Chunk ID to delete
            namespace: Optional namespace filter
            hard_delete: If True, permanently delete from index and DB. If False, soft delete (mark as deleted).
        
        Returns:
            Dictionary with deletion details:
            {
                "deleted_count": int (0 or 1),
                "vector_type": str ("text" or "image"),
                "vector_id": int,
                "index_updated": bool,
                "database_updated": bool,
                "deletion_type": "soft" or "hard"
            }
        """
        namespace = namespace or self.namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # First, check if chunk_id exists at all (even if already deleted)
        check_query = "SELECT COUNT(*) FROM vectors WHERE chunk_id = ?"
        check_params = [chunk_id]
        if namespace:
            check_query += " AND namespace = ?"
            check_params.append(namespace)
        
        cursor.execute(check_query, check_params)
        total_exists = cursor.fetchone()[0]
        
        # Find vector to delete (only non-deleted ones)
        query = "SELECT vector_id, vector_type FROM vectors WHERE chunk_id = ? AND deleted = 0"
        params = [chunk_id]
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            # Check if chunk_id exists but is already deleted
            if total_exists > 0:
                return {
                    "deleted_count": 0,
                    "vector_type": None,
                    "vector_id": None,
                    "index_updated": False,
                    "database_updated": False,
                    "message": f"Chunk '{chunk_id}' exists but is already marked as deleted"
                }
            else:
                return {
                    "deleted_count": 0,
                    "vector_type": None,
                    "vector_id": None,
                    "index_updated": False,
                    "database_updated": False,
                    "message": f"Chunk '{chunk_id}' not found in database"
                }
        
        vector_id, vector_type = row
        vector_id = int(vector_id)
        
        index_updated = False
        
        # Remove from FAISS index
        if vector_type == "text" and self.text_index is not None:
            self.text_index.remove_ids(np.array([vector_id], dtype=np.int64))
            index_updated = True
        elif vector_type == "image" and self.image_index is not None:
            self.image_index.remove_ids(np.array([vector_id], dtype=np.int64))
            index_updated = True
        
        # Update database based on deletion type
        if hard_delete:
            # Permanently delete from database
            cursor.execute("""
                DELETE FROM vectors WHERE chunk_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "hard"
            message = f"Permanently deleted chunk '{chunk_id}' from index and database (cannot be undone)"
        else:
            # Soft delete: mark as deleted
            cursor.execute("""
                UPDATE vectors SET deleted = 1 WHERE chunk_id = ? AND deleted = 0
            """ + (" AND namespace = ?" if namespace else ""), params)
            deleted_count = cursor.rowcount
            deletion_type = "soft"
            message = f"Soft deleted chunk '{chunk_id}' (can be restored)"
        
        database_updated = deleted_count > 0
        
        conn.commit()
        conn.close()
        
        return {
            "deleted_count": deleted_count,
            "vector_type": vector_type,
            "vector_id": vector_id,
            "index_updated": index_updated,
            "database_updated": database_updated,
            "deletion_type": deletion_type,
            "message": message
        }
    
    def restore_by_doc_id(self, doc_id: str, namespace: Optional[str] = None) -> dict:
        """
        Restore (undo soft delete) all vectors for a document.
        This only works for soft-deleted vectors. Hard-deleted vectors cannot be restored.
        
        Args:
            doc_id: Document ID to restore
            namespace: Optional namespace filter
        
        Returns:
            Dictionary with restoration details:
            {
                "restored_count": int,
                "text_vectors_restored": int,
                "image_vectors_restored": int,
                "database_updated": bool,
                "message": str
            }
        """
        namespace = namespace or self.namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Find soft-deleted vectors to restore
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
        
        # Separate by type
        text_ids = []
        image_ids = []
        
        for vector_id, vector_type in rows:
            if vector_type == "text":
                text_ids.append(int(vector_id))
            else:
                image_ids.append(int(vector_id))
        
        # Restore in database (set deleted = 0)
        cursor.execute("""
            UPDATE vectors SET deleted = 0 WHERE doc_id = ? AND deleted = 1
        """ + (" AND namespace = ?" if namespace else ""), params)
        
        restored_count = cursor.rowcount
        database_updated = restored_count > 0
        
        conn.commit()
        conn.close()
        
        # Note: We cannot restore vectors to FAISS index after they've been removed
        # The vectors are still in the database but not in the index
        # User would need to re-ingest the document to add vectors back to index
        
        return {
            "restored_count": restored_count,
            "text_vectors_restored": len(text_ids),
            "image_vectors_restored": len(image_ids),
            "database_updated": database_updated,
            "message": f"Restored {restored_count} vector(s) in database. Note: Vectors are not in FAISS index - re-ingest document to add them back to index."
        }
    
    def restore_by_chunk_id(self, chunk_id: str, namespace: Optional[str] = None) -> dict:
        """
        Restore (undo soft delete) a specific chunk.
        This only works for soft-deleted vectors. Hard-deleted vectors cannot be restored.
        
        Args:
            chunk_id: Chunk ID to restore
            namespace: Optional namespace filter
        
        Returns:
            Dictionary with restoration details:
            {
                "restored_count": int (0 or 1),
                "vector_type": str ("text" or "image"),
                "vector_id": int,
                "database_updated": bool,
                "message": str
            }
        """
        namespace = namespace or self.namespace
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Find soft-deleted vector to restore
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
        vector_id = int(vector_id)
        
        # Restore in database (set deleted = 0)
        cursor.execute("""
            UPDATE vectors SET deleted = 0 WHERE chunk_id = ? AND deleted = 1
        """ + (" AND namespace = ?" if namespace else ""), params)
        
        restored_count = cursor.rowcount
        database_updated = restored_count > 0
        
        conn.commit()
        conn.close()
        
        # Note: We cannot restore vector to FAISS index after it's been removed
        # The vector is still in the database but not in the index
        # User would need to re-ingest the chunk to add it back to index
        
        return {
            "restored_count": restored_count,
            "vector_type": vector_type,
            "vector_id": vector_id,
            "database_updated": database_updated,
            "message": f"Restored chunk '{chunk_id}' in database. Note: Vector is not in FAISS index - re-ingest chunk to add it back to index."
        }
    
    def compact(self, namespace: Optional[str] = None):
        """
        Rebuild indices excluding deleted vectors.
        
        Args:
            namespace: Optional namespace filter
        """
        namespace = namespace or self.namespace
        
        # Get all active vectors from database
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        query = "SELECT vector_id, vector_type, metadata_json FROM vectors WHERE deleted = 0"
        params = []
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Separate by type
        text_vectors = []
        text_ids = []
        text_metadata = []
        image_vectors = []
        image_ids = []
        image_metadata = []
        
        for vector_id, vector_type, metadata_json in rows:
            vector_id = int(vector_id)
            
            # Reconstruct vector from old index
            if vector_type == "text" and self.text_index is not None:
                try:
                    vec = self.text_index.reconstruct(vector_id)
                    text_vectors.append(vec)
                    text_ids.append(vector_id)
                    text_metadata.append(json.loads(metadata_json))
                except:
                    pass  # Vector might have been removed
            elif vector_type == "image" and self.image_index is not None:
                try:
                    vec = self.image_index.reconstruct(vector_id)
                    image_vectors.append(vec)
                    image_ids.append(vector_id)
                    image_metadata.append(json.loads(metadata_json))
                except:
                    pass
        
        # Rebuild text index
        if text_vectors:
            self.initialize_text_index(self.text_dimension)
            if text_vectors:
                self.text_index.add_with_ids(
                    np.array(text_vectors, dtype=np.float32),
                    np.array(text_ids, dtype=np.int64)
                )
        
        # Rebuild image index
        if image_vectors:
            self.initialize_image_index(self.image_dimension)
            if image_vectors:
                self.image_index.add_with_ids(
                    np.array(image_vectors, dtype=np.float32),
                    np.array(image_ids, dtype=np.int64)
                )
        
        # Remove deleted rows from database
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        query = "DELETE FROM vectors WHERE deleted = 1"
        if namespace:
            query += " AND namespace = ?"
            cursor.execute(query, [namespace])
        else:
            cursor.execute(query)
        conn.commit()
        conn.close()
    
    def save(self):
        """Save indices and manifest with atomic writes."""
        # Save text index atomically
        if self.text_index is not None:
            tmp_path = self.tmp_dir / "text_index.faiss.tmp"
            faiss.write_index(self.text_index, str(tmp_path))
            # Atomic rename
            os.rename(str(tmp_path), str(self.text_index_path))
        
        # Save image index atomically
        if self.image_index is not None:
            tmp_path = self.tmp_dir / "image_index.faiss.tmp"
            faiss.write_index(self.image_index, str(tmp_path))
            os.rename(str(tmp_path), str(self.image_index_path))
        
        # Save manifest
        manifest = {
            "version": "1.0.0",
            "snapshot_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(),
            "text_vector_count": int(self.text_index.ntotal) if self.text_index else 0,
            "image_vector_count": int(self.image_index.ntotal) if self.image_index else 0,
            "next_vector_id": self.next_vector_id,
            "indices": {
                "text": {
                    "path": str(self.text_index_path.relative_to(self.data_dir)),
                    "dimension": self.text_dimension,
                    "type": "IndexIDMap2"
                },
                "image": {
                    "path": str(self.image_index_path.relative_to(self.data_dir)),
                    "dimension": self.image_dimension,
                    "type": "IndexIDMap2"
                }
            }
        }
        
        tmp_manifest = self.tmp_dir / "manifest.json.tmp"
        with open(tmp_manifest, 'w') as f:
            json.dump(manifest, f, indent=2)
        os.rename(str(tmp_manifest), str(self.manifest_path))
    
    def create_snapshot(self, snapshot_name: Optional[str] = None) -> str:
        """
        Create a snapshot of current state.
        
        Args:
            snapshot_name: Optional snapshot name (defaults to timestamp)
        
        Returns:
            Snapshot directory path
        """
        if snapshot_name is None:
            snapshot_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        snapshot_dir = self.snapshots_dir / f"snapshot_{snapshot_name}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy indices
        if self.text_index_path.exists():
            shutil.copy2(self.text_index_path, snapshot_dir / "text_index.faiss")
        if self.image_index_path.exists():
            shutil.copy2(self.image_index_path, snapshot_dir / "image_index.faiss")
        
        # Copy database
        if self.metadata_db_path.exists():
            shutil.copy2(self.metadata_db_path, snapshot_dir / "metadata.db")
        
        # Copy manifest
        if self.manifest_path.exists():
            shutil.copy2(self.manifest_path, snapshot_dir / "manifest.json")
        
        return str(snapshot_dir)
    
    def load_snapshot(self, snapshot_name: str):
        """
        Load a snapshot.
        
        Args:
            snapshot_name: Snapshot name (without "snapshot_" prefix)
        """
        snapshot_dir = self.snapshots_dir / f"snapshot_{snapshot_name}"
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found: {snapshot_name}")
        
        # Restore indices
        if (snapshot_dir / "text_index.faiss").exists():
            shutil.copy2(snapshot_dir / "text_index.faiss", self.text_index_path)
        if (snapshot_dir / "image_index.faiss").exists():
            shutil.copy2(snapshot_dir / "image_index.faiss", self.image_index_path)
        
        # Restore database
        if (snapshot_dir / "metadata.db").exists():
            shutil.copy2(snapshot_dir / "metadata.db", self.metadata_db_path)
        
        # Restore manifest
        if (snapshot_dir / "manifest.json").exists():
            shutil.copy2(snapshot_dir / "manifest.json", self.manifest_path)
        
        # Reload
        self.load()
    
    def get_metadata(self, vector_id: int) -> Optional[Dict]:
        """Get metadata for a vector ID."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT doc_id, chunk_id, namespace, deleted, vector_type, metadata_json
            FROM vectors WHERE vector_id = ?
        """, (vector_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "vector_id": vector_id,
                "doc_id": row[0],
                "chunk_id": row[1],
                "namespace": row[2],
                "deleted": bool(row[3]),
                "vector_type": row[4],
                "metadata": json.loads(row[5]) if row[5] else {}
            }
        return None
    
    def search_text(self, query_vector: np.ndarray, k: int, namespace: Optional[str] = None, exclude_deleted: bool = True) -> List[Dict]:
        """
        Search text index.
        
        Args:
            query_vector: Query vector (1, dimension)
            k: Number of results
            namespace: Optional namespace filter
            exclude_deleted: Whether to exclude deleted vectors
        
        Returns:
            List of results with vector_id, score, metadata
        """
        if self.text_index is None or self.text_index.ntotal == 0:
            return []
        
        # Search
        scores, indices = self.text_index.search(query_vector, min(k * 2, self.text_index.ntotal))
        
        # Filter by namespace and deleted status
        results = []
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            
            vector_id = int(idx)
            
            # Get metadata
            query = "SELECT namespace, deleted, metadata_json FROM vectors WHERE vector_id = ?"
            cursor.execute(query, (vector_id,))
            row = cursor.fetchone()
            
            if not row:
                continue
            
            vec_namespace, deleted, metadata_json = row
            
            # Filter
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
        """Search image index (same as search_text)."""
        if self.image_index is None or self.image_index.ntotal == 0:
            return []
        
        scores, indices = self.image_index.search(query_vector, min(k * 2, self.image_index.ntotal))
        
        results = []
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        for idx, score in zip(indices[0], scores[0]):
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
        """Get statistics about the store."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Total vectors
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE deleted = 0")
        total_active = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE deleted = 1")
        total_deleted = cursor.fetchone()[0]
        
        # By type
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE vector_type = 'text' AND deleted = 0")
        text_active = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE vector_type = 'image' AND deleted = 0")
        image_active = cursor.fetchone()[0]
        
        # By namespace
        cursor.execute("SELECT namespace, COUNT(*) FROM vectors WHERE deleted = 0 GROUP BY namespace")
        namespace_counts = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "text_vector_count": int(self.text_index.ntotal) if self.text_index else 0,
            "image_vector_count": int(self.image_index.ntotal) if self.image_index else 0,
            "text_dimension": self.text_dimension,
            "image_dimension": self.image_dimension,
            "total_active_vectors": total_active,
            "total_deleted_vectors": total_deleted,
            "text_active": text_active,
            "image_active": image_active,
            "namespace_counts": namespace_counts,
            "next_vector_id": self.next_vector_id
        }
    
    def clear(self):
        """Clear all data."""
        self.text_index = None
        self.image_index = None
        self.next_vector_id = 1
        
        # Delete files
        for path in [self.text_index_path, self.image_index_path, self.manifest_path]:
            if path.exists():
                path.unlink()
        
        # Clear database
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vectors")
        conn.commit()
        conn.close()
