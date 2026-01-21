"""
Configuration loader for sharded FAISS vector database.

Reads environment variables to configure shard mode:
- SHARD_ID: Shard identifier (0, 1, 2, ...)
- SHARD_COUNT: Total number of shards
- PORT: Port to run this shard on
- DATA_ROOT: Base directory for all data (default: ../data)
"""

import os
from pathlib import Path


class ShardConfig:
    """Configuration for shard mode."""
    
    def __init__(self):
        # Shard identification
        self.shard_id = int(os.environ.get("SHARD_ID", "0"))
        self.shard_count = int(os.environ.get("SHARD_COUNT", "1"))
        
        # Port configuration
        self.port = int(os.environ.get("PORT", "5001"))
        
        # Data directory configuration
        # Base directory is parent of faiss-research (main project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_root = os.environ.get("DATA_ROOT", os.path.join(base_dir, "data"))
        self.data_root = Path(data_root)
        
        # Shard-specific data directory
        # For shard mode: data/shards/shard{SHARD_ID}/
        # For non-shard mode: data/
        if self.shard_count > 1:
            self.shard_data_dir = self.data_root / "shards" / f"shard{self.shard_id}"
        else:
            self.shard_data_dir = self.data_root
        
        # Ensure shard data directory exists
        self.shard_data_dir.mkdir(parents=True, exist_ok=True)
    
    def is_shard_mode(self) -> bool:
        """Check if running in shard mode."""
        return self.shard_count > 1
    
    def get_data_dir(self) -> str:
        """Get the data directory for this shard."""
        return str(self.shard_data_dir)
    
    def __repr__(self) -> str:
        return (
            f"ShardConfig(shard_id={self.shard_id}, shard_count={self.shard_count}, "
            f"port={self.port}, data_dir={self.shard_data_dir})"
        )


# Global config instance
_config = None


def get_config() -> ShardConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ShardConfig()
    return _config
