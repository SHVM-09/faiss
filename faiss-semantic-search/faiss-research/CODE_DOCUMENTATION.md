# Complete Code Documentation - Line by Line

This document provides a comprehensive, line-by-line explanation of the sharding implementation for the FAISS vector database application.

---

## Table of Contents

1. [Configuration Module (`src/config.py`)](#1-configuration-module-srcconfigpy)
2. [Router Service (`router.py`)](#2-router-service-routerpy)
3. [Shard Application (`app_v2.py`)](#3-shard-application-app_v2py)
4. [Vector Store (`src/store_v2.py`) - Key Changes](#4-vector-store-srcstore_v2py---key-changes)
5. [UI Dashboard (`static/index.html`)](#5-ui-dashboard-staticindexhtml)

---

## 1. Configuration Module (`src/config.py`)

### Purpose
Centralized configuration system that reads environment variables to configure shard mode. Ensures each shard instance uses isolated data directories.

### Line-by-Line Explanation

```python
"""
Configuration loader for sharded FAISS vector database.

Reads environment variables to configure shard mode:
- SHARD_ID: Shard identifier (0, 1, 2, ...)
- SHARD_COUNT: Total number of shards
- PORT: Port to run this shard on
- DATA_ROOT: Base directory for all data (default: ../data)
"""
```
**Lines 1-8**: Module docstring explaining purpose and environment variables.

```python
import os
from pathlib import Path
```
**Lines 11-12**: 
- `os`: Access environment variables
- `Path`: Platform-independent path handling

```python
class ShardConfig:
    """Configuration for shard mode."""
```
**Lines 15-16**: Class definition for shard configuration.

```python
    def __init__(self):
        # Shard identification
        self.shard_id = int(os.environ.get("SHARD_ID", "0"))
        self.shard_count = int(os.environ.get("SHARD_COUNT", "1"))
```
**Lines 18-21**:
- `__init__`: Constructor that reads environment variables
- `SHARD_ID`: Which shard this instance is (0, 1, 2, ...). Defaults to "0"
- `SHARD_COUNT`: Total number of shards in the cluster. Defaults to "1" (single instance)
- Converts strings to integers for numeric operations

```python
        # Port configuration
        self.port = int(os.environ.get("PORT", "5001"))
```
**Lines 23-24**:
- `PORT`: Port number this shard listens on. Defaults to "5001"
- Each shard must run on a different port

```python
        # Data directory configuration
        # Base directory is parent of faiss-research (main project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_root = os.environ.get("DATA_ROOT", os.path.join(base_dir, "data"))
        self.data_root = Path(data_root)
```
**Lines 26-30**:
- Calculates base directory: goes up 3 levels from `src/config.py` to project root
- `DATA_ROOT`: Optional override for data directory location
- Default: `{project_root}/data`
- Converts to `Path` object for easier manipulation

```python
        # Shard-specific data directory
        # For shard mode: data/shards/shard{SHARD_ID}/
        # For non-shard mode: data/
        if self.shard_count > 1:
            self.shard_data_dir = self.data_root / "shards" / f"shard{self.shard_id}"
        else:
            self.shard_data_dir = self.data_root
```
**Lines 32-38**:
- **Shard mode** (`shard_count > 1`): Creates isolated directory per shard
  - Example: `data/shards/shard0/`, `data/shards/shard1/`
- **Non-shard mode** (`shard_count == 1`): Uses base data directory
  - Example: `data/`
- Prevents file collisions between shards

```python
        # Ensure shard data directory exists
        self.shard_data_dir.mkdir(parents=True, exist_ok=True)
```
**Lines 40-41**:
- Creates the data directory if it doesn't exist
- `parents=True`: Creates parent directories if needed
- `exist_ok=True`: Doesn't error if directory already exists

```python
    def is_shard_mode(self) -> bool:
        """Check if running in shard mode."""
        return self.shard_count > 1
```
**Lines 43-45**:
- Helper method to check if running in shard mode
- Returns `True` if `shard_count > 1`, `False` otherwise

```python
    def get_data_dir(self) -> str:
        """Get the data directory for this shard."""
        return str(self.shard_data_dir)
```
**Lines 47-49**:
- Returns the shard-specific data directory as a string
- Used by `VectorStoreV2` to initialize storage location

```python
    def __repr__(self) -> str:
        return (
            f"ShardConfig(shard_id={self.shard_id}, shard_count={self.shard_count}, "
            f"port={self.port}, data_dir={self.shard_data_dir})"
        )
```
**Lines 51-55**:
- String representation for debugging
- Shows all configuration values when printed

```python
# Global config instance
_config = None
```
**Lines 58-59**:
- Global variable to store singleton config instance
- `None` initially, created on first access

```python
def get_config() -> ShardConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ShardConfig()
    return _config
```
**Lines 62-67**:
- **Singleton pattern**: Ensures only one config instance exists
- First call creates `ShardConfig()`, subsequent calls return same instance
- Prevents multiple config reads and ensures consistency

---

## 2. Router Service (`router.py`)

### Purpose
Central service that routes client requests to appropriate shards. Handles file distribution, parallel search, and result merging.

### Key Sections

#### 2.1 Imports and Setup (Lines 1-63)

```python
"""
Router service for sharded FAISS vector database.

Distributes documents across shards for parallel processing:
- Ingest: Splits files by filename hash across shards
- Delete/Restore: Routes by doc_id hash
- Search: Broadcasts to all shards and merges results
"""
```
**Lines 1-8**: Module docstring explaining router's role.

```python
import os
import json
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from pathlib import Path
```
**Lines 10-15**:
- `os`: Environment variables
- `json`: JSON parsing
- `typing`: Type hints for better code clarity
- `ThreadPoolExecutor`: Parallel HTTP requests to shards
- `as_completed`: Handle futures as they complete
- `Flask`: Web framework
- `Path`: Path manipulation

```python
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
    print("Warning: flask-cors not installed. CORS support disabled. Install with: pip install flask-cors")
```
**Lines 17-22**:
- **Optional CORS**: Tries to import `flask_cors`
- If available: Uses library for CORS
- If not: Falls back to manual CORS headers (see lines 40-46)
- Allows UI to make cross-origin requests

```python
try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    import hashlib
    HAS_XXHASH = False
```
**Lines 24-29**:
- **Optional xxhash**: Fast hashing library
- If available: Uses xxhash (faster)
- If not: Falls back to SHA1 (slower but always available)
- Both are deterministic (same input = same output)

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
```
**Lines 31-33**:
- `requests`: HTTP client for forwarding requests to shards
- `HTTPAdapter`: Custom session configuration
- `Retry`: Automatic retry on failures

```python
app = Flask(__name__)
# Enable CORS for all routes if available
if HAS_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})
else:
    # Manual CORS headers if flask-cors not available
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
```
**Lines 35-46**:
- Creates Flask app
- **CORS setup**:
  - If `flask_cors` available: Uses library (cleaner)
  - Otherwise: Adds CORS headers manually to all responses
  - Allows UI at different origin to access API

```python
# Configuration
SHARDS = os.environ.get("SHARDS", "http://127.0.0.1:5001,http://127.0.0.1:5002").split(",")
SHARDS = [s.strip() for s in SHARDS if s.strip()]
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "5003"))
```
**Lines 48-51**:
- `SHARDS`: Comma-separated list of shard URLs from environment
- Default: `["http://127.0.0.1:5001", "http://127.0.0.1:5002"]`
- Strips whitespace and filters empty strings
- `ROUTER_PORT`: Port router listens on (default: 5003)

```python
# Create a session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=2,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```
**Lines 53-62**:
- Creates HTTP session with retry logic
- **Retry strategy**:
  - `total=2`: Maximum 2 retries
  - `backoff_factor=0.1`: Wait 0.1s between retries
  - `status_forcelist`: Retry on server errors (500, 502, 503, 504)
- Mounts adapter to session for automatic retries

#### 2.2 Hashing Functions (Lines 65-100)

```python
def stable_hash(value: str) -> int:
    """
    Compute a stable hash of a string.
    
    Uses xxhash if available (faster), otherwise falls back to SHA1.
    Both are deterministic and stable across runs.
    """
    if HAS_XXHASH:
        return xxhash.xxh64(value.encode('utf-8')).intdigest()
    else:
        return int(hashlib.sha1(value.encode('utf-8')).hexdigest(), 16)
```
**Lines 65-75**:
- **Purpose**: Deterministic hashing (same input always produces same hash)
- **Why needed**: Consistent routing - same document always goes to same shard
- **Implementation**:
  - If xxhash available: Uses fast xxhash64
  - Otherwise: Uses SHA1 (slower but standard)
- **Returns**: Integer hash value

```python
def shard_for_namespace(namespace: str, shard_count: int) -> int:
    """
    Determine which shard should handle a namespace.
    
    Uses stable hashing to ensure consistent routing.
    """
    if shard_count <= 1:
        return 0
    hash_value = stable_hash(namespace)
    return hash_value % shard_count
```
**Lines 78-87**:
- **Purpose**: Route namespace to specific shard
- **Logic**: Hash namespace, modulo by shard count
- **Example**: `shard_for_namespace("alpha", 2)` → 0 or 1
- **Note**: Not used in final implementation (we use document-based routing)

```python
def shard_for_doc_id(doc_id: str, shard_count: int) -> int:
    """
    Determine which shard should handle a document.
    
    Uses stable hashing on doc_id to distribute documents across shards.
    This allows documents within the same namespace to be distributed.
    """
    if shard_count <= 1:
        return 0
    hash_value = stable_hash(doc_id)
    return hash_value % shard_count
```
**Lines 90-100**:
- **Purpose**: Route document to specific shard based on `doc_id`
- **Key feature**: Documents in same namespace can go to different shards
- **Logic**: Hash `doc_id`, modulo by shard count
- **Example**: `shard_for_doc_id("doc1", 2)` → 0 or 1
- **Used for**: Delete, restore operations

#### 2.3 Request Forwarding (Lines 128-166)

```python
def forward_request(shard_url: str, method: str, path: str, 
                   json_data: Optional[Dict] = None, 
                   timeout: float = 30.0) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Forward a request to a shard.
    
    Args:
        shard_url: Base URL of the shard
        method: HTTP method (GET, POST)
        path: Path including query string (e.g., "/vectors?namespace=test&limit=100")
        json_data: JSON data for POST requests
        timeout: Request timeout in seconds
    
    Returns:
        (response_json, error_message) - one will be None
    """
```
**Lines 128-144**: Function signature and docstring.

```python
    try:
        url = f"{shard_url.rstrip('/')}/{path.lstrip('/')}"
        if method == "GET":
            response = session.get(url, timeout=timeout)
        elif method == "POST":
            response = session.post(url, json=json_data, timeout=timeout)
        else:
            return None, f"Unsupported method: {method}"
```
**Lines 145-152**:
- Builds full URL by combining shard URL and path
- Handles GET and POST methods
- Uses session with retry strategy
- Returns error for unsupported methods

```python
        # Check for HTTP errors
        if response.status_code >= 400:
            # Try to get error details from response
            try:
                error_data = response.json()
                error_msg = error_data.get("error", f"{response.status_code} {response.reason}")
            except:
                error_msg = f"{response.status_code} Server Error: {response.reason} for url: {url}"
            return None, error_msg
```
**Lines 154-161**:
- Checks for HTTP errors (status >= 400)
- Tries to extract error message from JSON response
- Falls back to status code + reason if JSON parsing fails
- Returns `(None, error_message)` tuple

```python
        # Success
        try:
            return response.json(), None
        except:
            return None, f"Invalid JSON response from {url}"
```
**Lines 163-166**:
- Parses JSON response on success
- Returns `(response_json, None)` tuple
- Handles JSON parsing errors

#### 2.4 Ingest Endpoint (Lines 200-400+)

The `/ingest` endpoint is complex. Key logic:

1. **File Distribution** (Lines 220-280):
   - Lists all files in `docs_path`
   - Groups files by shard using `hash(filename + namespace) % shard_count`
   - Creates temporary subdirectories for each shard

2. **File Copying** (Lines 280-350):
   - Copies files (not symlinks) to shard-specific temp directories
   - Verifies file existence and size
   - More reliable than symlinks

3. **Parallel Requests** (Lines 350-400):
   - Sends ingest requests to all shards in parallel
   - Uses `ThreadPoolExecutor` for concurrency
   - Collects responses and errors

4. **Cleanup** (Lines 400-450):
   - Background thread with 60-second delay
   - Safely removes temporary directories
   - Allows shards to finish processing PDFs

#### 2.5 Search Endpoint (Lines 500-650)

```python
@app.route('/search', methods=['POST'])
def search():
    """
    Broadcast search to all shards and merge results.
    """
```
**Lines 500-503**: Endpoint definition.

```python
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        query = data.get("query")
        namespace = data.get("namespace", "default")
        k = data.get("k", 10)
        vector_type = data.get("vector_type", "both")
```
**Lines 504-512**: 
- Extracts request parameters
- Validates JSON body exists
- Gets query, namespace, result count (k), vector type

```python
        # Broadcast to ALL shards in parallel
        all_text_results = []
        all_image_results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
            futures = {
                executor.submit(forward_request, shard_url, "POST", "/search", json_data=data, timeout=60.0): shard_url
                for shard_url in SHARDS
            }
```
**Lines 514-523**:
- Creates lists to collect results from all shards
- Uses `ThreadPoolExecutor` for parallel requests
- Submits search request to each shard concurrently
- Stores future-to-shard mapping

```python
            for future in as_completed(futures):
                shard_url = futures[future]
                try:
                    response, error = future.result()
                    if error:
                        errors.append({"shard": shard_url, "error": error})
                    else:
                        if response:
                            all_text_results.extend(response.get("text_results", []))
                            all_image_results.extend(response.get("image_results", []))
                except Exception as e:
                    errors.append({"shard": shard_url, "error": str(e)})
```
**Lines 525-537**:
- Processes futures as they complete (not waiting for all)
- Collects text and image results separately
- Handles errors gracefully (continues with healthy shards)
- Extends result lists with shard responses

```python
        # Deduplicate results
        seen_text = set()
        seen_image = set()
        unique_text = []
        unique_image = []
        
        for result in all_text_results:
            key = (result.get("metadata", {}).get("source_file"), 
                   result.get("metadata", {}).get("chunk_id"))
            if key not in seen_text:
                seen_text.add(key)
                unique_text.append(result)
```
**Lines 539-550**:
- **Deduplication**: Removes duplicate results from multiple shards
- Uses `(source_file, chunk_id)` tuple as unique key
- Only adds results not seen before

```python
        # Sort and take top-k
        unique_text.sort(key=lambda x: x.get("score", 0), reverse=True)
        unique_image.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        final_text = unique_text[:k]
        final_image = unique_image[:k]
```
**Lines 552-556**:
- Sorts by score (descending) - highest scores first
- Takes top-k results from each type
- Ensures consistent result ordering

```python
        response = {
            "text_results": final_text,
            "image_results": final_image,
            "shards_queried": len(SHARDS) - len(errors),
            "total_shards": len(SHARDS),
            "shard_result_counts": {
                "text": len(all_text_results),
                "image": len(all_image_results)
            }
        }
        
        if errors:
            response["warnings"] = {
                "message": f"{len(errors)} shard(s) returned errors",
                "errors": errors
            }
        
        return jsonify(response), 200
```
**Lines 558-571**:
- Builds response with merged results
- Includes metadata about shards queried
- Adds warnings if any shards failed
- Returns 200 even if some shards failed (graceful degradation)

#### 2.6 Vectors Endpoint (Lines 950-1025)

```python
@app.route('/vectors', methods=['GET'])
def get_vectors():
    """
    Get all vectors from all shards.
    
    Query parameters:
    - namespace: Optional namespace filter
    - limit: Optional limit per shard (default: 10000)
    - offset: Optional offset for pagination (default: 0)
    """
```
**Lines 950-959**: Endpoint definition with query parameters.

```python
    try:
        namespace = request.args.get('namespace')
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', type=int, default=0)
```
**Lines 960-963**: Extracts query parameters from URL.

```python
        # Broadcast to all shards
        all_vectors = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
            futures = {}
            for shard_url in SHARDS:
                # Build query string for this request
                query_parts = []
                if namespace:
                    query_parts.append(f"namespace={namespace}")
                if limit:
                    query_parts.append(f"limit={limit}")
                if offset:
                    query_parts.append(f"offset={offset}")
                
                url = "/vectors"
                if query_parts:
                    url += "?" + "&".join(query_parts)
                
                future = executor.submit(forward_request, shard_url, "GET", url, timeout=120.0)
                futures[future] = shard_url
```
**Lines 965-985**:
- Builds query string with parameters
- Submits parallel GET requests to all shards
- Uses 120-second timeout (vectors can be large)

```python
            for future in as_completed(futures):
                shard_url = futures[future]
                try:
                    response, error = future.result()
                    if error:
                        errors.append({"shard": shard_url, "error": error})
                    else:
                        if response and response.get("vectors"):
                            all_vectors.extend(response["vectors"])
                except Exception as e:
                    errors.append({"shard": shard_url, "error": str(e)})
```
**Lines 987-997**: Collects vectors from all shards, handles errors.

```python
        # Sort by vector_id for consistent ordering
        all_vectors.sort(key=lambda x: x.get("vector_id", 0))
        
        # Apply global limit if specified
        if limit and len(all_vectors) > limit:
            all_vectors = all_vectors[:limit]
```
**Lines 999-1004**: Sorts by vector_id, applies global limit.

```python
        response = {
            "vectors": all_vectors,
            "count": len(all_vectors),
            "namespace": namespace,
            "limit": limit,
            "offset": offset,
            "shards_queried": len(SHARDS) - len(errors),
            "total_shards": len(SHARDS)
        }
        
        if errors:
            response["warnings"] = {
                "message": f"{len(errors)} shard(s) returned errors",
                "errors": errors
            }
        
        return jsonify(response), 200
```
**Lines 1006-1025**: Builds response with merged vectors, includes warnings if errors.

---

## 3. Shard Application (`app_v2.py`)

### Purpose
Flask application that runs as a shard instance. Handles vector operations for its assigned data partition.

### Key Sections

#### 3.1 Configuration Integration (Lines 71-91)

```python
# Import config
config_path = os.path.join(current_dir, "src", "config.py")
config_spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
get_config = config_module.get_config
```
**Lines 71-76**:
- Dynamically imports config module
- Uses `importlib` to load from file path
- Gets `get_config` function from module

```python
app = Flask(__name__)
# Base directory is parent of faiss-research (main project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load configuration
config = get_config()
```
**Lines 86-91**:
- Creates Flask app
- Calculates base directory
- **Loads config singleton**: Gets shard configuration

#### 3.2 Store Initialization (Lines 120-130)

```python
def get_store():
    global store
    if store is None:
        data_dir = config.get_data_dir()
        store = VectorStoreV2(data_dir=data_dir)
        store.load()
    return store
```
**Lines 120-127**:
- **Lazy initialization**: Creates store on first access
- **Uses config**: Gets shard-specific data directory
- **Loads existing data**: Restores indices and metadata
- **Singleton pattern**: Reuses same store instance

#### 3.3 Whoami Endpoint (Lines 130-150)

```python
@app.route('/whoami', methods=['GET'])
def whoami():
    try:
        store = get_store()
        stats = store.get_stats()
        namespaces_count = len(stats.get("namespaces", {}))
        return jsonify({
            "shard_id": config.shard_id,
            "shard_count": config.shard_count,
            "is_shard_mode": config.is_shard_mode(),
            "data_dir": str(config.shard_data_dir),
            "port": config.port,
            "namespaces_count": namespaces_count
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500
```
**Lines 130-150**:
- **Purpose**: Identify which shard this is
- Returns shard configuration and basic stats
- Useful for debugging and monitoring

#### 3.4 Vectors Endpoint (Lines 700-730)

```python
@app.route('/vectors', methods=['GET'])
def get_vectors():
    """
    Get all vectors from the database.
    
    Query parameters:
    - namespace: Optional namespace filter
    - limit: Optional limit (default: 1000)
    - offset: Optional offset for pagination (default: 0)
    """
    try:
        namespace = request.args.get('namespace')
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', type=int, default=0)
        
        store = get_store()
        vectors = store.get_all_vectors(namespace=namespace, limit=limit, offset=offset)
        
        return jsonify({
            "vectors": vectors,
            "count": len(vectors),
            "namespace": namespace,
            "limit": limit,
            "offset": offset
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500
```
**Lines 700-730**:
- **Purpose**: Return all vectors from this shard's database
- Supports namespace filtering and pagination
- Used by router to aggregate vectors from all shards

#### 3.5 Server Startup (Lines 850-870)

```python
if __name__ == "__main__":
    port = config.port
    shard_info = ""
    if config.is_shard_mode():
        shard_info = f" (Shard {config.shard_id}/{config.shard_count})"
    print(f"\nServer starting on http://localhost:{port}{shard_info}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
```
**Lines 850-870**:
- **Uses config port**: Each shard runs on different port
- **Shows shard info**: Displays which shard this is
- **Binds to 0.0.0.0**: Accepts connections from any interface

---

## 4. Vector Store (`src/store_v2.py`) - Key Changes

### Purpose
Core vector storage and retrieval logic. Modified to use config-provided data directories.

### Key Method: `get_all_vectors` (New)

```python
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
```
**Lines 1791-1802**: Method signature and docstring.

```python
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
```
**Lines 1804-1817**:
- Connects to SQLite database
- Builds SQL query dynamically
- Filters by namespace if provided
- Orders by vector_id for consistent results
- Adds pagination if limit specified

```python
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
```
**Lines 1819-1840**:
- Executes query and fetches results
- Parses JSON metadata for each vector
- Builds list of vector dictionaries
- Returns formatted vector data

### Bug Fix: `_rebuild_index` Method

**Problem**: `UnboundLocalError` when index variable not initialized in all code paths.

**Solution**: Explicitly initialize `index`, `index_type`, and `nlist` to `None` at start of block, then assign values in conditional branches. Added final check to ensure index is created.

---

## 5. UI Dashboard (`static/index.html`)

### Purpose
Professional web interface for searching vectors and viewing statistics. Monotone black & white design.

### Key Sections

#### 5.1 CSS Styling (Lines 7-400)

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
```
**Lines 8-12**: CSS reset - removes default browser styling.

```css
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    background: #ffffff;
    color: #000000;
    padding: 0;
    line-height: 1.6;
}
```
**Lines 14-20**: 
- System font stack for native look
- White background, black text
- Monotone color scheme

```css
.header {
    background: #000000;
    color: #ffffff;
    padding: 30px 40px;
    margin-bottom: 30px;
    border: 2px solid #000000;
}
```
**Lines 28-34**: 
- Black header with white text
- 2px solid border (boxy design)
- Consistent padding

```css
.tabs {
    display: flex;
    gap: 0;
    margin-bottom: 20px;
    border-bottom: 2px solid #000000;
}

.tab {
    padding: 12px 30px;
    background: #ffffff;
    color: #000000;
    border: 2px solid #000000;
    border-bottom: none;
    cursor: pointer;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.tab.active {
    background: #000000;
    color: #ffffff;
}
```
**Lines 400-430**: 
- Tab navigation styling
- Active tab: black background, white text
- Inactive tab: white background, black text
- Boxy borders

```css
.vectors-table-container {
    max-height: 600px;
    overflow-y: auto;
    overflow-x: auto;
    border: 2px solid #000000;
}

.vectors-table thead {
    position: sticky;
    top: 0;
    background: #000000;
    color: #ffffff;
    z-index: 10;
}
```
**Lines 450-470**: 
- Scrollable table container
- Sticky header (stays visible when scrolling)
- Black header with white text

#### 5.2 JavaScript Functions

##### `showTab` Function (Lines 540-560)

```javascript
function showTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    
    // Hide all containers
    document.getElementById('statsContainer').style.display = 'none';
    document.getElementById('vectorsContainer').style.display = 'none';
    
    // Activate clicked tab
    event.target.classList.add('active');
    
    if (tabName === 'stats') {
        document.getElementById('statsContainer').style.display = 'block';
    } else if (tabName === 'vectors') {
        document.getElementById('vectorsContainer').style.display = 'block';
        loadVectors();
    }
}
```
**Purpose**: Switch between Statistics and Vectors tabs.
- Removes active state from all tabs
- Hides all containers
- Shows selected container
- Loads vectors if vectors tab selected

##### `loadVectors` Function (Lines 709-730)

```javascript
async function loadVectors() {
    const container = document.getElementById('vectorsContainer');
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading vectors...</div>';

    try {
        const namespace = document.getElementById('namespace').value.trim() || undefined;
        const response = await fetch(`${API_BASE}/vectors?limit=${vectorsPerPage}&offset=${currentVectorsOffset}${namespace ? `&namespace=${encodeURIComponent(namespace)}` : ''}`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayVectors(data);
    } catch (error) {
        container.innerHTML = `
            <div class="error">
                ERROR: ${error.message}
            </div>
        `;
    }
}
```
**Purpose**: Fetch vectors from router API.
- Shows loading spinner
- Gets namespace from search form (if set)
- Builds API URL with pagination parameters
- Fetches data and displays or shows error

##### `displayVectors` Function (Lines 732-803)

```javascript
function displayVectors(data) {
    const container = document.getElementById('vectorsContainer');
    const vectors = data.vectors || [];
    const totalCount = data.count || 0;

    if (vectors.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>No Vectors Found</h3>
                <p>No vectors in the database. Ingest documents first.</p>
            </div>
        `;
        return;
    }

    let html = `
        <div class="table-controls">
            <div>
                <strong>Total: ${totalCount.toLocaleString()} vectors</strong>
                ${data.namespace ? ` | Namespace: ${data.namespace}` : ' | All namespaces'}
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <button onclick="previousPage()" ${currentVectorsOffset === 0 ? 'disabled' : ''}>PREV</button>
                <span>Page ${Math.floor(currentVectorsOffset / vectorsPerPage) + 1}</span>
                <button onclick="nextPage()" ${vectors.length < vectorsPerPage ? 'disabled' : ''}>NEXT</button>
                <button onclick="loadVectors()" style="margin-left: 10px;">REFRESH</button>
            </div>
        </div>
        <div class="vectors-table-container">
            <table class="vectors-table">
                <thead>
                    <tr>
                        <th>Vector ID</th>
                        <th>Namespace</th>
                        <th>Type</th>
                        <th>Document ID</th>
                        <th>Chunk ID</th>
                        <th>Text Preview</th>
                        <th>Created At</th>
                    </tr>
                </thead>
                <tbody>
    `;

    vectors.forEach((vector) => {
        const metadata = vector.metadata || {};
        const chunkText = metadata.chunk_text || '';
        const preview = chunkText.length > 100 ? chunkText.substring(0, 100) + '...' : chunkText;
        
        html += `
            <tr>
                <td><strong>${vector.vector_id || 'N/A'}</strong></td>
                <td>${escapeHtml(vector.namespace || 'default')}</td>
                <td><span class="badge">${vector.vector_type || 'N/A'}</span></td>
                <td>${escapeHtml(vector.doc_id || 'N/A')}</td>
                <td>${escapeHtml(vector.chunk_id || 'N/A')}</td>
                <td class="text-preview" title="${escapeHtml(chunkText)}">${escapeHtml(preview)}</td>
                <td>${vector.created_at || 'N/A'}</td>
            </tr>
        `;
    });

    html += `
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = html;
}
```
**Purpose**: Render vectors in table format.
- Checks for empty state
- Builds HTML with table controls (pagination, refresh)
- Creates table with headers
- Iterates through vectors, creating table rows
- Shows text preview (truncated to 100 chars, full text in tooltip)
- Uses `escapeHtml` to prevent XSS attacks

##### Pagination Functions (Lines 805-813)

```javascript
function nextPage() {
    currentVectorsOffset += vectorsPerPage;
    loadVectors();
}

function previousPage() {
    currentVectorsOffset = Math.max(0, currentVectorsOffset - vectorsPerPage);
    loadVectors();
}
```
**Purpose**: Navigate between pages of vectors.
- `nextPage`: Increases offset, reloads vectors
- `previousPage`: Decreases offset (minimum 0), reloads vectors

---

## Summary

This documentation explains:

1. **Configuration System**: How shards are configured via environment variables
2. **Router Service**: How requests are routed and results merged
3. **Shard Application**: How each shard instance operates
4. **Vector Store**: How data is stored and retrieved
5. **UI Dashboard**: How the web interface works

Each component is designed to work together to provide a scalable, sharded vector database with a professional user interface.
