"""
Router service for sharded FAISS vector database.

Distributes documents across shards for parallel processing:
- Ingest: Splits files by filename hash across shards
- Delete/Restore: Routes by doc_id hash
- Search: Broadcasts to all shards and merges results
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from pathlib import Path

try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
    print("Warning: flask-cors not installed. CORS support disabled. Install with: pip install flask-cors")

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    import hashlib
    HAS_XXHASH = False

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# Configuration
SHARDS = os.environ.get("SHARDS", "http://127.0.0.1:5001,http://127.0.0.1:5002").split(",")
SHARDS = [s.strip() for s in SHARDS if s.strip()]
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "5003"))

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


def shard_for_namespace(namespace: str, shard_count: int) -> int:
    """
    Determine which shard should handle a namespace.
    
    Uses stable hashing to ensure consistent routing.
    """
    if shard_count <= 1:
        return 0
    hash_value = stable_hash(namespace)
    return hash_value % shard_count


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


def get_shard_url(shard_index: int) -> Optional[str]:
    """Get the URL for a shard by index."""
    if shard_index < 0 or shard_index >= len(SHARDS):
        return None
    return SHARDS[shard_index]


def cleanup_old_temp_dirs(docs_path_obj: Path, max_age_seconds: int = 600):
    """Cleanup old temporary shard directories that are older than max_age_seconds."""
    import time
    import shutil
    
    current_time = time.time()
    for item in docs_path_obj.iterdir():
        if item.is_dir() and item.name.startswith("_shard_"):
            try:
                # Check if directory is old enough to cleanup
                dir_age = current_time - item.stat().st_mtime
                if dir_age > max_age_seconds:
                    shutil.rmtree(item)
            except Exception:
                # Ignore errors - directory may be in use
                pass


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
    try:
        url = f"{shard_url.rstrip('/')}/{path.lstrip('/')}"
        if method == "GET":
            response = session.get(url, timeout=timeout)
        elif method == "POST":
            response = session.post(url, json=json_data, timeout=timeout)
        else:
            return None, f"Unsupported method: {method}"
        
        # Check for HTTP errors
        if response.status_code >= 400:
            # Try to get error details from response
            try:
                error_data = response.json()
                error_msg = error_data.get("error", f"{response.status_code} {response.reason}")
            except:
                error_msg = f"{response.status_code} Server Error: {response.reason} for url: {url}"
            return None, error_msg
        
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.Timeout as e:
        return None, f"Timeout after {timeout}s: {str(e)}"
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ============================================================================
# ROUTER ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def root():
    """Router API information."""
    return jsonify({
        "message": "FAISS Semantic Search Router",
        "version": "1.0.0",
        "shards": SHARDS,
        "shard_count": len(SHARDS),
        "endpoints": {
            "GET /": "Router information",
            "GET /ui": "Search UI Dashboard",
            "GET /health": "Health check (checks all shards)",
            "GET /whoami": "Router information",
            "POST /ingest": "Ingest documents (routed by docs_path+namespace for distribution)",
            "POST /search": "Search vectors (broadcast to all shards)",
            "POST /delete": "Delete by doc_id or chunk_id (routed by doc_id)",
            "POST /restore": "Restore soft-deleted vectors (routed by doc_id)",
            "GET /stats": "Get statistics (aggregated from all shards)",
            "GET /vectors": "Get all vectors from all shards",
            "POST /reset": "Reset - Clear all data (routed by namespace)"
        }
    }), 200


@app.route('/ui', methods=['GET'])
def ui():
    """Serve the search UI dashboard."""
    try:
        ui_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
        if os.path.exists(ui_path):
            with open(ui_path, 'r', encoding='utf-8') as f:
                return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        else:
            return jsonify({"error": "UI file not found", "path": ui_path}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check - checks all shards."""
    shard_statuses = {}
    all_healthy = True
    
    with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
        futures = {
            executor.submit(forward_request, shard_url, "GET", "/health"): shard_url
            for shard_url in SHARDS
        }
        
        for future in as_completed(futures):
            shard_url = futures[future]
            try:
                response, error = future.result()
                if error:
                    shard_statuses[shard_url] = {"ok": False, "error": error}
                    all_healthy = False
                else:
                    shard_statuses[shard_url] = response or {"ok": True}
            except Exception as e:
                shard_statuses[shard_url] = {"ok": False, "error": str(e)}
                all_healthy = False
    
    status_code = 200 if all_healthy else 207  # 207 Multi-Status
    return jsonify({
        "ok": all_healthy,
        "shards": shard_statuses
    }), status_code


@app.route('/whoami', methods=['GET'])
def whoami():
    """Get router information."""
    return jsonify({
        "type": "router",
        "shards": SHARDS,
        "shard_count": len(SHARDS),
        "port": ROUTER_PORT
    }), 200


@app.route('/ingest', methods=['POST'])
def ingest():
    """
    Ingest documents - distribute files across shards for parallel processing.
    
    Lists files in docs_path, splits them across shards based on filename hash,
    and sends each shard a subset of files to process in parallel.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        namespace = data.get("namespace", "default")
        docs_path = data.get("docs_path", "./docs")
        shard_count = len(SHARDS)
        
        # Convert to absolute path
        if not os.path.isabs(docs_path):
            # Try to resolve relative to common base
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            docs_path = os.path.join(base_dir, docs_path)
        
        docs_path_obj = Path(docs_path)
        
        # Cleanup old temporary directories from previous runs
        try:
            cleanup_old_temp_dirs(docs_path_obj, max_age_seconds=600)  # Cleanup dirs older than 10 minutes
        except Exception:
            pass  # Ignore cleanup errors
        
        if not docs_path_obj.exists():
            # If path doesn't exist, fall back to single shard routing
            routing_key = f"{docs_path}:{namespace}"
            shard_index = shard_for_doc_id(routing_key, shard_count)
            shard_url = get_shard_url(shard_index)
            
            if not shard_url:
                return jsonify({"error": f"Invalid shard index: {shard_index}"}), 500
            
            response, error = forward_request(shard_url, "POST", "/ingest", json_data=data, timeout=300.0)
            
            if error:
                return jsonify({
                    "error": f"Shard error: {error}",
                    "shard": shard_url,
                    "namespace": namespace
                }), 502
            
            if response:
                response["routed_to_shard"] = shard_index
                response["shard_url"] = shard_url
            
            return jsonify(response), 200
        
        # List all files in the directory
        all_files = []
        for ext in ['*.txt', '*.md', '*.pdf', '*.PDF', '*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.svg']:
            all_files.extend(docs_path_obj.glob(ext))
        
        if not all_files:
            # No files found, route to single shard
            routing_key = f"{docs_path}:{namespace}"
            shard_index = shard_for_doc_id(routing_key, shard_count)
            shard_url = get_shard_url(shard_index)
            
            response, error = forward_request(shard_url, "POST", "/ingest", json_data=data, timeout=300.0)
            
            if error:
                return jsonify({"error": f"Shard error: {error}"}), 502
            
            if response:
                response["routed_to_shard"] = shard_index
                response["shard_url"] = shard_url
            
            return jsonify(response), 200
        
        # Distribute files across shards based on filename hash
        files_by_shard = {i: [] for i in range(shard_count)}
        
        for file_path in all_files:
            # Use filename (not full path) for routing to ensure same file always goes to same shard
            filename = file_path.name
            file_routing_key = f"{filename}:{namespace}"
            shard_index = shard_for_doc_id(file_routing_key, shard_count)
            files_by_shard[shard_index].append(str(file_path))
        
        # Create separate requests for each shard with filtered file lists
        shard_responses = {}
        errors = []
        
        with ThreadPoolExecutor(max_workers=shard_count) as executor:
            futures = {}
            
            for shard_index, files in files_by_shard.items():
                if not files:
                    continue  # Skip shards with no files
                
                shard_url = get_shard_url(shard_index)
                if not shard_url:
                    continue
                
                # Create a temporary directory with only this shard's files
                # OR: Modify request to include file filter
                # For now, we'll create symlinks or copy files to temp dirs
                # Actually, simpler: create subdirectories per shard
                
                # Create shard-specific subdirectory with symlinks/copies to files
                shard_subdir = os.path.join(str(docs_path_obj), f"_shard_{shard_index}")
                try:
                    os.makedirs(shard_subdir, exist_ok=True)
                    # Verify directory was created and is writable
                    if not os.path.exists(shard_subdir):
                        raise Exception(f"Failed to create shard subdirectory: {shard_subdir}")
                    if not os.access(shard_subdir, os.W_OK):
                        raise Exception(f"Shard subdirectory is not writable: {shard_subdir}")
                except Exception as dir_error:
                    error_msg = f"Error [Shard {shard_index}]: Failed to create subdirectory {shard_subdir}: {dir_error}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                # Copy files to shard subdirectory (more reliable than symlinks)
                # This ensures files are accessible even if symlinks fail
                import shutil
                copied_files = []
                for file_path in files:
                    src = Path(file_path).resolve()  # Get absolute path
                    if not src.exists():
                        print(f"Warning [Shard {shard_index}]: Source file does not exist: {src}")
                        continue
                    
                    dst = Path(shard_subdir) / src.name
                    
                    # Remove existing file/symlink
                    if dst.exists() or dst.is_symlink():
                        try:
                            if dst.is_symlink():
                                dst.unlink()
                            elif dst.is_file():
                                dst.unlink()
                        except Exception as e:
                            print(f"Warning [Shard {shard_index}]: Failed to remove existing {dst}: {e}")
                    
                    # Copy file (more reliable than symlinks)
                    try:
                        shutil.copy2(str(src), str(dst))
                        # Verify copy worked
                        if not dst.exists() or not dst.is_file():
                            raise OSError(f"Copy verification failed: {dst}")
                        # Verify file is readable and not empty (if source wasn't empty)
                        src_size = src.stat().st_size
                        dst_size = dst.stat().st_size
                        if dst_size == 0 and src_size > 0:
                            raise OSError(f"Copied file is empty: {dst} (source size: {src_size})")
                        if dst_size != src_size:
                            raise OSError(f"File size mismatch: {dst} (expected {src_size}, got {dst_size})")
                        copied_files.append(dst.name)
                    except Exception as copy_error:
                        error_msg = f"Error [Shard {shard_index}]: Failed to copy {src} to {dst}: {copy_error}"
                        print(error_msg)
                        raise Exception(error_msg)
                
                # Verify at least one file was copied
                if not copied_files:
                    raise Exception(f"No files were successfully copied to shard {shard_index} subdirectory")
                
                print(f"Shard {shard_index}: Copied {len(copied_files)} files to {shard_subdir}")
                
                # Create request for this shard
                shard_data = data.copy()
                # Use absolute path for docs_path to avoid path resolution issues
                shard_data["docs_path"] = os.path.abspath(shard_subdir)
                
                # Verify the subdirectory exists and has files before sending request
                if not os.path.exists(shard_subdir):
                    raise Exception(f"Shard subdirectory does not exist: {shard_subdir}")
                
                files_in_subdir = list(Path(shard_subdir).glob("*"))
                if not files_in_subdir:
                    raise Exception(f"No files found in shard subdirectory: {shard_subdir}")
                
                print(f"Shard {shard_index}: Sending ingest request with {len(files_in_subdir)} files from {shard_subdir}")
                
                # Submit request with longer timeout for PDF processing
                future = executor.submit(forward_request, shard_url, "POST", "/ingest", json_data=shard_data, timeout=600.0)
                futures[future] = (shard_index, shard_url, shard_subdir, files)
            
            # Collect results and wait for all to complete
            shard_dirs_to_cleanup = []
            shard_files_info = {}  # Track which files went to which shard
            
            for future in as_completed(futures):
                shard_index, shard_url, shard_subdir, files = futures[future]
                shard_dirs_to_cleanup.append(shard_subdir)  # Track for cleanup
                shard_files_info[shard_url] = {
                    "shard_index": shard_index,
                    "files": files,
                    "subdir": shard_subdir
                }
                try:
                    response, error = future.result()
                    if error:
                        # Enhanced error reporting
                        error_details = {
                            "shard": shard_url,
                            "shard_index": shard_index,
                            "error": error,
                            "files_count": len(files),
                            "files": [os.path.basename(f) for f in files[:5]],  # First 5 files
                            "subdir": shard_subdir
                        }
                        if len(files) > 5:
                            error_details["files_note"] = f"... and {len(files) - 5} more"
                        errors.append(error_details)
                        print(f"ERROR [Shard {shard_index}]: {error}")
                    else:
                        shard_responses[shard_url] = response
                        if response:
                            response["files_processed"] = len(files)
                            response["routed_to_shard"] = shard_index
                            response["files"] = [os.path.basename(f) for f in files]
                        print(f"SUCCESS [Shard {shard_index}]: Processed {len(files)} files")
                except Exception as e:
                    error_details = {
                        "shard": shard_url,
                        "shard_index": shard_index,
                        "error": str(e),
                        "files_count": len(files),
                        "files": [os.path.basename(f) for f in files[:5]]
                    }
                    errors.append(error_details)
                    print(f"EXCEPTION [Shard {shard_index}]: {str(e)}")
            
            # Cleanup temporary directories after processing completes
            # The HTTP response has returned, but we need to ensure files stay available
            # for any background processing (like PDF image conversion)
            import time
            import shutil
            import threading
            
            def safe_cleanup(shard_dirs, wait_seconds=60):
                """
                Safely cleanup temporary directories after ensuring processing is complete.
                
                Waits for background processing (PDF images) to complete before cleanup.
                """
                # Wait for background processing to complete
                time.sleep(wait_seconds)
                
                # Cleanup each directory
                for shard_subdir in shard_dirs:
                    try:
                        if os.path.exists(shard_subdir):
                            # Try multiple times in case files are still in use
                            for attempt in range(3):
                                try:
                                    shutil.rmtree(shard_subdir)
                                    break  # Success, exit retry loop
                                except (OSError, PermissionError) as e:
                                    if attempt < 2:
                                        time.sleep(5)  # Wait 5 seconds before retry
                                    else:
                                        # Final attempt failed, log but continue
                                        print(f"Warning: Could not cleanup {shard_subdir} after {attempt+1} attempts")
                    except Exception as cleanup_error:
                        # Ignore cleanup errors - files may still be in use
                        pass
            
            # Start cleanup in background thread (non-blocking)
            # Wait 60 seconds to ensure PDF image processing completes
            if shard_dirs_to_cleanup:
                cleanup_thread = threading.Thread(
                    target=safe_cleanup,
                    args=(shard_dirs_to_cleanup, 60),
                    daemon=True
                )
                cleanup_thread.start()
            
            # Also cleanup any old temp directories from previous runs (older than 10 minutes)
            try:
                cleanup_old_temp_dirs(docs_path_obj, max_age_seconds=600)
            except:
                pass
        
        # Aggregate results
        total_files = sum(len(files) for files in files_by_shard.values())
        total_processed = sum(1 for r in shard_responses.values() if r)
        
        aggregated_response = {
            "success": len(errors) == 0,
            "message": f"Documents ingested across {len(shard_responses)} shard(s)",
            "total_files": total_files,
            "shards_processed": len(shard_responses),
            "shard_responses": shard_responses,
            "file_distribution": {f"shard_{i}": len(files) for i, files in files_by_shard.items()}
        }
        
        if errors:
            aggregated_response["warnings"] = {
                "message": f"{len(errors)} shard(s) returned errors",
                "errors": errors
            }
        
        return jsonify(aggregated_response), 200 if len(errors) == 0 else 207
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Search vectors - broadcast to ALL shards and merge results.
    
    This ensures results from both shard 0 and shard 1 are included.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        query = data.get("query", "").strip()
        k = data.get("k", 5)
        namespace = data.get("namespace")
        vector_type = data.get("vector_type", "both")
        
        if not query:
            return jsonify({"error": "query cannot be empty"}), 400
        if k <= 0:
            return jsonify({"error": "k must be positive"}), 400
        
        # ALWAYS broadcast search to ALL shards in parallel
        # This ensures results from both shard 0 and shard 1
        shard_results = {}
        errors = []
        
        print(f"Search: Broadcasting to {len(SHARDS)} shards: {SHARDS}")
        
        with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
            futures = {
                executor.submit(forward_request, shard_url, "POST", "/search", json_data=data, timeout=60.0): shard_url
                for shard_url in SHARDS
            }
            
            for future in as_completed(futures):
                shard_url = futures[future]
                try:
                    response, error = future.result()
                    if error:
                        print(f"Search ERROR [Shard {shard_url}]: {error}")
                        errors.append({"shard": shard_url, "error": error})
                    else:
                        print(f"Search SUCCESS [Shard {shard_url}]: Got {len(response.get('text_results', []))} text results, {len(response.get('image_results', []))} image results")
                        shard_results[shard_url] = response
                except Exception as e:
                    print(f"Search EXCEPTION [Shard {shard_url}]: {str(e)}")
                    errors.append({"shard": shard_url, "error": str(e)})
        
        # Merge results
        merged_text_results = []
        merged_image_results = []
        seen_text = set()  # For de-duplication
        seen_image = set()
        
        for shard_url, result in shard_results.items():
            if not result:
                continue
            
            # Merge text results
            text_results = result.get("text_results", [])
            for item in text_results:
                # De-duplicate by (source_file, chunk_id) or vector_id
                dedup_key = (
                    item.get("metadata", {}).get("source_file", ""),
                    item.get("metadata", {}).get("chunk_id", ""),
                    item.get("vector_id")
                )
                if dedup_key not in seen_text:
                    seen_text.add(dedup_key)
                    merged_text_results.append(item)
            
            # Merge image results
            image_results = result.get("image_results", [])
            for item in image_results:
                # De-duplicate by (source_file, image_id/page_num) or vector_id
                metadata = item.get("metadata", {})
                dedup_key = (
                    metadata.get("source_file", ""),
                    metadata.get("image_id", ""),
                    metadata.get("page_num"),
                    item.get("vector_id")
                )
                if dedup_key not in seen_image:
                    seen_image.add(dedup_key)
                    merged_image_results.append(item)
        
        # Sort by score (descending) and take top-k
        merged_text_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        merged_image_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Filter by vector_type
        final_text_results = merged_text_results[:k] if vector_type in ["text", "both"] else []
        final_image_results = merged_image_results[:k] if vector_type in ["image", "both"] else []
        
        # Build response with detailed shard information
        response = {
            "query": query,
            "k": k,
            "namespace": namespace,
            "text_results": final_text_results,
            "image_results": final_image_results,
            "shards_queried": len(shard_results),
            "total_shards": len(SHARDS),
            "shards_searched": list(shard_results.keys())
        }
        
        # Add per-shard result counts for debugging
        shard_counts = {}
        for shard_url, result in shard_results.items():
            shard_counts[shard_url] = {
                "text_results": len(result.get("text_results", [])),
                "image_results": len(result.get("image_results", []))
            }
        response["shard_result_counts"] = shard_counts
        
        if errors:
            response["warnings"] = {
                "message": f"{len(errors)} shard(s) returned errors (results from {len(shard_results)} healthy shard(s))",
                "errors": errors
            }
        
        print(f"Search COMPLETE: Merged {len(final_text_results)} text + {len(final_image_results)} image results from {len(shard_results)}/{len(SHARDS)} shards")
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/delete', methods=['POST'])
def delete():
    """
    Delete vectors - route to shard based on doc_id or chunk_id.
    
    If doc_id is provided, route based on doc_id hash.
    If chunk_id is provided, extract doc_id from chunk_id and route.
    If neither, broadcast to all shards.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        doc_id = data.get("doc_id")
        chunk_id = data.get("chunk_id")
        namespace = data.get("namespace", "default")
        shard_count = len(SHARDS)
        
        # Determine routing key
        if doc_id:
            # Route based on doc_id - distributes documents across shards
            routing_key = doc_id
            shard_index = shard_for_doc_id(routing_key, shard_count)
        elif chunk_id:
            # Extract doc_id from chunk_id (format: "doc_id::chunk_0")
            if "::" in chunk_id:
                routing_key = chunk_id.split("::")[0]
            else:
                routing_key = chunk_id
            shard_index = shard_for_doc_id(routing_key, shard_count)
        else:
            # No doc_id or chunk_id - broadcast to all shards
            shard_responses = {}
            errors = []
            
            with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
                futures = {
                    executor.submit(forward_request, shard_url, "POST", "/delete", json_data=data, timeout=60.0): shard_url
                    for shard_url in SHARDS
                }
                
                for future in as_completed(futures):
                    shard_url = futures[future]
                    try:
                        response, error = future.result()
                        if error:
                            errors.append({"shard": shard_url, "error": error})
                        else:
                            shard_responses[shard_url] = response
                    except Exception as e:
                        errors.append({"shard": shard_url, "error": str(e)})
            
            response = {
                "success": len(errors) == 0,
                "shards": shard_responses,
                "shard_count": len(SHARDS)
            }
            
            if errors:
                response["warnings"] = {
                    "message": f"{len(errors)} shard(s) returned errors",
                    "errors": errors
                }
            
            return jsonify(response), 200
        
        shard_url = get_shard_url(shard_index)
        
        if not shard_url:
            return jsonify({"error": f"Invalid shard index: {shard_index}"}), 500
        
        # Forward request to appropriate shard
        response, error = forward_request(shard_url, "POST", "/delete", json_data=data, timeout=60.0)
        
        if error:
            return jsonify({
                "error": f"Shard error: {error}",
                "shard": shard_url,
                "namespace": namespace,
                "routing_key": routing_key
            }), 502
        
        # Add routing info to response
        if response:
            response["routed_to_shard"] = shard_index
            response["shard_url"] = shard_url
            response["routing_key"] = routing_key
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/restore', methods=['POST'])
def restore():
    """
    Restore soft-deleted vectors - route to shard based on doc_id or chunk_id.
    
    Same routing logic as delete.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        doc_id = data.get("doc_id")
        chunk_id = data.get("chunk_id")
        namespace = data.get("namespace", "default")
        shard_count = len(SHARDS)
        
        # Determine routing key
        if doc_id:
            routing_key = doc_id
            shard_index = shard_for_doc_id(routing_key, shard_count)
        elif chunk_id:
            if "::" in chunk_id:
                routing_key = chunk_id.split("::")[0]
            else:
                routing_key = chunk_id
            shard_index = shard_for_doc_id(routing_key, shard_count)
        else:
            # Broadcast to all shards
            shard_responses = {}
            errors = []
            
            with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
                futures = {
                    executor.submit(forward_request, shard_url, "POST", "/restore", json_data=data, timeout=60.0): shard_url
                    for shard_url in SHARDS
                }
                
                for future in as_completed(futures):
                    shard_url = futures[future]
                    try:
                        response, error = future.result()
                        if error:
                            errors.append({"shard": shard_url, "error": error})
                        else:
                            shard_responses[shard_url] = response
                    except Exception as e:
                        errors.append({"shard": shard_url, "error": str(e)})
            
            response = {
                "success": len(errors) == 0,
                "shards": shard_responses,
                "shard_count": len(SHARDS)
            }
            
            if errors:
                response["warnings"] = {
                    "message": f"{len(errors)} shard(s) returned errors",
                    "errors": errors
                }
            
            return jsonify(response), 200
        
        shard_url = get_shard_url(shard_index)
        
        if not shard_url:
            return jsonify({"error": f"Invalid shard index: {shard_index}"}), 500
        
        # Forward request to appropriate shard
        response, error = forward_request(shard_url, "POST", "/restore", json_data=data, timeout=60.0)
        
        if error:
            return jsonify({
                "error": f"Shard error: {error}",
                "shard": shard_url,
                "namespace": namespace,
                "routing_key": routing_key
            }), 502
        
        # Add routing info to response
        if response:
            response["routed_to_shard"] = shard_index
            response["shard_url"] = shard_url
            response["routing_key"] = routing_key
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics - aggregate from all shards."""
    shard_stats = {}
    errors = []
    
    with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
        futures = {
            executor.submit(forward_request, shard_url, "GET", "/stats"): shard_url
            for shard_url in SHARDS
        }
        
        for future in as_completed(futures):
            shard_url = futures[future]
            try:
                response, error = future.result()
                if error:
                    errors.append({"shard": shard_url, "error": error})
                else:
                    shard_stats[shard_url] = response
            except Exception as e:
                errors.append({"shard": shard_url, "error": str(e)})
    
    # Aggregate statistics
    aggregated = {
        "total_active_vectors": 0,
        "total_deleted_vectors": 0,
        "namespace_counts": {},
        "namespaces": {}
    }
    
    for shard_url, stats_data in shard_stats.items():
        if not stats_data:
            continue
        
        aggregated["total_active_vectors"] += stats_data.get("total_active_vectors", 0)
        aggregated["total_deleted_vectors"] += stats_data.get("total_deleted_vectors", 0)
        
        # Merge namespace counts
        namespace_counts = stats_data.get("namespace_counts", {})
        for ns, count in namespace_counts.items():
            aggregated["namespace_counts"][ns] = aggregated["namespace_counts"].get(ns, 0) + count
        
        # Merge namespace info (keep first occurrence)
        namespaces_info = stats_data.get("namespaces", {})
        for ns, info in namespaces_info.items():
            if ns not in aggregated["namespaces"]:
                aggregated["namespaces"][ns] = info
    
    response = {
        "aggregated": aggregated,
        "shards": shard_stats,
        "shard_count": len(SHARDS),
        "shards_queried": len(shard_stats)
    }
    
    if errors:
        response["warnings"] = {
            "message": f"{len(errors)} shard(s) returned errors",
            "errors": errors
        }
    
    return jsonify(response), 200


@app.route('/vectors', methods=['GET'])
def get_vectors():
    """
    Get all vectors from all shards.
    
    Query parameters:
    - namespace: Optional namespace filter
    - limit: Optional limit per shard (default: 10000)
    - offset: Optional offset for pagination (default: 0)
    """
    try:
        namespace = request.args.get('namespace')
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', type=int, default=0)
        
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
        
        # Sort by vector_id for consistent ordering
        all_vectors.sort(key=lambda x: x.get("vector_id", 0))
        
        # Apply global limit if specified
        if limit and len(all_vectors) > limit:
            all_vectors = all_vectors[:limit]
        
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
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """
    Reset - clear all data from ALL shards.
    
    Always broadcasts to all shards to ensure complete cleanup.
    This ensures both shards are cleared properly.
    """
    try:
        data = request.get_json() or {}
        
        # ALWAYS broadcast to all shards - reset should clear everything
        # This ensures both shard 0 and shard 1 are cleared
        shard_responses = {}
        errors = []
        
        with ThreadPoolExecutor(max_workers=len(SHARDS)) as executor:
            futures = {
                executor.submit(forward_request, shard_url, "POST", "/reset", json_data=data, timeout=120.0): shard_url
                for shard_url in SHARDS
            }
            
            for future in as_completed(futures):
                shard_url = futures[future]
                try:
                    response, error = future.result()
                    if error:
                        errors.append({"shard": shard_url, "error": error})
                    else:
                        shard_responses[shard_url] = response
                except Exception as e:
                    errors.append({"shard": shard_url, "error": str(e)})
        
        # Verify all shards responded
        all_cleared = len(shard_responses) == len(SHARDS) and len(errors) == 0
        
        response = {
            "success": all_cleared,
            "message": f"Reset completed on {len(shard_responses)}/{len(SHARDS)} shard(s)",
            "shards": shard_responses,
            "shard_count": len(SHARDS),
            "shards_cleared": len(shard_responses)
        }
        
        if errors:
            response["warnings"] = {
                "message": f"{len(errors)} shard(s) returned errors",
                "errors": errors
            }
        
        status_code = 200 if all_cleared else 207  # 207 Multi-Status if some failed
        return jsonify(response), status_code
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("FAISS Semantic Search Router")
    print("="*60)
    print(f"\nRouter Port: {ROUTER_PORT}")
    print(f"Shards: {', '.join(SHARDS)}")
    print(f"Shard Count: {len(SHARDS)}")
    print("\n" + "="*60)
    print("Router Endpoints:")
    print("  GET  /          - Router information")
    print("  GET  /ui        - Search UI Dashboard")
    print("  GET  /health     - Health check (all shards)")
    print("  GET  /whoami     - Router information")
    print("  POST /ingest     - Ingest documents (routed)")
    print("  POST /search     - Search vectors (broadcast)")
    print("  POST /delete     - Delete vectors (routed)")
    print("  POST /restore    - Restore vectors (routed)")
    print("  GET  /stats      - Statistics (aggregated)")
    print("  POST /reset      - Reset data (always broadcasts to all shards)")
    print("="*60)
    print(f"\nRouter starting on http://localhost:{ROUTER_PORT}")
    print(f"UI Dashboard: http://localhost:{ROUTER_PORT}/ui\n")
    
    app.run(host='0.0.0.0', port=ROUTER_PORT, debug=True)
