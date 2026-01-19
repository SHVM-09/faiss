"""
Demo Script: Validate Production Features
==========================================
Tests the complete workflow:
1. Insert vectors
2. Search
3. Delete
4. Verify deletion
5. Compact
6. Create snapshot
7. Load snapshot
"""

import requests
import json
import numpy as np
import time
import sys
import os

# Add parent directory to path to import from main src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.embed import Embedder
from src.transform import transform_to_chunks

BASE_URL = "http://localhost:5001"


def print_response(title, response):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)


def test_insert():
    """Test inserting text vectors."""
    print("\n" + "="*60)
    print("TEST 1: Insert Text Vectors")
    print("="*60)
    
    # Create sample documents
    documents = [
        {"filename": "doc1.txt", "content": "Machine learning is a subset of artificial intelligence."},
        {"filename": "doc2.txt", "content": "Natural language processing helps computers understand human language."},
        {"filename": "doc3.txt", "content": "Deep learning uses neural networks with multiple layers."}
    ]
    
    # Transform to chunks
    chunks = transform_to_chunks(documents, chunk_size=100, overlap=20)
    print(f"Created {len(chunks)} chunks")
    
    # Embed
    embedder = Embedder()
    embedder.load()
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    vectors = embedder.embed(chunk_texts)
    
    # Insert via API
    response = requests.post(f"{BASE_URL}/insert", json={
        "type": "text",
        "vectors": vectors.tolist(),
        "chunks": chunks,
        "namespace": "test"
    })
    
    print_response("Insert Response", response)
    return response.status_code == 200


def test_search():
    """Test searching."""
    print("\n" + "="*60)
    print("TEST 2: Search Vectors")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/search", json={
        "query": "machine learning",
        "k": 5,
        "namespace": "test"
    })
    
    print_response("Search Response", response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nFound {len(data.get('text_results', []))} text results")
        return len(data.get('text_results', [])) > 0
    return False


def test_delete():
    """Test deletion."""
    print("\n" + "="*60)
    print("TEST 3: Delete by doc_id")
    print("="*60)
    
    # Delete doc1.txt
    response = requests.post(f"{BASE_URL}/delete", json={
        "doc_id": "doc1.txt",
        "namespace": "test"
    })
    
    print_response("Delete Response", response)
    return response.status_code == 200


def test_search_after_delete():
    """Verify deleted vectors don't appear in search."""
    print("\n" + "="*60)
    print("TEST 4: Search After Delete (should not return deleted)")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/search", json={
        "query": "machine learning",
        "k": 5,
        "namespace": "test"
    })
    
    print_response("Search After Delete", response)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('text_results', [])
        print(f"\nFound {len(results)} results")
        
        # Check that doc1.txt is not in results
        doc1_found = any("doc1.txt" in str(r.get('metadata', {}).get('source_file', '')) for r in results)
        if doc1_found:
            print("⚠ WARNING: doc1.txt still appears in results (should be deleted)")
            return False
        else:
            print("✓ doc1.txt correctly excluded from results")
            return True
    return False


def test_stats():
    """Get statistics."""
    print("\n" + "="*60)
    print("TEST 5: Get Statistics")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/stats")
    print_response("Stats Response", response)
    return response.status_code == 200


def test_compact():
    """Test compaction."""
    print("\n" + "="*60)
    print("TEST 6: Compact Index")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/compact", json={
        "namespace": "test"
    })
    
    print_response("Compact Response", response)
    return response.status_code == 200


def test_snapshot():
    """Test snapshot creation."""
    print("\n" + "="*60)
    print("TEST 7: Create Snapshot")
    print("="*60)
    
    snapshot_name = f"demo_{int(time.time())}"
    response = requests.post(f"{BASE_URL}/snapshot", json={
        "name": snapshot_name
    })
    
    print_response("Snapshot Response", response)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("snapshot_path")
    return None


def test_list_snapshots():
    """List snapshots."""
    print("\n" + "="*60)
    print("TEST 8: List Snapshots")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/snapshots")
    print_response("List Snapshots Response", response)
    return response.status_code == 200


def test_load_snapshot(snapshot_name):
    """Test loading a snapshot."""
    print("\n" + "="*60)
    print("TEST 9: Load Snapshot")
    print("="*60)
    
    # Extract name from path
    if snapshot_name:
        name = snapshot_name.split("snapshot_")[-1] if "snapshot_" in snapshot_name else snapshot_name
    else:
        # Get latest snapshot
        response = requests.get(f"{BASE_URL}/snapshots")
        if response.status_code == 200:
            snapshots = response.json().get("snapshots", [])
            if snapshots:
                name = snapshots[0]["name"]
            else:
                print("No snapshots available")
                return False
        else:
            return False
    
    response = requests.post(f"{BASE_URL}/load_snapshot", json={
        "name": name
    })
    
    print_response("Load Snapshot Response", response)
    return response.status_code == 200


def main():
    """Run all tests."""
    print("="*60)
    print("FAISS Production Features Validation")
    print("="*60)
    print("\nMake sure the server is running: python app_v2.py")
    print("Waiting 2 seconds for server to be ready...")
    time.sleep(2)
    
    # Check health
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not responding. Please start the server first.")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start the server: python app_v2.py")
        return
    
    results = []
    
    # Run tests
    results.append(("Insert", test_insert()))
    time.sleep(1)
    
    results.append(("Search", test_search()))
    time.sleep(1)
    
    results.append(("Delete", test_delete()))
    time.sleep(1)
    
    results.append(("Search After Delete", test_search_after_delete()))
    time.sleep(1)
    
    results.append(("Stats", test_stats()))
    time.sleep(1)
    
    results.append(("Compact", test_compact()))
    time.sleep(1)
    
    snapshot_path = test_snapshot()
    results.append(("Snapshot", snapshot_path is not None))
    time.sleep(1)
    
    results.append(("List Snapshots", test_list_snapshots()))
    time.sleep(1)
    
    # Test loading snapshot (optional, might restore deleted data)
    # results.append(("Load Snapshot", test_load_snapshot(snapshot_path)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"\n{passed_count}/{total_count} tests passed")


if __name__ == "__main__":
    main()
