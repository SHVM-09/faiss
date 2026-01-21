"""
Demo Script: Validate Production Features
==========================================
Tests the complete workflow:
1. Ingest documents
2. Search
3. Delete (soft/hard)
4. Verify deletion
5. Restore (soft delete only)
6. Stats
7. Reset
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


def test_ingest():
    """Test ingesting documents."""
    print("\n" + "="*60)
    print("TEST 1: Ingest Documents")
    print("="*60)
    
    # Create a temporary directory with test files
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    
    # Create test files
    test_files = {
        "doc1.txt": "Machine learning is a subset of artificial intelligence.",
        "doc2.txt": "Natural language processing helps computers understand human language.",
        "doc3.txt": "Deep learning uses neural networks with multiple layers."
    }
    
    for filename, content in test_files.items():
        with open(os.path.join(temp_dir, filename), 'w') as f:
            f.write(content)
    
    # Ingest via API
    response = requests.post(f"{BASE_URL}/ingest", json={
        "docs_path": temp_dir,
        "chunk_size": 100,
        "overlap": 20,
        "namespace": "test"
    })
    
    print_response("Ingest Response", response)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
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


def test_restore():
    """Test restoring soft-deleted vectors."""
    print("\n" + "="*60)
    print("TEST 6: Restore Soft-Deleted Vectors")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/restore", json={
        "doc_id": "doc1.txt",
        "namespace": "test"
    })
    
    print_response("Restore Response", response)
    return response.status_code == 200


def test_hard_delete():
    """Test hard delete."""
    print("\n" + "="*60)
    print("TEST 7: Hard Delete (Permanent)")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/delete", json={
        "doc_id": "doc2.txt",
        "namespace": "test",
        "hard_delete": True
    })
    
    print_response("Hard Delete Response", response)
    return response.status_code == 200


def test_reset():
    """Test reset (clear all data)."""
    print("\n" + "="*60)
    print("TEST 8: Reset (Clear All Data)")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/reset")
    print_response("Reset Response", response)
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
    results.append(("Ingest", test_ingest()))
    time.sleep(2)
    
    results.append(("Search", test_search()))
    time.sleep(1)
    
    results.append(("Delete (Soft)", test_delete()))
    time.sleep(1)
    
    results.append(("Search After Delete", test_search_after_delete()))
    time.sleep(1)
    
    results.append(("Restore", test_restore()))
    time.sleep(1)
    
    results.append(("Hard Delete", test_hard_delete()))
    time.sleep(1)
    
    results.append(("Stats", test_stats()))
    time.sleep(1)
    
    # Reset is last (clears all data)
    results.append(("Reset", test_reset()))
    
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
