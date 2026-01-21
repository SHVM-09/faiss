# Quick Reference - Router API (Port 5003)

## Start Services
```bash
# Terminal 1
make shard0

# Terminal 2  
make shard1

# Terminal 3
make router
```

## Base URL
```
http://localhost:5003
```

---

## All Endpoints

### 1. Router Info
```bash
curl http://localhost:5003/
curl http://localhost:5003/whoami
```

### 2. Health Check
```bash
curl http://localhost:5003/health
```

### 3. Ingest Documents
```bash
curl -X POST http://localhost:5003/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "../docs", "namespace": "alpha"}'
```

### 4. Search
```bash
curl -X POST http://localhost:5003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "k": 10}'
```

### 5. Delete
```bash
curl -X POST http://localhost:5003/delete \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "file.pdf", "namespace": "alpha"}'
```

### 6. Restore
```bash
curl -X POST http://localhost:5003/restore \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "file.pdf", "namespace": "alpha"}'
```

### 7. Stats
```bash
curl http://localhost:5003/stats
```

### 8. Reset
```bash
curl -X POST http://localhost:5003/reset \
  -H "Content-Type: application/json" \
  -d '{"namespace": "alpha"}'
```

---

## Common Parameters

### Ingest
- `docs_path` - Path to documents folder
- `namespace` - Namespace name (default: "default")
- `chunk_size` - Characters per chunk (default: 800)
- `overlap` - Overlap between chunks (default: 120)
- `extract_images` - Index images (default: true)

### Search
- `query` - Search text (required)
- `k` - Number of results (default: 5)
- `namespace` - Filter by namespace (optional)
- `vector_type` - "text", "image", or "both" (default: "both")

### Delete/Restore
- `doc_id` OR `chunk_id` - Document or chunk to delete/restore
- `namespace` - Namespace filter
- `hard_delete` - Permanent delete (default: false)

---

## Ports
- **Router**: 5003
- **Shard 0**: 5001
- **Shard 1**: 5002

---

## Tips
- Always use router (port 5003), not shards directly
- Namespace determines which shard stores data
- Search queries all shards automatically
- Check `/health` if something isn't working
