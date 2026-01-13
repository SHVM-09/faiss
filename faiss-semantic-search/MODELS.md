# Model Guide: Current Models & Alternatives

This document explains the models currently used, their characteristics, and better alternatives you can use.

---

## Current Models

### 1. Text Embedding Model

**Current:** `sentence-transformers/all-MiniLM-L6-v2`

**Characteristics:**
- **Dimensions:** 384
- **Size:** ~80MB
- **Speed:** Very fast (optimized for speed)
- **Accuracy:** Good for general semantic search
- **Best for:** General-purpose semantic search, fast retrieval
- **Trade-off:** Prioritizes speed over maximum accuracy

**Why it's used:**
- Fast inference (important for real-time search)
- Small model size (easy to deploy)
- Good balance between speed and accuracy
- Works well for most use cases

---

### 2. Image Embedding Model

**Current:** `openai/clip-vit-base-patch32`

**Characteristics:**
- **Dimensions:** 512
- **Size:** ~500MB
- **Speed:** Fast
- **Accuracy:** Good for general image-text search
- **Best for:** Text-to-image search, image classification
- **Trade-off:** Released in 2021, newer models available

**Why it's used:**
- Industry standard for image-text search
- Well-tested and reliable
- Good balance of performance and speed
- Easy to use with HuggingFace

---

## Better Alternatives

### Text Embedding Models (Better Accuracy)

#### Option 1: `sentence-transformers/all-mpnet-base-v2` ⭐ **RECOMMENDED**

**Characteristics:**
- **Dimensions:** 768
- **Size:** ~420MB
- **Speed:** Moderate (slower than MiniLM)
- **Accuracy:** **Significantly better** than MiniLM
- **Best for:** When accuracy is more important than speed

**Performance:**
- Better semantic understanding
- Better handling of complex queries
- Better for domain-specific content

**Trade-off:** Slower but more accurate

---

#### Option 2: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

**Characteristics:**
- **Dimensions:** 768
- **Size:** ~420MB
- **Speed:** Moderate
- **Accuracy:** Very good
- **Best for:** Multilingual content (100+ languages)

**Performance:**
- Works with multiple languages
- Good for international applications

---

#### Option 3: `BAAI/bge-large-en-v1.5` ⭐ **BEST ACCURACY**

**Characteristics:**
- **Dimensions:** 1024
- **Size:** ~1.3GB
- **Speed:** Slower
- **Accuracy:** **State-of-the-art** for English
- **Best for:** Maximum accuracy, English-only content

**Performance:**
- Best accuracy on benchmarks
- Excellent for complex queries
- Better understanding of context

**Trade-off:** Much slower and larger

---

#### Option 4: `sentence-transformers/all-MiniLM-L12-v2`

**Characteristics:**
- **Dimensions:** 384
- **Size:** ~120MB
- **Speed:** Fast (slightly slower than L6)
- **Accuracy:** Better than L6, similar speed

**Performance:**
- Better than current model with similar speed
- Good upgrade path

---

### Image Embedding Models (Better Accuracy)

#### Option 1: `openai/clip-vit-large-patch14` ⭐ **RECOMMENDED**

**Characteristics:**
- **Dimensions:** 768
- **Size:** ~890MB
- **Speed:** Moderate (slower than base)
- **Accuracy:** **Significantly better** than base model
- **Best for:** Better image understanding

**Performance:**
- Better image-text matching
- Better understanding of complex scenes
- Better for detailed image search

**Trade-off:** Slower and larger

---

#### Option 2: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

**Characteristics:**
- **Dimensions:** 1024
- **Size:** ~2.5GB
- **Speed:** Slower
- **Accuracy:** **Very high** (trained on large dataset)
- **Best for:** Maximum accuracy

**Performance:**
- State-of-the-art CLIP model
- Trained on 2B image-text pairs
- Excellent for complex queries

**Trade-off:** Much slower and larger

---

#### Option 3: `sentence-transformers/clip-ViT-B-32`

**Characteristics:**
- **Dimensions:** 512
- **Size:** ~500MB
- **Speed:** Similar to current
- **Accuracy:** Similar to current (alternative implementation)

**Performance:**
- Alternative CLIP implementation
- Same performance, different library

---

## Model Comparison Table

### Text Models

| Model | Dimensions | Size | Speed | Accuracy | Best For |
|-------|-----------|------|-------|----------|----------|
| **all-MiniLM-L6-v2** (current) | 384 | 80MB | ⚡⚡⚡ Very Fast | ⭐⭐⭐ Good | General use, speed priority |
| all-MiniLM-L12-v2 | 384 | 120MB | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Better | Better accuracy, similar speed |
| **all-mpnet-base-v2** ⭐ | 768 | 420MB | ⚡⚡ Moderate | ⭐⭐⭐⭐⭐ Excellent | Best balance |
| paraphrase-multilingual-mpnet-base-v2 | 768 | 420MB | ⚡⚡ Moderate | ⭐⭐⭐⭐⭐ Excellent | Multilingual |
| **BAAI/bge-large-en-v1.5** ⭐⭐ | 1024 | 1.3GB | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | Maximum accuracy |

### Image Models

| Model | Dimensions | Size | Speed | Accuracy | Best For |
|-------|-----------|------|-------|----------|----------|
| **clip-vit-base-patch32** (current) | 512 | 500MB | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | General use, speed priority |
| **clip-vit-large-patch14** ⭐ | 768 | 890MB | ⚡⚡ Moderate | ⭐⭐⭐⭐⭐ Excellent | Better accuracy |
| CLIP-ViT-H-14-laion2B | 1024 | 2.5GB | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | Maximum accuracy |

---

## How to Switch Models

### Changing Text Embedding Model

Edit `src/embed.py`:

```python
# Current (line 54):
self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Better accuracy (recommended):
self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
self.dimension = 768  # Update dimension

# Best accuracy:
self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
self.dimension = 1024  # Update dimension
```

**Important:** If you change dimensions, you need to:
1. Clear the existing index: `POST /clear`
2. Re-ingest all documents
3. Update `self.dimension` in `src/store.py` if needed

---

### Changing Image Embedding Model

Edit `src/image_embed.py`:

```python
# Current (line 64):
self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
self.dimension = 512

# Better accuracy (recommended):
self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
self.dimension = 768  # Update dimension

# Best accuracy:
self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
self.dimension = 1024  # Update dimension
```

**Important:** Same as text - clear index and re-ingest after changing dimensions.

---

## Recommendations

### For Most Users (Balanced)

**Text:** `sentence-transformers/all-mpnet-base-v2`
- Better accuracy than current
- Still reasonably fast
- Good for most use cases

**Image:** `openai/clip-vit-large-patch14`
- Better accuracy than current
- Still reasonably fast
- Good for most use cases

### For Maximum Accuracy

**Text:** `BAAI/bge-large-en-v1.5`
- Best accuracy available
- Slower but worth it for accuracy-critical applications

**Image:** `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- Best accuracy available
- Slower but worth it for accuracy-critical applications

### For Speed Priority (Current)

**Text:** `sentence-transformers/all-MiniLM-L6-v2` (current)
- Fastest option
- Good enough for many use cases

**Image:** `openai/clip-vit-base-patch32` (current)
- Fastest option
- Good enough for many use cases

---

## Performance Benchmarks

### Text Embedding Models (MTEB Benchmark)

| Model | Average Score | Speed (docs/sec) |
|-------|--------------|------------------|
| all-MiniLM-L6-v2 (current) | 57.7 | ~10,000 |
| all-mpnet-base-v2 | **61.6** | ~3,000 |
| BAAI/bge-large-en-v1.5 | **64.2** | ~1,500 |

### Image Embedding Models (ImageNet Zero-Shot)

| Model | Top-1 Accuracy | Speed (images/sec) |
|-------|---------------|-------------------|
| clip-vit-base-patch32 (current) | 68.3% | ~50 |
| clip-vit-large-patch14 | **75.5%** | ~20 |
| CLIP-ViT-H-14-laion2B | **78.0%** | ~10 |

---

## When to Upgrade

### Upgrade Text Model If:
- ✅ You need better search accuracy
- ✅ You have complex queries
- ✅ You have domain-specific content
- ✅ Speed is less critical than accuracy

### Upgrade Image Model If:
- ✅ You need better image-text matching
- ✅ You have complex image queries
- ✅ You need to understand detailed scenes
- ✅ Speed is less critical than accuracy

### Keep Current Models If:
- ✅ Speed is critical
- ✅ Current accuracy is sufficient
- ✅ You have limited resources
- ✅ You're doing general-purpose search

---

## Quick Upgrade Guide

1. **Backup your data** (copy `data/` folder)

2. **Update model in code:**
   - Edit `src/embed.py` for text model
   - Edit `src/image_embed.py` for image model
   - Update dimension if changed

3. **Clear index:**
   ```bash
   curl -X POST http://localhost:5001/clear
   ```

4. **Re-ingest documents:**
   ```bash
   curl -X POST http://localhost:5001/ingest \
     -H "Content-Type: application/json" \
     -d '{"docs_path": "./docs"}'
   ```

5. **Test search quality** - you should see better results!

---

## Summary

**Current Models:**
- ✅ Fast and efficient
- ✅ Good for general use
- ⚠️ Not the most accurate

**Recommended Upgrade:**
- Text: `all-mpnet-base-v2` (better accuracy, still fast)
- Image: `clip-vit-large-patch14` (better accuracy, still fast)

**Best Accuracy:**
- Text: `BAAI/bge-large-en-v1.5` (best, but slower)
- Image: `laion/CLIP-ViT-H-14-laion2B` (best, but slower)

Choose based on your priorities: **Speed** (current) vs **Accuracy** (upgrade) vs **Best** (maximum accuracy).
