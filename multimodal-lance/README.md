# Multimodal Feature Engineering + ML Training with Lance on Databricks

<p>
  <img src="https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white" alt="Databricks"/>
  <img src="https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white" alt="Apache Spark"/>
  <img src="https://img.shields.io/badge/Ray-028CF0?style=flat-square&logo=ray&logoColor=white" alt="Ray"/>
  <img src="https://img.shields.io/badge/Lance-6B4FBB?style=flat-square&logoColor=white" alt="Lance"/>
</p>

A blueprint for building scalable multimodal ML training pipelines on Databricks using the [Lance](https://lancedb.github.io/lance/) columnar format. Covers GPU-accelerated frame extraction via Ray, incremental feature engineering, and optimized DataLoader-ready dataset construction.

---

## Why Lance instead of Delta?

Lance is a columnar format purpose-built for ML workloads. On Databricks, Delta is the default — and it works for analytics. But for multimodal training datasets it has a fundamental limitation.

**The Parquet row-group problem.** Delta stores data in Parquet row groups (~128MB or ~1M rows each). Fetching a single random row requires reading the entire row group it belongs to. For a training DataLoader that samples random batches each step, this means reading orders of magnitude more data than needed. At 10M+ rows — roughly when your dataset outgrows cluster memory and can no longer be cached — this becomes the primary training bottleneck.

**Why 10M rows is the rough threshold:** Below that, you can shuffle the dataset once per epoch in Spark and cache it in memory, paying the scan cost upfront. Above it, the dataset (~1TB+ for image data) can't fit in memory, so every training step hits cold I/O — and Parquet's row-group overhead compounds every batch.

Lance solves this with O(1) random access to any row. A batch of 64 frames from a 100M-row table costs the same as a batch from a 1M-row table.

|  | Lance | Delta |
|--|-------|-------|
| Random-access batch sampling | Native, O(1) per row | Row-group scan — expensive at scale |
| Binary/image storage | Efficient, designed for it | Works, Parquet encoding is wasteful |
| ML DataLoader integration | Native (`lance.torch.data`) | Requires Petastorm or custom bridge |
| Dataset versioning for ML iteration | First-class (append, delete, evolve schema) | Time-travel is SQL-oriented, not ML-oriented |
| UC 3-level namespace | Path-based only | Full UC integration |
| Cluster setup complexity | Needs extra JARs/libs | Zero extra setup |

### Unity Catalog and Lance

Lance tables cannot use `catalog.schema.table` SQL syntax — Unity Catalog natively manages only Delta tables. Lance datasets are accessed by path:

```python
# ✅ Path-based access
import lance
ds = lance.dataset("/Volumes/main/ml/media/frames/")

# ❌ Not supported
spark.read.table("main.ml.frames")
```

Organize Lance datasets inside UC Volumes to retain storage governance and access control. Optionally maintain a thin Delta manifest table in UC that records each Lance dataset's path, row count, and version.

---

## Architecture

```
Raw videos in Databricks Volumes
            │
            ▼
  [01] Frame Extraction — Ray (GPU)
       decord/ffmpeg → JPEG bytes inline
       lance.write_dataset(frames/, mode="append")
            │
            ▼
  [02] Feature Engineering — Ray (GPU)
       CLIP vision encoder → embedding per frame
       Lance version update (v1 → v2 with embeddings)
       └── Push embeddings → Mosaic Vector Search (retrieval, separate concern)
            │
            ▼
  [03] Inference + Training (three approaches)
       Stage 1: VLM structured output (LLaVA via Ray) — best quality, high cost
       Stage 2: CLIP zero-shot — free, no training, ~75% accuracy
       Stage 3: CLIP embeddings → trained MLP — production scale, domain-adapted
```

---

## Notebooks

### Main Track

| Notebook | Description |
|----------|-------------|
| `01_extract_frames_ray.ipynb` | GPU frame extraction → Lance `frames` table (inline JPEG + metadata) |
| `02_feature_engineering.ipynb` | CLIP embeddings → new Lance dataset version |
| `03_train_model.ipynb` | Three inference strategies — VLM structured output, CLIP zero-shot, trained MLP classifier — with cost/throughput trade-offs and Lance vs Delta benchmark |

### Optional

Located in `optional/`. Use when you need queryable metadata tables in Unity Catalog alongside the Lance training dataset.

| Notebook | Description |
|----------|-------------|
| `optional/ingest_videos.ipynb` | Video file metadata → Delta table (UC namespace, no binary data) |
| `optional/transcribe_audio_ray.ipynb` | Whisper transcription → Delta table (UC namespace, text only) |

---

## Dataset Schema

### `frames` Lance table — `/Volumes/{catalog}/{schema}/{volume}/frames/`

| Column | Type | Notes |
|--------|------|-------|
| frame_id | string | UUID |
| video_id | string | |
| frame_number | int32 | 0-indexed |
| timestamp_ms | int64 | |
| image | binary | JPEG bytes inline |
| width, height | int32 | |
| embedding | fixed_size_list\<float32\>[768] | Added in notebook 02 |
| label | string | Optional annotation |
| extracted_at | timestamp[us] | |

### Dataset versioning

```
v1  frames: image + metadata only
v2  + CLIP embedding column      (after notebook 02)
v3  + human-reviewed labels      (annotation updates)
v4  + additional feature columns (further FE iterations)
```

Pin a version for training reproducibility: `lance.dataset(path, version=2)`

### Multi-embedding experimentation

A common pattern during model development is testing several embedding models to find the best one for a downstream task. Lance handles this well due to its fixed-size list encoding and column-projected random access.

**Storage estimate at 10M frames:**

| Model | Dims | Storage |
|-------|------|---------|
| CLIP ViT-B/32 | 512 | ~20 GB |
| CLIP ViT-L/14 | 768 | ~30 GB |
| OpenCLIP ViT-H/14 | 1024 | ~40 GB |
| DINOv2 ViT-L/14 | 1024 | ~40 GB |
| SigLIP | 1152 | ~46 GB |

4–5 embedding columns on 10M frames adds roughly 150–180 GB of float data on top of ~1 TB of JPEG bytes. Neither Lance nor Delta has a hard table size limit you'd hit, but encoding efficiency matters at this scale.

**Why Lance handles this better than Delta:**

Lance stores `fixed_size_list<float32>[N]` as contiguous raw float blocks with no per-element overhead. Parquet (Delta's backing format) uses repeated field encoding for arrays — designed for variable-length data — so fixed-size embeddings carry unnecessary overhead. In practice, Lance is ~10–30% more compact for high-dimensional float columns.

More importantly, Lance's column projection means reading one embedding column during training has zero I/O cost from the others:

```python
# Only loads image + embedding_clip — other embedding columns are not read
ds = lance.dataset(path, version=3)
loader = lance.torch.data.LanceDataset(
    ds,
    columns=["image", "embedding_clip_vitl14"],
    batch_size=64,
)
```

In Parquet, each row group contains all columns. As you add more embedding columns, each row group grows larger — making random-access fetches more expensive even with column pruning, because more bytes must be skipped per row group seek.

**Recommended workflow for embedding selection:**

1. Run FE on a representative sample (~1–5M frames) rather than the full dataset to select the best embedding model.
2. Add each candidate as a separate named column (`embedding_clip`, `embedding_dinov2`, etc.) via `mode='merge'` — each merge creates a new Lance version.
3. Train lightweight probes or run zero-shot evals on each column independently.
4. Once the winner is selected, run that model's FE pass on the full dataset.

This avoids paying the GPU compute cost of generating all embeddings at full scale before knowing which one works.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `lance` | Core read/write |
| `ray[data]` | Distributed GPU processing |
| `decord` | Video frame decoding |
| `ffmpeg-python` | Audio extraction |
| `transformers` | CLIP (embeddings) + Whisper (transcription) |
| `torch` | Model training |
| `pyarrow` | Schema definition and batch construction |

> **Note:** `lance-spark` (Maven JAR) is available for reading Lance datasets via Spark for cross-table joins or aggregations, but is not required for the main training pipeline.

---

## Parking Lot

- Standardize `lance` library version across all notebooks via a shared `requirements.txt` or cluster init script.
