# Multimodal Image ML Training with Lance on Databricks

<p>
  <img src="https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white" alt="Databricks"/>
  <img src="https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white" alt="Apache Spark"/>
  <img src="https://img.shields.io/badge/Ray-028CF0?style=flat-square&logo=ray&logoColor=white" alt="Ray"/>
  <img src="https://img.shields.io/badge/Lance-6B4FBB?style=flat-square&logoColor=white" alt="Lance"/>
</p>

A blueprint demonstrating how the [Lance](https://lancedb.github.io/lance/) columnar format optimizes image-based ML training on Databricks — specifically addressing where Delta/Parquet breaks down for multimodal workloads.

The three foundational image ML problem types — **classification**, **object detection**, and **segmentation** — all share the same core training bottleneck on Databricks: random-access reads of large binary payloads (raw image bytes) from a Parquet-backed store are fundamentally mismatched with how ML training DataLoaders work. This blueprint uses BDD100K dashcam footage and a CNN object detection task to demonstrate Lance as the solution, with Ray Data and Ray Train handling distributed training across GPUs.

---

## The standard Databricks path for ML

Delta Lake + Parquet is the right default for most ML workloads on Databricks. It provides Unity Catalog governance, SQL access, Photon-accelerated queries, time-travel, and native integration with MLflow and Feature Store. For tabular features — structured data like user events, transactions, and numerical features — Delta is the correct choice with no caveats.

---

## Where it breaks for image ML training

Image ML training has three requirements that Parquet is architecturally unable to meet efficiently.

### 1. Inline binary is required for CNN training

CNN training needs raw pixels on every batch. The natural alternative — storing a path string pointing to an image file in object storage — reintroduces a per-image I/O hop that Delta cannot eliminate:

```
Path reference approach (Delta):
  read Lance/Delta row → get path string → GET /Volumes/.../frame_0001.jpg
                                            GET /Volumes/.../frame_0002.jpg
                                            ...64 concurrent requests per batch
```

At batch size 64 across 100 Ray workers, this produces 6,400 concurrent object storage GET requests per training step. Object storage API rate limits bite (~5,500 GET/s per S3 prefix), per-request latency (~10–100ms) compounds, and the GPU idles waiting on I/O.

### 2. Parquet row groups collapse with inline binary

Delta stores data in Parquet row groups, defaulting to ~128MB or ~1M rows per group. With structured tabular rows (~1KB each), random access costs ~1KB of wasted I/O per sample fetch. With inline image bytes (~100KB/frame), row groups collapse:

| Row type | Row size | Rows per row group | Wasted I/O per random fetch |
|----------|----------|--------------------|-----------------------------|
| Tabular features | ~1 KB | ~128,000 | ~1 KB |
| Inline JPEG frames | ~100 KB | ~1,280 | ~128 MB |

A random batch of 64 frames requires reading 64 × 128MB = **8GB of data** from Parquet to retrieve 64 × 100KB = 6.4MB of actual content. This gets worse as image resolution increases.

### 3. No blob isolation — metadata scans touch image bytes

In Parquet, all columns for a row group are co-located. A query filtering on `video_id` or `timestamp_ms` must scan through the image bytes in that row group to find the rows it needs. There is no way to isolate large binary payloads from structured metadata in Parquet.

---

## Lance: built for this

Lance addresses each failure mode directly.

**Blob layout.** Binary payloads are stored in a dedicated blob file, physically separate from structured metadata columns. A metadata scan on `video_id` or `frame_number` never reads image bytes. When a batch does need image bytes, reads go directly to the blob's byte offset — no row group to scan through.

**Fragment-level O(1) random access.** Lance datasets are composed of independent fragments. Any row in any fragment is addressable by a direct byte-offset seek. The cost of fetching row 1 is identical to fetching row 5,000,000. Batch reads are coalesced into sequential byte-range reads within fragments — no per-image request overhead.

**Arrow-native storage.** Lance stores data in Arrow IPC format internally. Ray Data reads directly into Arrow blocks with no Parquet deserialization step — the CPU cycles that Parquet burns on deserialization go to image augmentation instead.

**Fragment-parallel reads for Ray.** Each Ray worker actor reads its assigned Lance fragment(s) independently. No cross-worker coordination, no shared shuffle state. The data loading pipeline scales linearly with the number of Ray workers.

|  | Lance | Delta |
|--|-------|-------|
| Random-access batch sampling | O(1) per row, fragment-level addressing | Row-group scan — 128MB per 1,280-row group for images |
| Inline binary storage | Blob layout — isolated, byte-offset addressed | All columns co-located in row group |
| Metadata scan with binary columns | Skips blob data entirely | Must traverse binary bytes to find metadata rows |
| Ray Data integration | `ray.data.read_lance` — Arrow-native, fragment-parallel | `ray.data.read_parquet` — Parquet deserialization overhead |
| UC 3-level namespace | Path-based only | Full UC integration |
| Dataset versioning for ML | First-class — every write is a new version | SQL time-travel, not ML-oriented |

### Ray Data + Lance

This blueprint uses `ray.data.read_lance` rather than `ray.data.read_databricks_tables` or `ray.data.read_parquet` for three reasons:

**1. Random access without shuffle overhead.** Parquet forces Ray Data to read entire row groups to retrieve a handful of samples. Lance uses fragment-level addressing for O(1) point lookups — [benchmarked by LanceDB](https://lancedb.github.io/lance/) at 100–1000x faster random access than Parquet depending on access pattern.

**2. Near-zero deserialization overhead.** Lance stores data in Arrow IPC format natively. Ray Data reads directly into Arrow blocks with no Parquet → Arrow conversion step. That CPU budget goes to preprocessing and augmentation instead.

**3. Fragment-parallel reads map to Ray's actor model.** Each Ray worker reads its assigned Lance fragments independently with no cross-worker coordination. Binary payloads (image bytes) are stored in a separate blob layout — a worker loading a batch of embeddings or metadata never touches the image bytes.

---

## Dataset: BDD100K

[BDD100K](https://bdd-data.berkeley.edu/) (Berkeley DeepDrive) is a large-scale driving dataset collected by UC Berkeley.

- **100,000 dashcam videos**, ~40 seconds each, 720p at 30fps
- **Object detection labels** for 10 categories on a 10K-video subset: car, truck, bus, person, rider, bicycle, motorcycle, traffic light, traffic sign, train — bounding boxes per frame
- **Video-level labels** for all 100K videos: weather (clear/rainy/foggy/snowy/overcast), scene (city/highway/residential), time of day (day/dawn-dusk/night)
- **License:** BSD — open for research use

**Scale at 1fps sampling:** ~40 frames/video × 100K videos = ~4M frames. At 3fps: ~12M frames, past the threshold where Lance's random-access advantage over Parquet is measurable in end-to-end training throughput.

---

## Architecture

```
BDD100K videos in Databricks Volumes
            │
            ▼
  [01] Create Lance dataset — Ray (GPU)
       decord → JPEG bytes inline + BDD100K bbox annotations
       lance.write_dataset(frames/, mode="append")
            │
            ▼
  [02] CNN training — Ray Data + Ray Train
       ray.data.read_lance → preprocessing pipeline → DDP training
       ResNet-50 + FPN, 10-category object detection
       Lance vs Delta throughput benchmark + MLflow logging
```

---

## Notebooks

### Main Track

| Notebook | Description |
|----------|-------------|
| `01_create_lance_dataset.ipynb` | Ray GPU frame extraction from BDD100K → Lance `frames` table (inline JPEG + bounding box labels) |
| `02_cnn_training.ipynb` | Distributed CNN object detection training via Ray Data + Ray Train; Lance vs Delta throughput benchmark |

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
| image | binary | Inline JPEG bytes — blob layout, isolated from metadata |
| width, height | int32 | |
| bbox_labels | list\<struct\> | `{category, x1, y1, x2, y2}` per object; joined from BDD100K annotations at write time |
| extracted_at | timestamp[us] | |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `lance` | Core read/write, blob layout |
| `ray[data]` | Distributed data loading |
| `ray[train]` | Distributed DDP training |
| `decord` | GPU-accelerated video frame decoding |
| `torch` + `torchvision` | Model definition and training loop |
| `pyarrow` | Schema definition and batch construction |

> **Note:** `lance-spark` (Maven JAR) is available for reading Lance datasets via Spark for cross-table analytics, but is not required for the training pipeline.

---

## Caveat: Unity Catalog governance tradeoffs

Lance datasets live in UC Volumes, not UC-registered tables. For most ML practitioners this is a non-issue — training pipelines are Python, not SQL, and the DataLoader doesn't care about catalog registration. Storage governance (file-level `READ FILES` / `WRITE FILES` grants, audit logging, Catalog Explorer visibility) is fully retained at the Volume level.

Two UC features that Delta provides natively are unavailable for Lance:

- **Data lineage.** UC captures column-level lineage for registered Delta tables. Path-based reads of Lance files produce no lineage entries — reads and writes are invisible to the UC lineage graph.
- **Time travel.** Delta's `VERSION AS OF` / `TIMESTAMP AS OF` SQL syntax does not apply to Lance. Lance has its own immutable fragment versioning, but it is not SQL-queryable and carries no UC-managed retention policy.

Both can be recovered with a thin Delta manifest table:

```sql
CREATE TABLE main.ml.lance_manifest (
  dataset_name  STRING,
  volume_path   STRING,
  lance_version BIGINT,
  row_count     BIGINT,
  schema_json   STRING,
  written_at    TIMESTAMP,
  written_by    STRING,
  run_id        STRING
);
```

Each write to the Lance dataset appends one row. The manifest is a registered Delta table in UC — it appears in Catalog Explorer, participates in lineage, and is queryable via SQL. To reproduce a training run or roll back to a prior dataset state, look up the `lance_version` in the manifest and call `lance.dataset(path).checkout_version(n)`.

---

## Parking Lot

- **Format benchmark** — run end-to-end training throughput (samples/sec, GPU utilization, data loading wall time) on BDD100K across Lance, Mosaic Streaming MDS, and Delta/Parquet using the same ResNet-50 + Ray Train setup. MDS is the Databricks-native alternative for distributed training and is meaningfully better than Parquet for binary workloads, but lacks Lance's blob isolation — the benchmark would quantify that gap concretely.

  | | Lance | Mosaic MDS | Delta/Parquet |
  |---|---|---|---|
  | Blob isolation | Yes — dedicated blob file | No — co-located per record | No — co-located per row group |
  | Random access unit | Single row, O(1) | Within-shard index | Full row group (~128MB for images) |
  | Wasted I/O per random fetch | ~0 (byte-offset seek) | Up to shard size (~67MB) | Up to row group (~128MB) |
  | Cloud-native | Yes | Yes | Yes |
  | Databricks integration | Volumes path | Volumes path | Native UC table |

- Standardize `lance` library version across all notebooks via a shared `requirements.txt` or cluster init script.
- **Audio-visual classification** — store video frames (image binary) and audio spectrograms (binary) as two heterogeneous blob columns in a single Lance table. This is the strongest demonstration of Lance's mixed-blob layout advantage over Parquet, which has no equivalent for co-locating heterogeneous binary payloads. Model: audio-visual classifier (e.g., AVSlowFast) trained on VGGSound or AudioSet.
- **Semantic segmentation** — store image binary and pixel-level mask binary as two blob columns per row. Row-group collapse in Parquet is doubly pronounced (both blobs contribute to row group size shrinkage). Model: U-Net or SegFormer on BDD100K drivable area labels, which are already available in the dataset.
