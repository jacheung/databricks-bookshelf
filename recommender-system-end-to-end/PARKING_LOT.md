# Parking Lot

Items deferred for future work. Reference these before starting new sessions on this project.

---

## ~~1 — Enrich Training Data with Item Features~~ ✓ Done

> Completed: `item_catalog` and `user_features` tables added to `00_data_preparation.ipynb` Section 3. Notebook 03 Section 1 joins these for two-tower feature enrichment.

---

## ~~1 — Wire Feature Tables into Notebook 03 Data Prep (`03_two_tower.ipynb`)~~ ✓ Done

> Completed: `item_catalog` and `user_features` joined in Section 1. `UserTower` and `ItemTower` each accept `[id_embedding ‖ feature_vector]` concat as input. Feature tensors constructed from vocab tables for eval and precompute.

---

## 3 — Temporal Train/Test Split

**What:** Replace the current random 80/20 split with a temporal split — train on the earliest 80% of each user's orders, test on the most recent 20%. More realistic evaluation that simulates predicting what a user will buy next.

**Why deferred:** Changes the data distribution for notebooks 01 and 02 — their Hit@K baselines will shift. Worth doing as a deliberate re-baselining exercise, not bundled with other changes.

**Entry point:** `00_data_preparation.ipynb` Cell 5 — replace `randomSplit([0.8, 0.2], seed=42)` with a window-function-based temporal split using `order_timestamp` (now available on the cleaned dataset).

---

## 4 — Scale Synthetic Data + Widen Two-Tower Model

**What:** Two changes to stress the A10 GPUs and make DDP meaningful:
1. Increase synthetic data in `00_data_preparation.ipynb`: `num_users = 50_000`, `num_orders = 50_000` → ~150K interaction pairs, ~150 steps/epoch
2. Widen the model in `03_two_tower.ipynb` Cell 3: `embed_dim = 256`, `hidden_dim = 512`, `batch_size = 8192`

**Why deferred:** Current dataset (2K users, 15K orders → ~45K pairs, ~44 steps/epoch) is sufficient for an end-to-end run. GPU saturation is not a priority until the pipeline is validated.

**Why it matters:** At `embed_dim=64`, GEMM inner dimensions are too thin for A10 tensor cores (~15-25% utilization regardless of data volume). Widening to 256+ and raising batch_size to 8192 is the right lever alongside more data.

**Entry points:** `00_data_preparation.ipynb` Cell 2 (`num_users`, `num_orders`); `03_two_tower.ipynb` Cell 3 (`embed_dim`, `hidden_dim`, `batch_size`).

---

## 5 — Add Compute Sizing Narrative to Notebook 03

**What:** Bake the model size and cluster headroom analysis into a markdown cell in `03_two_tower.ipynb` — likely as a callout at the top of Section 2 (Ray Cluster Setup), before the `setup_ray_cluster` call.

**The artifact to use:**

| Component | Params | Memory (float32) |
|---|---|---|
| User embedding (2K users × 64) | 128,000 | 512 KB |
| Item embedding (18 items × 64) | 1,152 | 5 KB |
| User MLP (71→128→64) | 17,472 | 70 KB |
| Item MLP (68→128→64) | 17,088 | 68 KB |
| **Total** | **~164K** | **~656 KB** |

Model + gradients + Adam optimizer states ≈ **3 MB per worker**. Dataset shard (~22K rows) ≈ **1.5 MB per worker**. Total per Ray worker: ~15-20 MB.

Key point to make in the narrative: this model is memory-trivial on any modern cluster — the sizing constraint is GPU compute throughput (GEMM efficiency), not RAM. For CPU-only clusters, drop `num_gpus_worker_node` from `setup_ray_cluster` and set `use_gpu=False` in `ScalingConfig`.

**Entry point:** `03_two_tower.ipynb` Cell 7 (Section 2 markdown) — expand the cluster config callout with this table and narrative.

---

## 2 — Vector Search Serving for Two-Tower Model (`03_two_tower.ipynb`)

**What:** Replace the precomputed top-K Lakebase lookup with real-time ANN (approximate nearest neighbor) search over item embeddings at serve time.

**Approach:**
- Store item embeddings in a Databricks Vector Search index (Delta Sync index on `two_tower_item_embeddings` table)
- At serve time: look up user embedding from Lakebase (or recompute from the user tower), query the Vector Search index for top-K nearest items
- Enables personalization for new/updated user signals without re-running the full precompute pipeline

**Why deferred:** The precomputed approach is simpler and matches the serving pattern already established in notebooks 01 and 02. Vector Search adds operational complexity (index sync latency, embedding versioning) that is only warranted at scale or when real-time personalization is needed.

**Entry point:** `03_two_tower.ipynb` Section 6 — replace the Lakebase precomputed recs table with a Vector Search index creation + a new pyfunc that calls `VectorSearchClient.get_index().similarity_search()` at predict time.
