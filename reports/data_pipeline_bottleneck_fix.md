# Data Pipeline Bottleneck: Root Cause & Fix

**Date:** 2026-05-22
**Observed during:** 760M pretraining run on 8 GPUs

---

## Symptoms

- All 8 GPUs showed 0% utilization in `nvidia-smi` despite processes running
- Training throughput: ~13,710 tokens/s on 8 large GPUs (should be 8× higher)
- Projected training time: ~3.2 days for one epoch

---

## Root Cause 1: No Data Sharding (Wrong Throughput)

**File:** `unbox_platform/data/dataset.py` — `build_dataloader`

```python
# Before (broken)
dataset = StreamingPretrainDataset(tokenizer, config, split=split)
# rank and world_size parameters accepted but never passed to dataset
```

`StreamingPretrainDataset.__iter__` had no sharding logic. All 8 DDP ranks
independently iterated the **full dataset** and computed **identical gradients**.
The all-reduce then averaged 8 copies of the same gradient — equivalent to
training on a single GPU. Effective throughput was 1× instead of 8×.

**Fix:** Round-robin document sharding across `world_size × num_workers` partitions:

```python
def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    num_workers = worker_info.num_workers if worker_info is not None else 1

    shard_id = self.rank * num_workers + worker_id
    total_shards = self.world_size * num_workers
    local_skip = self.skip_chunks // total_shards

    for doc_idx, text in enumerate(self._iter_texts()):
        if doc_idx % total_shards != shard_id:
            continue
        # tokenize and yield ...
```

`rank` and `world_size` are now passed at dataset construction time.
Within-rank worker sharding uses `get_worker_info()` so DataLoader workers
further subdivide each rank's shard without duplication.

---

## Root Cause 2: num_workers=0 (GPU Starvation)

**File:** `unbox_platform/data/dataset.py` — `build_dataloader`

```python
# Before (broken)
num_workers=0,  # comment: "multiple workers each iterate the full dataset independently"
```

With `num_workers=0` the DataLoader tokenizes synchronously in the main thread.
The GPU sat idle after every forward/backward pass while the CPU tokenized the
next batch. This caused severe GPU utilization bubbles.

**Fix:** Use `config.num_workers` (configured to 4 in `configs/pretrain/760m.yaml`):

```python
num_workers=config.num_workers,
```

With 4 workers per rank, tokenization runs ahead of the training loop and
keeps the GPU fed continuously.

---

## Root Cause 3: Double File Scan Per Worker (Slow Skip / Startup)

**File:** `unbox_platform/data/dataset.py` — `StreamingPretrainDataset._iter_jsonl`

```python
# Before (broken): two full 21 GB scans per worker instance
with open(path) as f:
    total_docs = sum(1 for l in f if l.strip())  # scan 1: count lines
n_eval = max(1, int(eval_fraction * total_docs))

with open(path) as f:
    for i, line in enumerate(f):  # scan 2: actual iteration
        ...
```

With `num_workers=4` per rank and 8 ranks, every startup would have triggered
`8 × 4 × 2 = 64` full scans of the 21 GB JSONL (≈1.3 TB of sequential reads).
Even with `num_workers=0`, the two-pass design made resume startup slow:
the skip phase took **~35 minutes** because it had to tokenize through ~20% of
the dataset before yielding any training chunks.

**Fix:** Precompute `n_eval` once in `__init__` (main process, before workers fork):

```python
def __init__(self, ...):
    ...
    self._n_eval = self._compute_n_eval()  # single scan, result pickled to workers

def _iter_jsonl(self, path):
    n_eval = self._n_eval  # no extra scan
    with open(path) as f:
        for i, line in enumerate(f):
            ...
```

Workers inherit `self._n_eval` via pickle — zero additional file scans.

---

## Side Effect Fix: Eval Loader Sharding

**File:** `unbox_platform/train/pretrain.py`

The eval loader was constructed with `rank=rank, world_size=world_size`.
Since `evaluate()` only runs on rank 0, this caused rank 0 to see only
`1/world_size` of the eval set (≈300 docs instead of 9,672). Fixed by
always using `rank=0, world_size=1` for eval:

```python
eval_loader = build_dataloader(
    tokenizer, data_cfg, split="eval",
    batch_size=train_cfg.batch_size,
    rank=0, world_size=1,  # eval runs on rank 0 only
)
```

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| Effective data parallelism | 1× (all ranks same data) | 8× (disjoint shards) |
| GPU starvation | Yes (sync tokenization) | No (4 async workers/rank) |
| Resume skip time | ~35 min (tokenize 20% of data) | ~5 min (tokenize 20% ÷ 32 shards) |
| Projected training time | ~3.2 days | ~9 hours |

---

## Files Changed

- `unbox_platform/data/dataset.py` — sharding, `num_workers`, `_n_eval` precompute
- `unbox_platform/train/pretrain.py` — eval loader uses `rank=0, world_size=1`
