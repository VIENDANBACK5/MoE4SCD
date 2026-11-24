# TreeFlowNet Crop Sampling Policy V3

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Governing Standard:** Full-Crown Representation, Epoch Seeding Determinism, and Boundary-Truncation Control.

---

## 1. Multi-Epoch Worker RNG Fix

### The Defect in Legacy Dataset
In legacy `CanopyCropDataset` (`clean_benchmark/dataset.py:L46`):
```python
rng = np.random.default_rng(i if self.fixed else (torch.initial_seed() + i) % (2 ** 32))
```
In PyTorch multi-process data loading, worker process seeds remain identical across training epochs unless the DataLoader worker initialization or sampler injects the current epoch index. Consequently, workers extracted the exact same sequence of crops every epoch, collapsing multi-epoch data augmentation into single-epoch repetition.

### The Standardized Fix
Every crop extraction MUST deterministically derive its RNG seed from four factors:
$$\text{seed} = \text{uint32}(\text{sha256}(\text{f"{master\_seed}:{epoch}:{split}:{tile\_index}:{crop\_idx}"})[:4])$$
```python
def get_crop_rng(master_seed: int, epoch: int, split: str, tile_idx: int, crop_idx: int) -> np.random.Generator:
    key = f"{master_seed}:{epoch}:{split}:{tile_idx}:{crop_idx}".encode("utf-8")
    seed_int = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return np.random.default_rng(seed_int)
```
- For `TRAIN`: advances epoch index $E \in \{0, 1, \dots\}$ to guarantee unique, non-repeating spatial crops every epoch.
- For `DEVELOPMENT` / `VAL`: uses fixed $E = 0$ to guarantee identical validation crops for comparable loss tracking across epochs.

---

## 2. Full-Crown Sampling Categories

To overcome the limitation of historical pilots (where Q sanity evaluated 100% truncated fragments), the sampling pipeline enforces a balanced distribution of eight crown context categories:

| Category Code | Description | Minimum Requirement | Scientific Role |
|---|---|---|---|
| `CAT_BG_ONLY` | Pure background / gap crop ($N_{GT} = 0$, valid $\ge 0.5$) | At least 5% of training crops | Teaches no-object suppression and background flow decay. |
| `CAT_SINGLE_CROWN` | Exactly one complete crown ($N_{complete} = 1, N_{partial} = 0$) | At least 10% of training crops | Isolates single-crown representation learning without crowding. |
| `CAT_FEW_CROWNS` | 2 to 4 complete crowns | At least 25% of training crops | Tests local pairwise individuation and neighbor separation. |
| `CAT_CROWDED` | $\ge 5$ complete crowns | At least 20% of training crops | Tests dense packing, query capacity, and basin watershed limits. |
| `CAT_SMALL_CROWNS` | $\ge 50\%$ of crowns $< 1.0\text{ m}^2$ | At least 10% of training crops | Prevents small-crown under-representation and vanishing centroids. |
| `CAT_TOUCHING` | Crowns with boundary distance $\le 1\text{ px}$ | At least 15% of training crops | Tests seam detection, flow divergence, and merge avoidance. |
| `CAT_MULTI_LOBED` | Crowns with multiple local maxima | At least 5% of training crops | Tests robustness against over-segmentation into artificial sub-crowns. |
| `CAT_TRUNCATED` | Contains edge-intersecting crowns | Permitted up to 30% of crops | Tests partial-object handling with explicit boundary flags. |

---

## 3. Truncation and Margin Accounting

### Physical Field of View vs. Crown Size
- Crown diameters in QuebecTrees range from $0.5\text{ m}$ to $12.0\text{ m}$ (17 to 400 pixels at $0.03\text{ m}$ GSD).
- A 1024×1024 crop spans $30.72\text{ m} \times 30.72\text{ m}$, easily accommodating 10 to 40 complete crowns.
- A 256×256 crop spans only $7.68\text{ m} \times 7.68\text{ m}$, making edge truncation almost unavoidable for mature trees ($>8\text{ m}$).
- **Policy:** The standard training crop size for complete-crown learning is $1024 \times 1024$ pixels. Where $256 \times 256$ is required for constrained compute, a target margin of $m \ge 64\text{ px}$ must be used to preserve full-crown centroid context.

### Crop Metadata Record
Every sampled training crop must output a structured metadata record:
```json
{
  "tile_id": "train:sblz1_0042",
  "crop_coordinates": {"y0": 256, "x0": 512, "height": 1024, "width": 1024},
  "valid_fraction": 0.985,
  "n_complete_instances": 14,
  "n_partial_instances": 4,
  "n_ignored_tiny_instances": 1,
  "query_capacity_sufficient": true,
  "category": "CAT_CROWDED"
}
```

### Capacity Overflow Invariant
If ground-truth instances $N_{GT}^{crop} > Q_{max}$ (where $Q_{max}$ is model query capacity, e.g. 16 or 100), the sampler MUST either:
1. Log an explicit capacity overflow warning and track unassigned targets as forced False Negatives; OR
2. Reject the crop during balanced mini-batch assembly.
Under no circumstances may excess targets be silently dropped from the ground-truth matching denominator.
