# DTE-aerial Scope Lock and Gate-0 Experiment Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-25
- Verification Status: PARTIALLY VERIFIED
- Version Label: dte_research_loop_v2

## Frozen research scope

**Problem:** RGB aerial imagery to three-class semantic segmentation on DTE-aerial.

- Input: RGB aerial image.
- Output labels: `0=background`, `1=tree cover`, `2=mortality`.
- Training/validation source: DTE-aerial-train.
- Test source: DTE-aerial-bench.
- Provisional direction, not yet a method claim: **Robust Tree-Mortality Segmentation under Resolution and Geographic Shift**.

Out of scope until the failure gate is passed: LiDAR, SAM/SAM2, object tokens, region representations, biomass, satellite imagery, additional datasets, new backbones, and new decoders.

## Official protocol to reproduce

Source of truth: Sharma et al. (2026), official DTE-aerial repository commit `7888b3b8ac4e66013c82e19753de9925b3643dac`.

- Split DTE-aerial-train by complete orthophoto: 90% train and 10% validation.
- Official counts: 1,959 training orthophotos (about 346K patches) and 217 validation orthophotos (about 39K patches).
- DTE-aerial-bench is a separate test set: 25 orthophotos and 525 non-overlapping 1024x1024 patches.
- Each benchmark site contributes the same scene at three resolutions: 16 tiles at 5 cm, 4 tiles at 10 cm, and 1 tile at 20 cm.
- Training crop: 640x640 pixels.
- Seeds: `0`, `100`, `200`.
- Batch size: 32 effective samples.
- Optimizer: AdamW, learning rate `1e-4`, minimum learning rate `1e-5`, weight decay `0.01`, betas `(0.9, 0.99)`.
- Schedule: cosine decay over 100K iterations.
- Loss: multiclass Tversky; background/tree cover alpha=beta=0.5; mortality alpha=0.3, beta=0.7.
- Early stopping: mean per-site F1 or mortality per-site F1 must improve by at least `1e-3` within two evaluation epochs.
- Ignore label: `255`.
- Model selection and all tuning use only the official validation split. DTE-aerial-bench is test-only.

## Stage G0: minimal parity gate

G0 closes after three tasks only:

1. Acquire DTE-aerial-bench and freeze its checksums.
2. QC the benchmark image, mask, and metadata schema.
3. Evaluate the released MiT-B3 checkpoint with the official site-macro evaluator.

Published MiT-B3 parity targets are:

| Metric | Target |
|---|---:|
| Mortality F1 | 0.59 |
| Mortality IoU | 0.45 |
| Mortality precision | 0.72 |
| Mortality recall | 0.54 |
| Tree-cover F1 | 0.89 |

Published mortality F1 by resolution:

| 5 cm | 10 cm | 20 cm |
|---:|---:|---:|
| 0.60 | 0.55 | 0.45 |

### G0 pass rule

G0 passes only when all of the following hold:

1. The benchmark contains the expected 25 orthophotos and 525 tiles, with 21 paired multi-resolution tiles per site.
2. Images are RGB, masks use only `0/1/2/255`, and every row has valid site, biome, resolution, image path, and mask path fields.
3. Same-scene scale pairing is verified for 5/10/20 cm; no duplicated or missing metadata rows remain unexplained.
4. The released checkpoint loads without missing or unexpected learned weights.
5. Aggregation is F1 per site followed by a macro average over sites, matching the paper.
6. Overall and grouped metrics are numerically close to the authors' values after resolving rounding and evaluator details; the principal mortality trend is `5 cm > 10 cm > 20 cm`.
7. Dataset, checkpoint, code commit, environment, and evaluation output are hashed and recorded.

The initial numerical tolerance is `0.01` absolute for F1. A larger discrepancy is investigated as data, preprocessing, label, checkpoint, or aggregation mismatch. Once these checks pass, reproduction stops: U-Net, Mask2Former, and twelve baseline training runs are not prerequisites for research.

## Stage G1: minimal diagnosis

Research begins immediately after G0. The goal is to identify the mechanism behind the resolution failure, not to produce a complete benchmark table.

The MiT-B3 checkpoint predictions must be frozen at tile level so diagnosis can be repeated without rerunning inference.

### Required strata

- Resolution: 5 cm, 10 cm, 20 cm.
- Site/geography: each of the 25 benchmark orthophotos.
- Biome: temperate, (sub)tropical, boreal/montane, drylands/Mediterranean.
- Semantic class: tree cover and mortality reported separately.
- Mortality-region size: connected-component area converted to square metres using GSD; report fixed physical-area bins plus the raw area.
- Boundary quality: report per class and per resolution.

### Required metrics

- Per class: IoU, Dice/F1, precision, recall.
- Boundary F1 at a fixed physical tolerance (primary) and one-pixel tolerance (diagnostic).
- Connected-component diagnostics for mortality:
  - `miss`: a ground-truth component has no overlap-graph edge;
  - `split`: a ground-truth component connects to two or more predicted components;
  - `merge`: a predicted component connects to two or more ground-truth components.
  - An overlap-graph edge requires intersection divided by the smaller component area to be at least 0.10. The threshold is fixed before examining model comparisons.

### Required comparisons

1. Paired within-site degradation from 5 to 10 to 20 cm; do not compare unrelated scenes across GSD.
2. Mortality degradation versus tree-cover degradation.
3. Recall loss versus precision loss.
4. Pixel overlap degradation versus boundary degradation.
5. Small-region degradation versus medium and large regions.
6. Per-site and per-biome distributions, not only pooled means.
7. Bootstrap confidence intervals over sites for the principal paired 5-to-20 cm deltas.

### Required visual diagnosis

1. Same-scene rows: `RGB | GT | MiT-B3` at 5, 10, and 20 cm.
2. Error maps separating mortality TP, FP, and FN.
3. Zooms of small mortality regions, boundary contraction, mortality-to-tree-cover confusion, and canopy merges.
4. Mortality recall plotted against ground-truth physical component area.

### One architecture sanity check

Train or obtain predictions from DeepLabV3+ using the same data protocol. Its purpose is only to test whether the 5-to-20 cm degradation is architecture-specific.

The design trigger is satisfied when:

1. MiT-B3 parity has passed;
2. a concrete failure mechanism is visible in paired masks and supported by at least one quantitative diagnostic; and
3. DeepLabV3+ shows the same directional resolution degradation.

This trigger permits a single-seed prototype. It is not yet a publication claim and does not require Mask2Former or three-seed baseline reproduction.

## Stage G2: hypothesis and method design

Select exactly one mechanism supported by G1. Do not brainstorm or implement multiple architectures in parallel.

Examples of mechanism-to-method routing:

| Observed dominant mechanism | First method hypothesis to test |
|---|---|
| Small mortality regions disappear and recall collapses | Multi-resolution semantic consistency |
| Errors remain strongly dependent on known GSD after size control | GSD-conditioned feature modulation or decoder |
| Pixel overlap is acceptable but physical boundaries/regions collapse | Physical-area-aware small-region or boundary objective |

For a multi-resolution consistency prototype, the registered form is:

`L = L_seg + lambda_cons * [D(down(P_5), P_10) + D(down(P_10), P_20)]`.

For GSD conditioning, the registered form is feature-wise modulation:

`F'_l = gamma_l(g) * F_l + beta_l(g)`, where `g` is metres per pixel.

For physical small-region preservation, region size is always measured as:

`A_physical = A_pixels * GSD^2`.

The exact method is chosen only after G1. A literature search is then restricted to the selected mechanism and method family.

## Stage G3: rapid research loop

Each hypothesis is tested through one controlled loop:

`observation -> hypothesis -> algorithm v0 -> single-seed prototype -> metrics/visuals -> revise or reject`.

Prototype evaluation must include:

- mortality and tree-cover IoU, F1, precision, and recall;
- 5/10/20 cm breakdown;
- the mechanism-specific metric, such as small-region recall or Boundary F1;
- baseline-versus-method error maps on the same scenes;
- clean-resolution performance to detect regressions.

An example of supporting evidence is improvement at 20 cm together with improvement in small-object recall while 5 cm performance remains stable. Aggregate F1 improvement alone is insufficient to validate a mechanism.

## Stage G4: publication evaluation

Only after a prototype shows a mechanism-consistent signal do we run the complete comparison:

- U-Net ResNet-34;
- DeepLabV3+ ResNet-50;
- SegFormer MiT-B3;
- Mask2Former Small if compute permits;
- the frozen proposed method;
- seeds `0`, `100`, and `200`;
- ablation, cross-resolution, cross-biome/site, boundary, component-size, and qualitative analysis.

Published reference targets for later parity checks are retained below:

| Model | Mortality F1 | Mortality IoU | Mortality Precision | Mortality Recall | Tree-cover F1 |
|---|---:|---:|---:|---:|---:|
| U-Net (ResNet-34) | 0.56 | 0.42 | 0.70 | 0.52 | 0.89 |
| DeepLabV3+ (ResNet-50) | 0.55 | 0.41 | 0.71 | 0.49 | 0.88 |
| SegFormer (MiT-B3) | 0.59 | 0.45 | 0.72 | 0.54 | 0.89 |
| Mask2Former (Small) | 0.57 | 0.43 | 0.71 | 0.52 | 0.89 |

The publication claim gate requires three-seed consistency, paired site-level evidence, uncertainty intervals for principal deltas, and a mechanism-specific improvement. This stronger gate applies to the final claim, not to permission to begin method design.

## Expected artifacts

| Artifact | Suggested path | Success criterion |
|---|---|---|
| Frozen benchmark manifest | `dte_runs/manifests/` | 25 sites, 525 tiles, paired scales, hashed |
| Environment lock | `dte_runs/environment/` | Python/PyTorch/CUDA/package versions recorded |
| G0 parity outputs | `dte_runs/g0_parity/` | Official checkpoint metrics within tolerance |
| Tile confusion counts | `dte_runs/predictions/<model>/<seed>/per_tile.parquet` | One row per benchmark tile |
| Site/class metrics | `dte_runs/analysis/per_site_metrics.parquet` | Recomputable from stored counts |
| Component diagnostics | `dte_runs/analysis/components.parquet` | GSD, physical area, split/merge/miss fields present |
| Gate-0 report | `dte_runs/reports/gate0.md` | Explicit PASS/FAIL with target deltas |
| Minimal diagnosis | `dte_runs/reports/minimal_diagnosis.md` | Paired visuals, error mechanism, DeepLab sanity result |
| Prototype runs | `dte_runs/prototypes/<hypothesis>/<version>/` | Single controlled hypothesis per version |
| Full evaluation | `dte_runs/final/` | Three seeds, full baselines, ablations, robustness, visuals |

## Current readiness (2026-08-25)

- Official code cloned to `DTE-aerial-official/` at the pinned commit above.
- Official MiT-B3 checkpoint downloaded to `DTE-aerial-model/DTE_aerial_model.safetensors`.
- Checkpoint SHA-256: `4ddd6dbe3eee496e606d8cf428266d5cc7221bdabef3dddf1ad3b7a23b30976d`.
- Checkpoint load and a 512x512 CUDA forward pass succeed on the local RTX 5880 Ada (48 GB).
- Existing `DeadTrees/` data is a 100-tile sample of deadtrees.earth product layers and is not DTE-aerial-train or DTE-aerial-bench.
- Blocking input: Harvard Dataverse DOI `10.7910/DVN/IYCUML` currently requires authorization (`401`), and no Dataverse token is configured locally.
- Storage warning: only about 121 GB is free. The full 385K-tile training release must be size-checked before download; do not start it until adequate storage is reserved.
- Official repository issues to account for in a reproducibility patch without changing the scientific protocol: README calls `evaluation.py` while the file is `eval.py`; `requirements.txt` currently concatenates `pyarrow` and `from`; train/validation CSV paths remain placeholders.

## Immediate execution order

1. **Obtain data** — resolve Dataverse `401` (token or manual download). Target: `datasets/DTE-aerial-bench/`.
2. **Run G0 + G1 as daemons** — once bench arrives:
   ```bash
   mkdir -p dte_runs/logs
   BENCH=datasets/DTE-aerial-bench bash scripts/dte_launch.sh
   # G0 chains into G1 automatically on PASS
   ```
3. **G0 PASS** — site-macro parity within 0.03 absolute F1 and `5cm > 10cm > 20cm`. **STOP reproduction here.** Do not run U-Net, Mask2Former, or 12-run baselines.
4. **G1 visual review** — open `dte_runs/g1_diagnosis/minimal_diagnosis.md` and figures. Identify dominant failure mechanism from paired masks.
5. **DeepLabV3+ sanity** — single-seed only, to confirm task-level failure (not architecture-specific).
6. **G2 — select one mechanism** — one of: small-region recall collapse / GSD-dependent shift / boundary collapse. Targeted literature search only for that mechanism.
7. **G3 — prototype loop** — single hypothesis, single seed, mechanism-specific metrics, revise.
8. **G4 — full evaluation** — only after prototype shows mechanism-consistent signal. Full baselines, 3 seeds, ablations, cross-biome.

---

## ⚡ Transition: G0 PASS → Research starts immediately

The moment `gate0.md` shows `**Verdict: PASS**`:

```
Do NOT:  continue baseline reproduction
Do NOT:  run Mask2Former or U-Net
Do NOT:  wait for 12 seeds

Do:      open dte_runs/g1_diagnosis/figures/
Do:      read minimal_diagnosis.md
Do:      identify one mechanism
Do:      design algorithm v0
```

Full baselines (G4) are a publication formality, not a research gate.
