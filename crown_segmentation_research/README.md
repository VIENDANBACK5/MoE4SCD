# Individual Tree Crown (ITC) Instance Segmentation Research

A unified research repository for state-of-the-art Individual Tree Crown (ITC) instance segmentation in high-resolution optical aerial orthomosaics across dense, continuous forest canopies.

---

## 📁 Modular Directory Structure

```
crown_segmentation_research/
├── datasets/                            # Dataset loaders, manifest caches, target helpers
│   ├── bam_coco_dataset.py              # BAMFORESTS COCO dataset streamer (1024x1024 crops)
│   ├── bam_topometric_dataset.py        # TopoMetric multi-task flow & saddle target dataset
│   └── bam_zip_manifest.json            # Fast index cache for Bamberg_coco2048.zip
│
├── methods/                             # Core segmentation paradigms (self-contained modules)
│   │
│   ├── canopy_watershed/                # [Option A: Neural Potential + Persistence Watershed]
│   │   ├── model.py                     # ResNet50-FPN Multi-Head (Surface U, Boundary B, Canopy M)
│   │   ├── dataset.py                   # 1:1 Native Resolution crop streaming dataset
│   │   ├── targets.py                   # Euclidean Distance Transform & Saddle Ridge generators
│   │   ├── decode.py                    # C++ Dilation Apex Extraction & Morphological Watershed
│   │   ├── train.py                     # End-to-end multi-task training script
│   │   ├── evaluate.py                  # Evaluation & multi-biome 4-panel visual preview generator
│   │   ├── sweep.py                     # Parallel hyperparameter grid search
│   │   ├── test_scale.py                # Scale calibration & distance threshold tests
│   │   └── test_surface.py              # Potential surface thresholding diagnostics
│   │
│   ├── topometric_flow/                 # [Option B: Centripetal Vector Flow + GPU Euler Transport]
│   │   ├── model.py                     # TopoMetricFlowNet (Flow V, Saddle S, Surface U, Canopy C)
│   │   ├── dataset.py                   # TopoMetric batch loader
│   │   ├── targets.py                   # Centripetal flow vector field & saddle barrier targets
│   │   ├── decode.py                    # 100% GPU Tensorized Euler transport & Sink clustering
│   │   ├── train.py                     # End-to-end multi-task training pipeline
│   │   └── evaluate.py                  # TopoMetric flow evaluation & preview script
│   │
│   ├── star_convex/                     # [Baseline 1: StarDist / Radial Ray Casting]
│   │   ├── model.py                     # StarConvexNet (Center probability + 32/64 radial rays)
│   │   ├── targets.py                   # Polar ray distance transform target computation
│   │   ├── decode.py                    # Radial ray polygon reconstruction & NMS
│   │   ├── train.py                     # Star-convex training pipeline
│   │   ├── evaluate.py                  # Evaluation on held-out validation tiles
│   │   ├── evaluate_with_merge.py       # Star-convex with post-hoc instance merging
│   │   ├── sweep.py                     # Decoding parameter grid search
│   │   ├── diagnose.py                  # Visual failure analysis on non-convex touching crowns
│   │   └── precompute_targets.py        # Target precomputation script
│   │
│   ├── foundation_sam/                  # [Baseline 2: Prompted Foundation Model / SAM AMG]
│   │   ├── model.py                     # CrownTransformerSAM (Transformer queries -> SAM masks)
│   │   ├── train_bam.py                 # BAMFORESTS fine-tuning for CrownTransformer
│   │   ├── train_sam.py                 # SAM decoder adapter training
│   │   ├── evaluate.py                  # SAM2 AMG evaluation and tile preview
│   │   └── dense_grid_inference.py      # Dense prompt-grid SAM benchmark
│   │
│   ├── tree_flow/                       # [Cellpose-style Flow & Graph Decoder]
│   │   ├── model.py                     # TreeFlowNet (2D vector flow + SDT + centroid)
│   │   ├── targets.py                   # Normalized gradient vector flow targets
│   │   ├── decode.py                    # CPU / GPU Euler trajectory integration
│   │   ├── train.py                     # Training script (40 epochs converged)
│   │   ├── evaluate.py                  # Evaluation on validation sites
│   │   ├── eval_dte_bench.py            # DTE-Aerial benchmark evaluation
│   │   ├── preview.py                   # Visual preview generation
│   │   ├── sweep.py                     # Flow decode parameter sweep
│   │   ├── graph_decode.py              # Divergence-based graph partition decoder
│   │   └── standalone_flow.py           # Standalone single-script flow segmenter
│   │
│   └── dynamic_mask/                    # [Deadwood & Dynamic Mask Experiments]
│       ├── dynamic_crown_mask_net.py    # Dynamic filter mask network
│       ├── train_dynamic_crown_mask.py  # Training pipeline
│       ├── neuro_deadwood_net.py        # Unified deadwood segmentation network
│       ├── neuro_deadwood_loss.py       # Deadwood multi-task loss
│       ├── neuro_deadwood_decode.py     # Deadwood instance decoder
│       └── train_neuro_deadwood.py      # Deadwood training script
│
├── evaluation/                          # Unified cross-paradigm evaluation & plotting
│   ├── benchmark_all_methods.py         # Side-by-side benchmark across all 4 paradigms
│   ├── generate_master_comparison_figure.py # 6-panel publication comparison figure generator
│   ├── visualize_gt_vs_pred.py          # Ground truth vs prediction visual inspector
│   ├── diagnose_sample0.py              # Detailed IoU diagnostic tool
│   ├── evaluate_bam_metrics.py          # Fast COCO / Panoptic metric evaluation
│   ├── evaluate_merge.py                # Post-hoc instance merge benchmarking
│   └── merge_module.py                  # Graph-based instance merge module
│
├── legacy/                              # Scratch modules, unit tests, exploratory code
│   ├── tests/                           # Unit tests for losses and representations
│   ├── dense_hv_representation.py       # Horizontal-Vertical gradient representation
│   ├── discriminative_loss.py           # Metric embedding contrastive loss
│   └── treecog_lite.py                  # Lightweight TreeCog prototype
│
├── docs/                                # Theoretical notes & research documentation
│   └── note.md                          # Comprehensive research notes & paradigm analysis
│
└── images/                              # Visual galleries & publication figures
    ├── master_paradigm_comparison.png   # 6-panel cross-paradigm comparison
    ├── previews_canopy_watershed/       # 4-panel visual previews across 5 biomes
    └── diagnostics/                     # Side-by-side Ground Truth vs Pred comparisons
```

---

## 🚀 Quick Start Commands

All commands can be run directly from the repository root `/home/chung/RS/Image Segmentation`:

### 1. Evaluate CanopyWatershedNet (Option A)
```bash
./cs2_venv/bin/python3 crown_segmentation_research/methods/canopy_watershed/evaluate.py --samples 50
```

### 2. Train CanopyWatershedNet from Scratch
```bash
./cs2_venv/bin/python3 crown_segmentation_research/methods/canopy_watershed/train.py --epochs 15 --batch_size 4
```

### 3. Run Unified Multi-Paradigm Benchmark
```bash
./cs2_venv/bin/python3 crown_segmentation_research/evaluation/benchmark_all_methods.py --samples 40
```

### 4. Generate Master 6-Panel Comparison Figure
```bash
./cs2_venv/bin/python3 crown_segmentation_research/evaluation/generate_master_comparison_figure.py
```

### 5. Train & Evaluate TopoMetric Flow (Option B)
```bash
# Train TopoMetric Flow
./cs2_venv/bin/python3 crown_segmentation_research/methods/topometric_flow/train.py --epochs 12

# Evaluate TopoMetric Flow
./cs2_venv/bin/python3 crown_segmentation_research/methods/topometric_flow/evaluate.py
```

### 6. Evaluate StarDist Baseline (Baseline 1)
```bash
./cs2_venv/bin/python3 crown_segmentation_research/methods/star_convex/evaluate.py
```

---

## 📊 Paradigm Comparison Summary

| Method | Paradigm | External Weights | Latency (ms) | FPS | Memory | Handling Touching Seams |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **StarDist** | Radial Rays $R(\theta)$ | None (0%) | 184.5 ms | 5.42 | 1.8 GB | Truncated / Flattened |
| **CrownTransformerSAM** | Prompted SAM AMG ViT-H | SAM ViT-H (100%) | 13,450.0 ms | 0.07 | 16.4 GB | Overlapping Circular Discs |
| **TreeFlowNet** | Centripetal Flow Field | None (0%) | 142.3 ms | 7.03 | 2.1 GB | Topological Sinks |
| **CanopyWatershedNet (Ours)** | Neural Potential + Watershed | **None (0%)** | **129.2 ms** | **7.74** | **2.1 GB** | **Natural Non-Convex Basins** |
