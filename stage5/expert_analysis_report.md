# Expert Analysis Report — Stage 4C

## Expert Load Distribution

| expert | tokens | fraction |
|--------|--------|----------|
| expert_0 | 12,245 | 23.9% |
| expert_1 | 12,657 | 24.7% |
| expert_2 | 13,424 | 26.2% |
| expert_3 | 12,840 | 25.1% |

## Expert–Class Matrix

| expert | background | water | soil/impervious | vegetation | building | farmland | low_veg |
|--------|--------|--------|--------|--------|--------|--------|--------|
| expert_0 | 0 (0%) | 635 (5%) | 540 (4%) | 963 (8%) | 60 (0%) | 0 (0%) | 10047 (82%) |
| expert_1 | 0 (0%) | 484 (4%) | 679 (5%) | 1047 (8%) | 67 (1%) | 0 (0%) | 10380 (82%) |
| expert_2 | 0 (0%) | 269 (2%) | 889 (7%) | 1787 (13%) | 51 (0%) | 0 (0%) | 10428 (78%) |
| expert_3 | 0 (0%) | 273 (2%) | 876 (7%) | 1299 (10%) | 56 (0%) | 0 (0%) | 10336 (80%) |

## Purity Summary

| expert | dominant_class | purity | interpretation |
|--------|----------------|--------|----------------|
| expert_0 | low_veg | 0.820 | strong specialization ✓ |
| expert_1 | low_veg | 0.820 | strong specialization ✓ |
| expert_2 | low_veg | 0.777 | strong specialization ✓ |
| expert_3 | low_veg | 0.805 | strong specialization ✓ |

## Specialization Conclusions

- **Expert 0** (24% of tokens): dominant=low_veg, purity=0.820 → **strong** specialization, change_ratio=0.46
- **Expert 1** (25% of tokens): dominant=low_veg, purity=0.820 → **strong** specialization, change_ratio=0.24
- **Expert 2** (26% of tokens): dominant=low_veg, purity=0.777 → **strong** specialization, change_ratio=0.25
- **Expert 3** (25% of tokens): dominant=low_veg, purity=0.805 → **strong** specialization, change_ratio=0.45

## Recommendations

- Train Stage 4D with `--router_version v2` and `--expert_dropout 0.1`
- Monitor `expert_fracs` in training log for convergence toward uniform load
- Re-run this analysis after Stage 4D training to compare purity scores
