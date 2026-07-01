# Expert Analysis Report — Stage 4C

## Expert Load Distribution

| expert | tokens | fraction |
|--------|--------|----------|
| expert_0 | 147,886 | 25.1% |
| expert_1 | 147,737 | 25.0% |
| expert_2 | 140,270 | 23.8% |
| expert_3 | 154,028 | 26.1% |

## Expert–Class Matrix

| expert | background | water | soil/impervious | vegetation | building | farmland | low_veg |
|--------|--------|--------|--------|--------|--------|--------|--------|
| expert_0 | 0 (0%) | 8472 (6%) | 6558 (4%) | 8303 (6%) | 1630 (1%) | 0 (0%) | 122923 (83%) |
| expert_1 | 0 (0%) | 5043 (3%) | 8944 (6%) | 14341 (10%) | 1741 (1%) | 0 (0%) | 117668 (80%) |
| expert_2 | 0 (0%) | 9492 (7%) | 5346 (4%) | 8796 (6%) | 2036 (1%) | 0 (0%) | 114600 (82%) |
| expert_3 | 0 (0%) | 4836 (3%) | 8869 (6%) | 13936 (9%) | 1924 (1%) | 0 (0%) | 124463 (81%) |

## Purity Summary

| expert | dominant_class | purity | interpretation |
|--------|----------------|--------|----------------|
| expert_0 | low_veg | 0.831 | strong specialization ✓ |
| expert_1 | low_veg | 0.796 | strong specialization ✓ |
| expert_2 | low_veg | 0.817 | strong specialization ✓ |
| expert_3 | low_veg | 0.808 | strong specialization ✓ |

## Specialization Conclusions

- **Expert 0** (25% of tokens): dominant=low_veg, purity=0.831 → **strong** specialization, change_ratio=0.21
- **Expert 1** (25% of tokens): dominant=low_veg, purity=0.796 → **strong** specialization, change_ratio=0.24
- **Expert 2** (24% of tokens): dominant=low_veg, purity=0.817 → **strong** specialization, change_ratio=0.43
- **Expert 3** (26% of tokens): dominant=low_veg, purity=0.808 → **strong** specialization, change_ratio=0.43

## Recommendations

- Train Stage 4D with `--router_version v2` and `--expert_dropout 0.1`
- Monitor `expert_fracs` in training log for convergence toward uniform load
- Re-run this analysis after Stage 4D training to compare purity scores
