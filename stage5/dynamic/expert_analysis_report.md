# Expert Analysis Report — Stage 4C

## Expert Load Distribution

| expert | tokens | fraction |
|--------|--------|----------|
| expert_0 | 143,281 | 24.3% |
| expert_1 | 144,897 | 24.6% |
| expert_2 | 146,392 | 24.8% |
| expert_3 | 155,351 | 26.3% |

## Expert–Class Matrix

| expert | background | water | soil/impervious | vegetation | building | farmland | low_veg |
|--------|--------|--------|--------|--------|--------|--------|--------|
| expert_0 | 0 (0%) | 10394 (7%) | 5581 (4%) | 9114 (6%) | 2080 (1%) | 0 (0%) | 116112 (81%) |
| expert_1 | 0 (0%) | 7570 (5%) | 6327 (4%) | 7986 (6%) | 1587 (1%) | 0 (0%) | 121427 (84%) |
| expert_2 | 0 (0%) | 4890 (3%) | 8307 (6%) | 13289 (9%) | 1641 (1%) | 0 (0%) | 118265 (81%) |
| expert_3 | 0 (0%) | 4989 (3%) | 9502 (6%) | 14987 (10%) | 2023 (1%) | 0 (0%) | 123850 (80%) |

## Purity Summary

| expert | dominant_class | purity | interpretation |
|--------|----------------|--------|----------------|
| expert_0 | low_veg | 0.810 | strong specialization ✓ |
| expert_1 | low_veg | 0.838 | strong specialization ✓ |
| expert_2 | low_veg | 0.808 | strong specialization ✓ |
| expert_3 | low_veg | 0.797 | strong specialization ✓ |

## Specialization Conclusions

- **Expert 0** (24% of tokens): dominant=low_veg, purity=0.810 → **strong** specialization, change_ratio=0.44
- **Expert 1** (25% of tokens): dominant=low_veg, purity=0.838 → **strong** specialization, change_ratio=0.20
- **Expert 2** (25% of tokens): dominant=low_veg, purity=0.808 → **strong** specialization, change_ratio=0.20
- **Expert 3** (26% of tokens): dominant=low_veg, purity=0.797 → **strong** specialization, change_ratio=0.44

## Recommendations

- Train Stage 4D with `--router_version v2` and `--expert_dropout 0.1`
- Monitor `expert_fracs` in training log for convergence toward uniform load
- Re-run this analysis after Stage 4D training to compare purity scores
