"""
What each existing checkpoint has seen, and a guard against initialising new runs from the leaked ones.

Audit of 28/09/2026 (meeting/week1/KIEM_TOAN_TOAN_BO_KET_QUA_NGHIEN_CUU.md):
  - tree_flow_exhaustive_v1 was trained on BAMFORESTS train + Quebec Z1, Z2 AND Z3 (the test zone): the cached
    npz directory it read holds 200 patches of every zone and was never filtered by zone.
  - tree_flow_v2_warmstart / v3 / hr (and exploitation_plan_results/clean_hr_50ep, trained on Z1+Z2) all
    warm-start from that chain.
  - SimMIM encoders (methods/ssl/*) were pretrained on the same 600 patches, Z3 imagery included.
Any run initialised from one of them has seen the Quebec test zone (and BAMFORESTS), so its Z2/Z3/Bamberg numbers are
not held-out and it cannot be called zero-shot on Bamberg.
"""
from __future__ import annotations

from pathlib import Path

LEAKED_CHECKPOINT_PARTS = (
    "tree_flow_exhaustive_v1",
    "tree_flow_v2_warmstart",
    "tree_flow_v2_ssl",
    "tree_flow_v2_run",
    "tree_flow_v3",
    "tree_flow_hr",
    "clean_hr_50ep",
    "exploitation_plan_results",
    "ssl_in_domain_weights",
    "simmim",
)


def check_clean_init(path: str | Path | None, allow_leaked: bool = False) -> None:
    """Refuse to warm-start from a checkpoint whose lineage includes the Quebec test zone.

    `allow_leaked=True` is for reproducing the old runs only; results of such runs are not held-out."""
    if not path:
        return
    p = str(path)
    hit = next((part for part in LEAKED_CHECKPOINT_PARTS if part in p), None)
    if hit is None:
        return
    msg = (f"{p} descends from '{hit}', which saw Quebec Z3 (test) during training (see lineage.py). "
           f"Initialise from ImageNet instead, or pass --allow-leaked-init to reproduce an old run.")
    if not allow_leaked:
        raise SystemExit(msg)
    print("WARNING (lineage): " + msg, flush=True)
