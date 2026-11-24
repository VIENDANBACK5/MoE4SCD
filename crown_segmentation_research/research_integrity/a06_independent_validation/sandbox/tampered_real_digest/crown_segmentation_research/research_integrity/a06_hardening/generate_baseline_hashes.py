"""Generate baseline SHA256 checksums for all critical research artifacts."""
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = Path(__file__).resolve().parent

FILES = [
    "crown_segmentation_research/execution_plan_audit_2026-10-15/BAO_CAO_A00_A05.md",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/blockers.json",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/data_summary.json",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/spatial_summary.json",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/border_overlap_summary.json",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/cross_grid_summary.json",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/area_definition_audit.json",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/historical_hash_check.json",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/audit.py",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/check_metrics.py",
    "crown_segmentation_research/execution_plan_audit_2026-10-15/measurement_spec_v3_draft.md",
    "crown_segmentation_research/clean_benchmark/dataset.py",
    "crown_segmentation_research/clean_benchmark/targets_fast.py",
    "crown_segmentation_research/clean_benchmark/evaluate.py",
    "crown_segmentation_research/clean_benchmark/seed_scorer.py",
    "crown_segmentation_research/diagnosis_2026-10-11/study.py",
    "crown_segmentation_research/diagnosis_2026-10-11/fgq_revision_2026-10-13/grouping.py",
    "crown_segmentation_research/diagnosis_2026-10-11/fgq_revision_2026-10-13/q_sanity.py",
]

def generate():
    hashes = {}
    for rel in FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes[rel] = {"bytes": p.stat().st_size, "sha256": h}
        else:
            hashes[rel] = "MISSING"
            
    out_file = BASE_DIR / "BASELINE_HASHES.json"
    out_file.write_text(json.dumps(hashes, indent=2))
    print(f"Generated BASELINE_HASHES.json with {len(hashes)} items.")

if __name__ == "__main__":
    generate()
