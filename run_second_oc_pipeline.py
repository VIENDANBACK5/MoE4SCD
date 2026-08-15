"""
run_second_oc_pipeline.py — Run SECOND-OC benchmark pipeline end-to-end.

Waits for SAM2 mask generation to finish, then runs Phase 0 → 1 → 2 → 3A → 4.
Each phase must pass its validation assertions before the next phase starts.

Log: SECOND-OC/pipeline_run.log
Run: nohup python3.11 run_second_oc_pipeline.py > /tmp/pipeline_stdout.log 2>&1 &
"""
import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("SECOND-OC/pipeline_run.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MASK_T1_DIR = Path("SECOND/sam2_masks_T1_test")
MASK_T2_DIR = Path("SECOND/sam2_masks_T2_test")
EXPECTED_MASKS = 1694
POLL_INTERVAL  = 60   # seconds between mask-count checks

PHASES = [
    ("Phase 0 — verify",        "SECOND-OC/phase0_verify.py"),
    ("Phase 1 — instances",     "SECOND-OC/phase1_extract_instances.py"),
    ("Phase 2 — changes",       "SECOND-OC/phase2_classify_changes.py"),
    ("Phase 3A — captions",     "SECOND-OC/phase3a_template_descriptions.py"),
    ("Phase 4 — package",       "SECOND-OC/phase4_package.py"),
]


def count_masks(d: Path) -> int:
    return len(list(d.glob("*.npz"))) if d.exists() else 0


def wait_for_masks():
    log.info("── Waiting for SAM2 mask generation to complete ──")
    while True:
        n1 = count_masks(MASK_T1_DIR)
        n2 = count_masks(MASK_T2_DIR)
        log.info(f"   T1: {n1}/{EXPECTED_MASKS}   T2: {n2}/{EXPECTED_MASKS}")
        if n1 >= EXPECTED_MASKS and n2 >= EXPECTED_MASKS:
            log.info("✅ SAM2 masks complete.")
            return
        remaining = max(EXPECTED_MASKS - n1, EXPECTED_MASKS - n2)
        eta_min = int(remaining * 2.2 / 60)   # ~2.2s/pair
        log.info(f"   ETA ≈ {eta_min} min — sleeping {POLL_INTERVAL}s ...")
        time.sleep(POLL_INTERVAL)


def run_phase(name: str, script: str) -> bool:
    log.info(f"\n{'='*60}")
    log.info(f"  START  {name}")
    log.info(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # let stdout/stderr pass through to log handler
    )

    elapsed = int(time.time() - t0)
    if result.returncode == 0:
        log.info(f"  ✅ {name} passed  ({elapsed}s)")
        return True
    else:
        log.error(f"  ❌ {name} FAILED (exit {result.returncode}, {elapsed}s)")
        log.error(f"     Fix the issue and re-run:  python3.11 {script}")
        return False


def main():
    log.info("=" * 60)
    log.info(f"  SECOND-OC Pipeline  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    wait_for_masks()

    for name, script in PHASES:
        ok = run_phase(name, script)
        if not ok:
            log.error("\n⛔ Pipeline stopped. See errors above.")
            sys.exit(1)

    log.info("\n" + "=" * 60)
    log.info("  ✅ ALL PHASES COMPLETE")
    log.info(f"  Benchmark: SECOND-OC/benchmark.json")
    log.info(f"  Tier1:     SECOND-OC/tier1/tier1.json")
    log.info(f"  Stats:     SECOND-OC/stats.json")
    log.info(f"  Log:       {LOG_PATH}")
    log.info("=" * 60)
    log.info("\nNext step:")
    log.info("  python SECOND-OC/eval/object_eval.py \\")
    log.info("    --gt SECOND-OC/annotations/change_annotations.json \\")
    log.info("    --pred <your_predictions.json>")


if __name__ == "__main__":
    main()
