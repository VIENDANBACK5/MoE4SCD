#!/usr/bin/env python3
"""
extract_run_metrics.py
======================
Parses training_log.csv and config.json from a run directory,
extracts the best epoch metrics, and updates the experiment database.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def parse_run_directory(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    log_csv = run_dir / "training_log.csv"
    config_json = run_dir / "config.json"
    
    if not log_csv.exists():
        print(f"Error: {log_csv} not found.", file=sys.stderr)
        sys.exit(1)
        
    # Read config
    config_data = {}
    if config_json.exists():
        try:
            with open(config_json, "r") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read config.json: {e}", file=sys.stderr)
            
    # Read log csv and find best epoch
    # We define best epoch based on highest val_f1, or lowest val_loss if F1 is not present
    best_row = None
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    
    epochs_data = []
    
    with open(log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs_data.append(row)
            # Try parsing F1/IoU/loss
            try:
                epoch = int(row.get("epoch", 0))
                val_loss = float(row.get("val_total", row.get("val_loss", float("inf"))))
                val_f1 = float(row.get("val_f1", 0.0))
                val_iou = float(row.get("val_iou", 0.0))
                val_change = float(row.get("val_change", 0.0))
                val_delta = float(row.get("val_delta", 0.0))
            except ValueError:
                continue
                
            # Decision rule for "best" epoch
            if val_f1 > 0:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_row = row
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_row = row
                    
    if best_row is None:
        print("Error: No valid epochs parsed from CSV.", file=sys.stderr)
        sys.exit(1)
        
    # Build structured dict
    metrics = {
        "epoch": int(best_row["epoch"]),
        "train_loss": float(best_row["train_total"]),
        "val_loss": float(best_row.get("val_total", best_row.get("val_loss", 0.0))),
        "val_change": float(best_row.get("val_change", 0.0)),
        "val_delta": float(best_row.get("val_delta", 0.0)),
        "val_f1": float(best_row.get("val_f1", 0.0)),
        "val_iou": float(best_row.get("val_iou", 0.0)),
        "lr": best_row.get("lr", ""),
        "time_s": float(best_row.get("time_s", 0.0))
    }
    
    # Expert fractions for MoE if present
    if "expert_fracs" in best_row and best_row["expert_fracs"]:
        metrics["expert_fracs"] = [float(f) for f in best_row["expert_fracs"].split("|") if f]
        
    return {
        "run_name": run_dir.name,
        "config": config_data,
        "best_epoch_metrics": metrics,
        "total_epochs": len(epochs_data)
    }

def update_database(db_path: Path, run_data: Dict[str, Any]):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    database = {"runs": []}
    if db_path.exists():
        try:
            with open(db_path, "r") as f:
                database = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing database, overwriting. Reason: {e}")
            
    # Remove existing run with same name if it exists to avoid duplication
    database["runs"] = [r for r in database["runs"] if r["run_name"] != run_data["run_name"]]
    database["runs"].append(run_data)
    
    with open(db_path, "w") as f:
        json.dump(database, f, indent=2)
        
    print(f"Successfully updated experiment database at: {db_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract run metrics and update local experiment DB.")
    parser.add_argument("run_dir", type=str, help="Directory containing config.json and training_log.csv")
    parser.add_argument("--db_path", type=str, default="docs/research_agent/experiments/experiment_database.json",
                        help="Path to the database JSON file")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    db_path = Path(args.db_path)
    
    run_data = parse_run_directory(run_dir)
    update_database(db_path, run_data)
    
    # Print out summary report in Markdown format
    best = run_data["best_epoch_metrics"]
    print("\n" + "="*50)
    print(f"RUN SUMMARY: {run_data['run_name']}")
    print("="*50)
    print(f"Model Type:    {run_data['config'].get('model_type', 'unknown')}")
    print(f"Best Epoch:    {best['epoch']} (out of {run_data['total_epochs']})")
    print(f"Val F1 Score:  {best['val_f1']:.4f}")
    print(f"Val IoU:       {best['val_iou']:.4f}")
    print(f"Val Change:    {best['val_change']:.4f}")
    print(f"Val Loss:      {best['val_loss']:.4f}")
    if "expert_fracs" in best:
        print(f"Expert Load:   {best['expert_fracs']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
