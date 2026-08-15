# evaluate_and_report.py
import argparse
import subprocess
import sys
from pathlib import Path

def run_eval(checkpoint_path: str, output_name: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"Error: Checkpoint {checkpoint} does not exist!")
        sys.exit(1)

    print(f"=== [1/3] Running eval_test_set.py for {checkpoint_path} ===")
    pred_dir = f"output/{output_name}_preds"
    eval_cmd = [
        "python", "eval_test_set.py",
        "--checkpoint", checkpoint_path,
        "--tokens_T1", "SECOND/tokens_T1_test_v2",
        "--tokens_T2", "SECOND/tokens_T2_test_v2",
        "--matches", "SECOND/matches_test",
        "--use_spectral",
        "--save-preds", pred_dir,
        "--device", "cuda"
    ]
    subprocess.run(eval_cmd, check=True)

    print(f"=== [2/3] Running token_to_object_predictions.py ===")
    pred_json = f"SECOND-OC/predictions/predictions_{output_name}.json"
    convert_cmd = [
        "python", "SECOND-OC/baselines/token_to_object_predictions.py",
        "--token-dir", f"{pred_dir}/tokens",
        "--out", pred_json
    ]
    subprocess.run(convert_cmd, check=True)

    print(f"=== [3/3] Running object_eval.py ===")
    out_json = f"SECOND-OC/baseline_results/{output_name}_results.json"
    eval_obj_cmd = [
        "python", "SECOND-OC/eval/object_eval.py",
        "--gt", "SECOND-OC/annotations/change_annotations.json",
        "--pred", pred_json,
        "--out", out_json
    ]
    subprocess.run(eval_obj_cmd, check=True)

    print(f"=== Evaluation completed successfully for {output_name}! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint best_model.pt")
    parser.add_argument("--name", required=True, help="Output name prefix (e.g. spectral_transition)")
    args = parser.parse_args()
    run_eval(args.checkpoint, args.name)
