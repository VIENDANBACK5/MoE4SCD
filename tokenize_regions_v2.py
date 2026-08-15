# tokenize_regions_v2.py
"""
Giống tokenize_regions.py nhưng thêm spectral features vào mỗi token.
Dùng centroid-area matching để map chính xác từng token trong .pt file 
về mask tương ứng trong .npz file.
Sử dụng ProcessPoolExecutor để xử lý song song, tối ưu hóa tốc độ.
"""
import os
import glob
import torch
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from spectral_extractor import extract_spectral_for_stem

# ── Config ───────────────────────────────────────────────────────────────────
TOKENS_T1_DIR    = "SECOND/tokens_T1"          # tokens gốc (không sửa)
TOKENS_T2_DIR    = "SECOND/tokens_T2"
MASKS_T1_DIR     = "SECOND/sam2_masks_T1"      # SAM2 masks
MASKS_T2_DIR     = "SECOND/sam2_masks_T2"
IM1_DIR          = "SECOND/im1"                 # ảnh RGB gốc
IM2_DIR          = "SECOND/im2"
OUT_T1_DIR       = "SECOND/tokens_T1_v2"       # output mới
OUT_T2_DIR       = "SECOND/tokens_T2_v2"
# ─────────────────────────────────────────────────────────────────────────────

def process_single_stem(args):
    stem, tokens_dir, masks_dir, masks_other_dir, im_dir, im_other_dir, out_dir, time = args
    token_path  = os.path.join(tokens_dir, stem + ".pt")
    mask_path   = os.path.join(masks_dir,  stem + ".npz")
    out_path    = os.path.join(out_dir,    stem + ".pt")

    if not os.path.exists(mask_path):
        return False, "no_mask"

    try:
        # Load token gốc
        token_data = torch.load(token_path, map_location="cpu")
        centroids = token_data["centroids"] # (N, 2)
        areas = token_data["areas"]         # (N,)
        N = centroids.shape[0]

        # Load raw masks và tính centroids/areas để match
        npz_data = np.load(mask_path)
        raw_masks = npz_data["masks"] # (M, H, W)
        M = raw_masks.shape[0]

        raw_centroids = []
        raw_areas = []
        for i in range(M):
            mask = raw_masks[i]
            ys, xs = np.where(mask)
            if len(ys) == 0:
                cx, cy = 0.5, 0.5
                area = 0.0
            else:
                cx = float(xs.mean()) / 511.0
                cy = float(ys.mean()) / 511.0
                area = float(mask.sum()) / (512.0 * 512.0)
            raw_centroids.append([cx, cy])
            raw_areas.append(area)
        raw_centroids = np.array(raw_centroids)
        raw_areas = np.array(raw_areas)

        # Extract spectral features cho toàn bộ M masks
        if time == "T1":
            spectral_all = extract_spectral_for_stem(
                stem, im_dir, im_other_dir, masks_dir,
                masks_other_dir
            )["spectral_T1"]
        else:
            spectral_all = extract_spectral_for_stem(
                stem, im_other_dir, im_dir,
                masks_other_dir, masks_dir
            )["spectral_T2"]

        # Match từng token k về mask i
        matched_spectral = []
        for k in range(N):
            cx, cy = centroids[k].tolist()
            area = areas[k].item()
            dists = (raw_centroids[:, 0] - cx)**2 + (raw_centroids[:, 1] - cy)**2 + (raw_areas - area)**2
            idx = np.argmin(dists)
            matched_spectral.append(spectral_all[idx])

        matched_spectral = np.array(matched_spectral, dtype=np.float32)

        # Tạo token mới giữ nguyên toàn bộ keys cũ và thêm spectral
        new_token = {
            "tokens":    token_data["tokens"],
            "centroids": token_data["centroids"],
            "areas":     token_data["areas"],
            "cvs":       token_data.get("cvs", torch.zeros(N)),
            "spectral":  torch.from_numpy(matched_spectral).float(),
        }
        torch.save(new_token, out_path)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def process_split(tokens_dir, masks_dir, masks_other_dir, im_dir, im_other_dir, out_dir, time="T1"):
    os.makedirs(out_dir, exist_ok=True)
    stems = sorted([Path(f).stem for f in glob.glob(f"{tokens_dir}/*.pt")])
    print(f"Processing {len(stems)} stems for {time} using ProcessPoolExecutor...")

    num_workers = min(16, multiprocessing.cpu_count())
    tasks = [(stem, tokens_dir, masks_dir, masks_other_dir, im_dir, im_other_dir, out_dir, time) for stem in stems]

    skipped = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_stem, tasks), total=len(stems)))

    for success, msg in results:
        if not success:
            if msg == "no_mask":
                skipped += 1
            else:
                print(f"Error: {msg}")
                errors += 1

    print(f"✅ Done split. Skipped: {skipped}/{len(stems)} | Errors: {errors}")

    # Validate 1 sample
    sample_path = os.path.join(out_dir, stems[0] + ".pt")
    d = torch.load(sample_path)
    assert "spectral" in d, "spectral key missing!"
    assert d["spectral"].shape[1] == 24, f"Expected 24, got {d['spectral'].shape[1]}"
    assert d["spectral"].shape[0] == d["tokens"].shape[0], f"N mismatch: spectral {d['spectral'].shape[0]} vs tokens {d['tokens'].shape[0]}"
    print(f"✅ Validation passed: {stems[0]}.pt has spectral {d['spectral'].shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--masks-T1", default=None)
    parser.add_argument("--masks-T2", default=None)
    parser.add_argument("--out-dir", default=None, help="If specified, only process T1 and output to this directory")
    args = parser.parse_args()

    if args.out_dir is not None:
        if args.split == "test":
            tokens_t1_dir = "SECOND/tokens_T1_test"
            masks_t1_dir = args.masks_T1 or "SECOND/sam2_masks_T1_test"
            masks_t2_dir = args.masks_T2 or "SECOND/sam2_masks_T2_test"
            im1_dir = "SECOND/test/im1"
            im2_dir = "SECOND/test/im2"
        else:
            tokens_t1_dir = "SECOND/tokens_T1"
            masks_t1_dir = args.masks_T1 or "SECOND/sam2_masks_T1"
            masks_t2_dir = args.masks_T2 or "SECOND/sam2_masks_T2"
            im1_dir = "SECOND/im1"
            im2_dir = "SECOND/im2"
        
        process_split(tokens_t1_dir, masks_t1_dir, masks_t2_dir, im1_dir, im2_dir, args.out_dir, "T1")
    else:
        if args.split == "test":
            tokens_t1 = "SECOND/tokens_T1_test"
            tokens_t2 = "SECOND/tokens_T2_test"
            masks_t1  = args.masks_T1 or "SECOND/sam2_masks_T1_test"
            masks_t2  = args.masks_T2 or "SECOND/sam2_masks_T2_test"
            im1       = "SECOND/test/im1"
            im2       = "SECOND/test/im2"
            out_t1    = "SECOND/tokens_T1_test_v2"
            out_t2    = "SECOND/tokens_T2_test_v2"
        else:
            tokens_t1 = "SECOND/tokens_T1"
            tokens_t2 = "SECOND/tokens_T2"
            masks_t1  = args.masks_T1 or "SECOND/sam2_masks_T1"
            masks_t2  = args.masks_T2 or "SECOND/sam2_masks_T2"
            im1       = "SECOND/im1"
            im2       = "SECOND/im2"
            out_t1    = "SECOND/tokens_T1_v2"
            out_t2    = "SECOND/tokens_T2_v2"

        process_split(tokens_t1, masks_t1, masks_t2, im1, im2, out_t1, "T1")
        process_split(tokens_t2, masks_t2, masks_t1, im2, im1, out_t2, "T2")

