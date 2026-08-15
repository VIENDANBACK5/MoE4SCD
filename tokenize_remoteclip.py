import os
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from remoteclip_encoder import RemoteCLIPRegionEncoder

SECOND_ROOT = "SECOND"
REMOTECLIP_WEIGHTS = "/home/chung/.cache/huggingface/hub/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-L-14.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = [
    ("train", "tokens_T1_v2", "tokens_T1_rc", "T1"),
    ("train", "tokens_T2_v2", "tokens_T2_rc", "T2"),
    ("test",  "tokens_T1_test_v2", "tokens_T1_test_rc", "T1"),
    ("test",  "tokens_T2_test_v2", "tokens_T2_test_rc", "T2"),
]

encoder = RemoteCLIPRegionEncoder(
    weights_path=REMOTECLIP_WEIGHTS,
    device=DEVICE
)

for (split, orig_tokens_dir, out_dir, time) in SPLITS:
    os.makedirs(f"{SECOND_ROOT}/{out_dir}", exist_ok=True)
    stems = sorted([Path(f).stem for f in glob.glob(f"{SECOND_ROOT}/{orig_tokens_dir}/*.pt")])

    print(f"\nProcessing {split} {time}: {len(stems)} stems -> {out_dir}")

    for stem in tqdm(stems):
        orig_path = f"{SECOND_ROOT}/{orig_tokens_dir}/{stem}.pt"
        out_path  = f"{SECOND_ROOT}/{out_dir}/{stem}.pt"

        if os.path.exists(out_path):
            continue

        # Load original tokens (centroids, areas, spectral are kept)
        orig = torch.load(orig_path, map_location="cpu")
        N = orig["tokens"].shape[0]

        # Load image and masks
        img_dir   = f"{SECOND_ROOT}/im1" if time == "T1" else f"{SECOND_ROOT}/im2"
        masks_dir = f"{SECOND_ROOT}/sam2_masks_T1" if time == "T1" else f"{SECOND_ROOT}/sam2_masks_T2"
        if split == "test":
            img_dir   = f"{SECOND_ROOT}/test/im1" if time == "T1" else f"{SECOND_ROOT}/test/im2"
            masks_dir = f"{SECOND_ROOT}/sam2_masks_T1_test" if time == "T1" else f"{SECOND_ROOT}/sam2_masks_T2_test"

        img_path  = os.path.join(img_dir, stem + ".png")
        mask_path = os.path.join(masks_dir, stem + ".npz")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        masks = np.load(mask_path)["masks"]  # (N_masks, H, W)

        N_align = min(N, len(masks))

        # Extract RemoteCLIP region features
        rc_feats = encoder.encode_all_regions(image, masks[:N_align])

        new_token = {
            "tokens":    torch.from_numpy(rc_feats).float(),  # (N_align, 768)
            "centroids": orig["centroids"][:N_align],
            "areas":     orig["areas"][:N_align],
        }
        
        # Preserve spectral features and cvs if they exist
        if "spectral" in orig:
            new_token["spectral"] = orig["spectral"][:N_align]
        if "cvs" in orig:
            new_token["cvs"] = orig["cvs"][:N_align]

        torch.save(new_token, out_path)

print("\n✅ RemoteCLIP tokenization done.")
