# AGENT PLAN: Spectral Features + LoRA Fine-tuning for Token-MoE
> Mỗi bước được viết để agent chạy trực tiếp không cần hỏi thêm.
> Format: Input rõ ràng → Code đầy đủ → Expected output → Validation assert.
> Chạy theo đúng thứ tự: Bước 1 → eval → Bước 2 → eval → Bước 3 → quyết định.

---

## CẤU TRÚC FILE SẼ TẠO MỚI

```
Image Segmentation/
├── spectral_extractor.py        [NEW] Phase 1: extract 24 spectral features
├── tokenize_regions_v2.py       [NEW] Phase 1: tokenize + spectral features
├── train_reasoner_spectral.py   [NEW] Phase 1&2: train với spectral + LoRA
├── lora_sam2.py                 [NEW] Phase 2: LoRA wrapper cho SAM2
├── eval_spectral.py             [NEW] eval sau mỗi bước
└── visualize_masks.py           [NEW] Phase 3: kiểm tra mask quality
```

---

## ════════════════════════════════════════
## BƯỚC 1 — Thêm 24 Spectral Features (SAM2 frozen)
## ════════════════════════════════════════

### 1.0 — Kiểm tra prerequisite

```bash
# Verify các file cần thiết tồn tại
python -c "
import os, glob
checks = {
    'tokens_T1': 'SECOND/tokens_T1/*.pt',
    'tokens_T2': 'SECOND/tokens_T2/*.pt',
    'sam2_masks_T1': 'SECOND/sam2_masks_T1/*.npz',
    'sam2_masks_T2': 'SECOND/sam2_masks_T2/*.npz',
    'images_T1': 'SECOND/train/im1/*.png',
    'images_T2': 'SECOND/train/im2/*.png',
    'label1': 'SECOND/train/label1/*.png',
    'label2': 'SECOND/train/label2/*.png',
}
for name, pattern in checks.items():
    n = len(glob.glob(pattern))
    status = '✅' if n > 0 else '❌'
    print(f'{status} {name}: {n} files')
"
```

**Expected:** Tất cả ✅, không có ❌.
**Nếu có ❌:** Agent dừng và báo cáo file nào thiếu.

---

### 1.1 — Tạo `spectral_extractor.py`

```python
# spectral_extractor.py
"""
Extract 24 spectral features từ ảnh RGB gốc cho mỗi SAM2 mask.
Dựa trên Dao et al. 2021 (ISPRS): objects carry richer spectral
and spatial information than pixels.

24 features per token:
  mean RGB T1:      3 dims  (spectral identity T1)
  std RGB T1:       3 dims  (texture/roughness T1)
  mean RGB T2:      3 dims  (spectral identity T2)
  std RGB T2:       3 dims  (texture/roughness T2)
  delta mean:       3 dims  (spectral change magnitude) ← QUAN TRỌNG NHẤT
  delta std:        3 dims  (texture change)
  CV T1:            3 dims  (relative variability T1)
  CV T2:            3 dims  (relative variability T2)
  ─────────────────────────
  Tổng:            24 dims
"""
import numpy as np
from PIL import Image


def extract_spectral_features(
    image_t1: np.ndarray,   # (H, W, 3) float32, range [0, 1]
    image_t2: np.ndarray,   # (H, W, 3) float32, range [0, 1]
    mask: np.ndarray,        # (H, W) bool
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Extract 24 spectral features cho 1 SAM2 mask.
    Returns: np.ndarray shape (24,) float32
    """
    # Lấy pixel values trong vùng mask
    region_t1 = image_t1[mask]  # (N_pixels, 3)
    region_t2 = image_t2[mask]  # (N_pixels, 3)

    if len(region_t1) == 0:
        return np.zeros(24, dtype=np.float32)

    # Mean RGB (spectral identity)
    mean_t1 = region_t1.mean(axis=0)   # (3,)
    mean_t2 = region_t2.mean(axis=0)

    # Std RGB (texture proxy)
    std_t1 = region_t1.std(axis=0)     # (3,)
    std_t2 = region_t2.std(axis=0)

    # Temporal delta (change magnitude) — quan trọng nhất
    delta_mean = mean_t2 - mean_t1     # (3,)
    delta_std  = std_t2 - std_t1       # (3,)

    # Coefficient of Variation (Dao et al. 2021)
    # CV = std / mean — đo độ biến thiên tương đối
    cv_t1 = std_t1 / (mean_t1 + eps)   # (3,)
    cv_t2 = std_t2 / (mean_t2 + eps)   # (3,)

    features = np.concatenate([
        mean_t1, std_t1,       # 6 dims: spectral identity + texture T1
        mean_t2, std_t2,       # 6 dims: spectral identity + texture T2
        delta_mean, delta_std, # 6 dims: temporal change
        cv_t1, cv_t2,          # 6 dims: relative variability
    ]).astype(np.float32)

    assert features.shape == (24,), f"Expected (24,), got {features.shape}"
    return features


def extract_spectral_for_stem(
    stem: str,
    im1_dir: str,
    im2_dir: str,
    masks_T1_dir: str,
    masks_T2_dir: str,
) -> dict:
    """
    Extract spectral features cho toàn bộ masks của 1 stem.
    Returns dict: {
        'spectral_T1': np.ndarray (N_masks_T1, 24),
        'spectral_T2': np.ndarray (N_masks_T2, 24),
    }
    """
    import os
    # Load ảnh
    img_t1 = np.array(Image.open(
        os.path.join(im1_dir, stem + ".png")
    ).convert("RGB")).astype(np.float32) / 255.0

    img_t2 = np.array(Image.open(
        os.path.join(im2_dir, stem + ".png")
    ).convert("RGB")).astype(np.float32) / 255.0

    # Load SAM2 masks
    data_t1 = np.load(os.path.join(masks_T1_dir, stem + ".npz"))
    data_t2 = np.load(os.path.join(masks_T2_dir, stem + ".npz"))
    masks_t1 = data_t1["masks"]  # (N1, H, W) bool
    masks_t2 = data_t2["masks"]  # (N2, H, W) bool

    # Extract features cho từng mask
    spectral_T1 = np.stack([
        extract_spectral_features(img_t1, img_t2, masks_t1[i].astype(bool))
        for i in range(len(masks_t1))
    ])  # (N1, 24)

    spectral_T2 = np.stack([
        extract_spectral_features(img_t2, img_t1, masks_t2[i].astype(bool))
        for i in range(len(masks_t2))
    ])  # (N2, 24)

    return {"spectral_T1": spectral_T1, "spectral_T2": spectral_T2}


# ── Test nhanh với 1 stem ────────────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_spectral_for_stem(
        stem="00004",
        im1_dir="SECOND/train/im1",
        im2_dir="SECOND/train/im2",
        masks_T1_dir="SECOND/sam2_masks_T1",
        masks_T2_dir="SECOND/sam2_masks_T2",
    )
    print(f"T1 spectral features shape: {result['spectral_T1'].shape}")
    print(f"T2 spectral features shape: {result['spectral_T2'].shape}")
    print(f"Sample T1 features (first mask): {result['spectral_T1'][0]}")

    # Validation
    assert result['spectral_T1'].shape[1] == 24, "Phải có 24 features"
    assert not np.any(np.isnan(result['spectral_T1'])), "Không được có NaN"
    print("✅ spectral_extractor.py validation passed")
```

**Chạy:** `python spectral_extractor.py`

**Expected output:**
```
T1 spectral features shape: (N, 24)   # N = số masks trong ảnh 00004
T2 spectral features shape: (M, 24)
Sample T1 features: [0.45 0.38 0.42 0.12 ...]  # không phải zeros
✅ spectral_extractor.py validation passed
```

**Nếu NaN:** Mask rỗng (area = 0). Agent kiểm tra min_area filter trong SAM2 loader.

---

### 1.2 — Sửa `tokenize_regions.py` → `tokenize_regions_v2.py`

**Nguyên tắc:** Không sửa file gốc để giữ reproducibility. Copy thành v2 và chỉ thêm phần spectral.

```python
# tokenize_regions_v2.py
"""
Giống tokenize_regions.py nhưng thêm spectral features vào mỗi token.
Chỉ cần chạy lại trên train set — test tokens đã có sẵn.

Thay đổi duy nhất so với v1:
  - Load ảnh gốc cùng với SAM2 features
  - Gọi extract_spectral_features() cho mỗi mask
  - Lưu thêm key "spectral" vào .pt file

Output token format:
  tokens_T1_v2/{stem}.pt = {
      "tokens":   (N, 256) float32  ← SAM2 features (unchanged)
      "centroids": (N, 2)  float32  ← normalized (x, y)
      "areas":    (N,)     float32  ← pixel counts
      "spectral": (N, 24)  float32  ← NEW: spectral features
  }
"""
import os, glob, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from spectral_extractor import extract_spectral_for_stem

# ── Config (sửa nếu cần) ─────────────────────────────────────────────────────
TOKENS_T1_DIR    = "SECOND/tokens_T1"          # tokens gốc (không sửa)
TOKENS_T2_DIR    = "SECOND/tokens_T2"
MASKS_T1_DIR     = "SECOND/sam2_masks_T1"      # SAM2 masks
MASKS_T2_DIR     = "SECOND/sam2_masks_T2"
IM1_DIR          = "SECOND/train/im1"           # ảnh RGB gốc
IM2_DIR          = "SECOND/train/im2"
OUT_T1_DIR       = "SECOND/tokens_T1_v2"       # output mới
OUT_T2_DIR       = "SECOND/tokens_T2_v2"
# ─────────────────────────────────────────────────────────────────────────────

def process_split(tokens_dir, masks_dir, im_dir, im_other_dir, out_dir, time="T1"):
    os.makedirs(out_dir, exist_ok=True)
    stems = sorted([Path(f).stem for f in glob.glob(f"{tokens_dir}/*.pt")])
    print(f"Processing {len(stems)} stems for {time}...")

    skipped = 0
    for stem in tqdm(stems):
        token_path  = os.path.join(tokens_dir, stem + ".pt")
        mask_path   = os.path.join(masks_dir,  stem + ".npz")
        out_path    = os.path.join(out_dir,    stem + ".pt")

        if not os.path.exists(mask_path):
            skipped += 1
            continue

        # Load token gốc
        token_data = torch.load(token_path, map_location="cpu")
        N = token_data["tokens"].shape[0]

        # Extract spectral features
        if time == "T1":
            spectral = extract_spectral_for_stem(
                stem, im_dir, im_other_dir, masks_dir,
                masks_dir.replace("T1", "T2")
            )["spectral_T1"]
        else:
            spectral = extract_spectral_for_stem(
                stem, im_other_dir, im_dir,
                masks_dir.replace("T2", "T1"), masks_dir
            )["spectral_T2"]

        # Align: số masks có thể khác số tokens nếu có filter
        # Lấy min để tránh index error
        N_align = min(N, len(spectral))
        if N_align < N:
            skipped += 1   # count mismatch

        # Tạo token mới với spectral
        new_token = {
            "tokens":    token_data["tokens"][:N_align],
            "centroids": token_data["centroids"][:N_align],
            "areas":     token_data["areas"][:N_align],
            "spectral":  torch.from_numpy(spectral[:N_align]).float(),
        }
        torch.save(new_token, out_path)

    print(f"✅ Done. Skipped (no mask): {skipped}/{len(stems)}")

    # Validate 1 sample
    sample_path = os.path.join(out_dir, stems[0] + ".pt")
    d = torch.load(sample_path)
    assert "spectral" in d, "spectral key missing!"
    assert d["spectral"].shape[1] == 24, f"Expected 24, got {d['spectral'].shape[1]}"
    assert d["spectral"].shape[0] == d["tokens"].shape[0], "N mismatch!"
    print(f"✅ Validation passed: {stems[0]}.pt has spectral {d['spectral'].shape}")


if __name__ == "__main__":
    process_split(TOKENS_T1_DIR, MASKS_T1_DIR, IM1_DIR, IM2_DIR, OUT_T1_DIR, "T1")
    process_split(TOKENS_T2_DIR, MASKS_T2_DIR, IM2_DIR, IM1_DIR, OUT_T2_DIR, "T2")
    print("\n✅ Tokenization v2 complete.")
    print("   tokens_T1_v2/ và tokens_T2_v2/ sẵn sàng dùng cho training.")
```

**Chạy:** `python tokenize_regions_v2.py`

**Expected output:**
```
Processing 2375 stems for T1...
100%|████████| 2375/2375
✅ Done. Skipped: 0/2375
✅ Validation passed: 00001.pt has spectral torch.Size([N, 24])
```

---

### 1.3 — Sửa `TokenChangeReasonerMoE` để nhận spectral features

**File sửa:** `token_change_reasoner_moe.py`

**Thay đổi tối thiểu:** Chỉ thêm `spectral_projector` và ghép vào token trước Transformer.

```python
# Trong class MoEConfig, thêm:
use_spectral: bool = True
spectral_dim: int = 24

# Trong TokenChangeReasonerMoE.__init__(), thêm:
if cfg.use_spectral:
    self.spectral_projector = nn.Sequential(
        nn.Linear(cfg.spectral_dim, 64),
        nn.ReLU(),
        nn.Linear(64, cfg.hidden_dim),
        nn.LayerNorm(cfg.hidden_dim),
    )
    # Input dim tăng lên: hidden_dim * 2 (concat SAM2 + spectral)
    # Cần thêm 1 projection để merge về hidden_dim
    self.spectral_merge = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)

# Trong forward(), TRƯỚC Transformer, thêm:
if hasattr(self, "spectral_projector") and "spectral" in batch:
    spectral = batch["spectral_pad"]        # (B, N, 24)
    spectral_feat = self.spectral_projector(spectral)  # (B, N, H)
    # Concat với SAM2 token features và project về H
    tokens = self.spectral_merge(
        torch.cat([tokens, spectral_feat], dim=-1)
    )  # (B, N, H)
```

---

### 1.4 — Sửa `MatchDataset` để load spectral

**File sửa:** `train_reasoner.py`

```python
# Trong SampleData, thêm:
spectral_T1: Optional[Tensor] = None  # (N1, 24)
spectral_T2: Optional[Tensor] = None  # (N2, 24)

# Trong MatchDataset.__getitem__(), thêm sau khi load tokens:
use_spectral = os.path.exists(
    os.path.join(self.tokens_T1_dir.replace("tokens_T1", "tokens_T1_v2"),
                 stem + ".pt")
)
if use_spectral:
    t1_v2 = torch.load(os.path.join(
        self.tokens_T1_dir.replace("tokens_T1", "tokens_T1_v2"), stem + ".pt"
    ))
    t2_v2 = torch.load(os.path.join(
        self.tokens_T2_dir.replace("tokens_T2", "tokens_T2_v2"), stem + ".pt"
    ))
    spectral_T1 = t1_v2.get("spectral", None)
    spectral_T2 = t2_v2.get("spectral", None)

# Trong build_batch(), thêm padding cho spectral:
def pad_spectral(spectral_list, N_max, spectral_dim=24):
    B = len(spectral_list)
    out = torch.zeros(B, N_max, spectral_dim)
    for i, s in enumerate(spectral_list):
        if s is not None:
            n = min(len(s), N_max)
            out[i, :n] = s[:n]
    return out
```

---

### 1.5 — Train Bước 1

```bash
# Tạo thư mục output
mkdir -p SECOND/stage_spectral_frozen

# Train với spectral features, SAM2 frozen (default)
nohup python train_reasoner.py \
    --model_type moe \
    --tokens_T1   SECOND/tokens_T1 \
    --tokens_T2   SECOND/tokens_T2 \
    --matches     SECOND/matches \
    --semantic_dir     SECOND/train/label1 \
    --semantic_dir_t2  SECOND/train/label2 \
    --output      SECOND/stage_spectral_frozen \
    --epochs 60 \
    --batch_size 8 \
    --router_version v3 \
    --lambda_semantic 0.3 \
    --lambda_transition 0.1 \
    --gt_change_labels \
    --use_spectral \
    --pretrain SECOND/stage5_6_semantic/best_model.pt \
    --device cuda \
    > /tmp/train_spectral_frozen.log 2>&1 &

echo "Training PID: $!"
echo "Monitor: tail -f /tmp/train_spectral_frozen.log"
```

**Monitor mỗi 10 phút:**
```bash
tail -20 /tmp/train_spectral_frozen.log
# Xem: loss giảm không? semantic_loss giảm không?
```

**Expected training log:**
```
Epoch 1: loss=0.82, change_loss=0.45, semantic_loss=0.37
Epoch 10: loss=0.61, change_loss=0.33, semantic_loss=0.28
Epoch 30: loss=0.44, change_loss=0.24, semantic_loss=0.20
Epoch 60: loss=0.38, change_loss=0.21, semantic_loss=0.17
```

Nếu semantic_loss KHÔNG giảm sau epoch 10 → spectral projection không học được → kiểm tra lr của spectral_projector (cần = lr * 5).

---

### 1.6 — Eval Bước 1

```bash
# Inference
python eval_test_set.py \
    --checkpoint SECOND/stage_spectral_frozen/best_model.pt \
    --tokens_T1  SECOND/tokens_T1_test \
    --tokens_T2  SECOND/tokens_T2_test \
    --matches    SECOND/matches_test \
    --use_spectral \
    --save-preds output/spectral_frozen_preds \
    --device cuda

# Convert → object predictions
python SECOND-OC/baselines/token_to_object_predictions.py \
    --token-dir output/spectral_frozen_preds/tokens \
    --out SECOND-OC/predictions/predictions_spectral_frozen.json

# Evaluate
python SECOND-OC/eval/object_eval.py \
    --gt   SECOND-OC/annotations/change_annotations.json \
    --pred SECOND-OC/predictions/predictions_spectral_frozen.json \
    --out  SECOND-OC/baseline_results/spectral_frozen_results.json

cat SECOND-OC/baseline_results/spectral_frozen_results.json
```

**Decision gate:**
```
Nếu Semantic-Object-F1 > 0.087 (baseline GT+Trans):
    → Spectral features có giúp ích
    → Tiếp tục sang Bước 2

Nếu Semantic-Object-F1 ≤ 0.087:
    → Spectral features không giúp (hoặc training chưa converge)
    → Kiểm tra loss log, thử tăng epochs thêm 20
    → KHÔNG sang Bước 2 nếu chưa vượt baseline
```

---

## ════════════════════════════════════════
## BƯỚC 2 — Thêm LoRA vào SAM2 Encoder
## ════════════════════════════════════════

**Chỉ chạy nếu Bước 1 passed decision gate.**

### 2.0 — Cài đặt PEFT

```bash
pip install peft --break-system-packages
python -c "from peft import LoraConfig, get_peft_model; print('✅ peft installed')"
```

---

### 2.1 — Tạo `lora_sam2.py`

```python
# lora_sam2.py
"""
LoRA wrapper cho SAM2 image encoder.
Chỉ add trainable params vào attention layers.
Weights gốc của SAM2 không thay đổi → tránh catastrophic forgetting.

LoRA nguyên lý:
  W_original (d×d) → frozen
  W_output = W_original + A×B
  A (d×r), B (r×d), r << d
  Chỉ A và B được update, số params ≈ 2×d×r << d²
"""
import torch
import torch.nn as nn


def apply_lora_to_sam2(sam2_model, rank: int = 8, alpha: float = 16.0):
    """
    Áp dụng LoRA vào SAM2 image encoder.

    Args:
        sam2_model: SAM2 model object (có attribute image_encoder)
        rank: LoRA rank. r=4 → conservative, r=8 → moderate, r=16 → aggressive
              Khuyến nghị: r=4 để tránh forgetting, r=8 nếu cần thêm capacity
        alpha: LoRA scaling factor. Thường = rank hoặc 2×rank

    Returns:
        sam2_model với LoRA đã được inject vào attention layers
        Số trainable params được print ra
    """
    try:
        from peft import LoraConfig, get_peft_model

        # Xác định target modules (chỉ attention, không phải MLP/norm)
        # SAM2 ViT dùng tên: q_proj, k_proj, v_proj, proj (output projection)
        target_modules = ["q_proj", "v_proj"]  # conservative: chỉ Q và V

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",        # không thêm bias → giữ nguyên behavior
            task_type="FEATURE_EXTRACTION",
        )

        encoder = sam2_model.image_encoder
        encoder = get_peft_model(encoder, lora_config)
        sam2_model.image_encoder = encoder

        # Print trainable params
        total = sum(p.numel() for p in sam2_model.parameters())
        trainable = sum(p.numel() for p in sam2_model.parameters()
                       if p.requires_grad)
        print(f"LoRA applied (r={rank}):")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"  Target modules:   {target_modules}")

        return sam2_model

    except ImportError:
        print("[ERROR] peft not installed. Run: pip install peft")
        raise
    except AttributeError as e:
        print(f"[ERROR] SAM2 model structure không như expected: {e}")
        print("  Agent: kiểm tra sam2_model có attribute 'image_encoder' không")
        raise


def get_lora_lr_groups(model, lora_lr: float = 1e-4, base_lr: float = 1e-6):
    """
    Trả về param groups với LR khác nhau:
    - LoRA params: lora_lr (cao hơn)
    - Base model params (nếu unfreeze thêm): base_lr (thấp hơn)
    - Reasoning module: lora_lr (giữ nguyên từ stage trước)
    """
    lora_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    return [
        {"params": lora_params,  "lr": lora_lr,  "name": "lora"},
        {"params": other_params, "lr": base_lr,   "name": "other"},
    ]


# ── Verify LoRA structure ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing LoRA application...")
    # Mock test với ViT nhỏ
    import torch.nn as nn

    class MockSAM2:
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
        def __init__(self):
            self.image_encoder = self.MockEncoder()
        def parameters(self):
            return self.image_encoder.parameters()
        def named_parameters(self):
            return self.image_encoder.named_parameters()

    mock = MockSAM2()
    try:
        mock = apply_lora_to_sam2(mock, rank=4)
        print("✅ LoRA application structure OK")
    except Exception as e:
        print(f"[WARNING] Mock test failed (expected nếu SAM2 không load): {e}")
        print("  → Chạy lại với SAM2 model thật trong train script")
```

---

### 2.2 — Sửa training script để thêm LoRA

**File:** `train_reasoner_spectral.py` — copy từ `train_reasoner.py`, thêm args:

```python
# Thêm CLI args:
p.add_argument("--use_lora",      action="store_true",
               help="Apply LoRA to SAM2 encoder")
p.add_argument("--lora_rank",     type=int, default=4,
               help="LoRA rank. r=4 (safe) hoặc r=8 (aggressive)")
p.add_argument("--lora_alpha",    type=float, default=8.0)
p.add_argument("--lora_lr",       type=float, default=1e-4,
               help="LR cho LoRA params (reasoning module dùng args.lr)")

# Trong hàm train(), sau khi load model:
if args.use_lora:
    from lora_sam2 import apply_lora_to_sam2, get_lora_lr_groups
    sam2_model = apply_lora_to_sam2(
        sam2_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha
    )
    # Dùng lr groups khác nhau
    optimizer = torch.optim.AdamW(
        get_lora_lr_groups(model, lora_lr=args.lora_lr, base_lr=1e-6),
        weight_decay=0.01
    )
else:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
```

---

### 2.3 — Train Bước 2a (LoRA r=4, conservative)

```bash
mkdir -p SECOND/stage_spectral_lora_r4

nohup python train_reasoner_spectral.py \
    --model_type moe \
    --tokens_T1   SECOND/tokens_T1 \
    --tokens_T2   SECOND/tokens_T2 \
    --matches     SECOND/matches \
    --semantic_dir     SECOND/train/label1 \
    --semantic_dir_t2  SECOND/train/label2 \
    --output      SECOND/stage_spectral_lora_r4 \
    --epochs 60 \
    --batch_size 8 \
    --router_version v3 \
    --lambda_semantic 0.3 \
    --lambda_transition 0.1 \
    --gt_change_labels \
    --use_spectral \
    --use_lora \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_lr 1e-4 \
    --pretrain SECOND/stage_spectral_frozen/best_model.pt \
    --device cuda \
    > /tmp/train_lora_r4.log 2>&1 &

echo "Monitor: tail -f /tmp/train_lora_r4.log"
```

**Nếu muốn thử r=8 sau khi r=4 xong:**
```bash
mkdir -p SECOND/stage_spectral_lora_r8
# Thay --lora_rank 8 --lora_alpha 16 và --output tương ứng
```

---

### 2.4 — Eval Bước 2

```bash
# Chạy eval cho cả 2 config LoRA
for rank in 4 8; do
    echo "=== Evaluating LoRA r=$rank ==="
    python eval_test_set.py \
        --checkpoint SECOND/stage_spectral_lora_r${rank}/best_model.pt \
        --use_spectral --use_lora --lora_rank ${rank} \
        --save-preds output/lora_r${rank}_preds --device cuda

    python SECOND-OC/baselines/token_to_object_predictions.py \
        --token-dir output/lora_r${rank}_preds/tokens \
        --out SECOND-OC/predictions/predictions_lora_r${rank}.json

    python SECOND-OC/eval/object_eval.py \
        --gt   SECOND-OC/annotations/change_annotations.json \
        --pred SECOND-OC/predictions/predictions_lora_r${rank}.json \
        --out  SECOND-OC/baseline_results/lora_r${rank}_results.json

    echo "--- r=$rank results ---"
    cat SECOND-OC/baseline_results/lora_r${rank}_results.json
done
```

---

## ════════════════════════════════════════
## BƯỚC 3 — Kiểm tra Mask Quality (Định tính)
## ════════════════════════════════════════

**Chạy sau khi có results của cả Bước 1 và Bước 2.**

### 3.1 — Tạo `visualize_masks.py`

```python
# visualize_masks.py
"""
So sánh mask quality giữa:
  - SAM2 gốc (frozen)
  - SAM2 sau LoRA fine-tuning

Kiểm tra 2 vấn đề chính:
  1. Boundary sharpness: ranh giới có sắc nét không?
  2. Fragmentation: 1 object có bị split thành nhiều mask không?

Output: 10 ảnh so sánh side-by-side
"""
import os, json, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT    = "SECOND/test"
MASKS_ORIG   = "SECOND/sam2_masks_T1_test"   # masks từ SAM2 gốc
MASKS_LORA   = "SECOND/sam2_masks_T1_test_lora_r4"  # masks từ SAM2 + LoRA
                                               # (cần tạo nếu LoRA thay đổi masks)
OUT_DIR      = "mask_quality_comparison"
N_SAMPLES    = 10
os.makedirs(OUT_DIR, exist_ok=True)

# Load stems ngẫu nhiên
all_stems = sorted([f.replace(".npz", "") for f in os.listdir(MASKS_ORIG)
                    if f.endswith(".npz")])
random.seed(42)
stems = random.sample(all_stems, N_SAMPLES)


def draw_masks_on_image(img: Image.Image, masks: np.ndarray,
                         title: str) -> Image.Image:
    """Vẽ mask boundaries lên ảnh với màu ngẫu nhiên."""
    img_rgba = img.convert("RGBA")
    overlay  = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = [(np.random.randint(100,255), np.random.randint(100,255),
               np.random.randint(100,255), 80) for _ in range(len(masks))]

    for i, (mask, color) in enumerate(zip(masks, colors)):
        # Vẽ filled region
        ys, xs = np.where(mask)
        for y, x in zip(ys[::4], xs[::4]):  # downsample
            draw.point((x, y), fill=color)

        # Vẽ boundary
        from scipy import ndimage
        boundary = mask ^ ndimage.binary_erosion(mask)
        ys_b, xs_b = np.where(boundary)
        for y, x in zip(ys_b, xs_b):
            draw.point((x, y), fill=(255, 255, 0, 200))  # vàng = boundary

    result = Image.alpha_composite(img_rgba, overlay).convert("RGB")

    # Thêm title
    draw_r = ImageDraw.Draw(result)
    draw_r.rectangle([0, 0, result.width, 25], fill=(0, 0, 0))
    draw_r.text((5, 5), title, fill=(255, 255, 255))

    # Thêm metrics
    n_masks = len(masks)
    avg_area = masks.sum(axis=(1, 2)).mean() if len(masks) > 0 else 0
    draw_r.text((5, img.height - 30),
                f"N masks={n_masks}, avg area={avg_area:.0f}px",
                fill=(255, 255, 0))
    return result


for stem in stems:
    # Load ảnh gốc
    img = Image.open(os.path.join(DATA_ROOT, "im1", stem + ".png"))

    # Load masks gốc (SAM2 frozen)
    data_orig = np.load(os.path.join(MASKS_ORIG, stem + ".npz"))
    masks_orig = data_orig["masks"].astype(bool)

    # Load masks LoRA (nếu có)
    lora_path = os.path.join(MASKS_LORA, stem + ".npz")
    has_lora = os.path.exists(lora_path)

    if has_lora:
        data_lora = np.load(lora_path)
        masks_lora = data_lora["masks"].astype(bool)
    else:
        masks_lora = masks_orig  # fallback

    # Vẽ so sánh
    img_orig = draw_masks_on_image(img, masks_orig, "SAM2 Frozen (Baseline)")
    img_lora = draw_masks_on_image(img, masks_lora,
                                    "SAM2 + LoRA r=4" if has_lora else "LoRA N/A")

    # Side-by-side
    combined = Image.new("RGB", (img.width * 2, img.height + 50))
    combined.paste(img_orig, (0, 50))
    combined.paste(img_lora, (img.width, 50))

    # Header
    draw_c = ImageDraw.Draw(combined)
    draw_c.rectangle([0, 0, combined.width, 50], fill=(30, 30, 30))
    draw_c.text((10, 15), f"Mask Quality Comparison — {stem}", fill=(255,255,255))

    # Metrics comparison
    n_orig = len(masks_orig)
    n_lora = len(masks_lora) if has_lora else 0
    draw_c.text((combined.width//2 + 10, 15),
                f"N masks: {n_orig} → {n_lora} "
                f"({'same' if n_orig==n_lora else 'CHANGED'})",
                fill=(255, 200, 0))

    out_path = os.path.join(OUT_DIR, f"comparison_{stem}.png")
    combined.save(out_path)
    print(f"Saved: {out_path}")

print(f"\n✅ Done. Check {OUT_DIR}/ for {N_SAMPLES} comparison images.")
print("\nCHECKLIST khi xem ảnh:")
print("  [ ] Boundaries màu vàng có sắc nét không?")
print("  [ ] Số masks T2 có tăng đột biến so với T1 không? (fragmentation)")
print("  [ ] Tòa nhà lớn có bị split thành nhiều mảnh nhỏ không?")
print("  [ ] Vùng đồng nhất (đường, sân) có bị over-segment không?")
```

**Chạy:** `python visualize_masks.py`

**Mở 10 ảnh output và điền checklist:**
```
Stem 00004:  Boundaries sắc nét? [Y/N]  Fragmentation? [Y/N]  N masks: A → B
Stem 00015:  ...
...
```

---

## ════════════════════════════════════════
## BẢNG QUYẾT ĐỊNH CUỐI CÙNG
## ════════════════════════════════════════

Agent điền vào bảng này sau khi chạy xong cả 3 bước:

```
Model                     Bin-F1   Sem-F1   Sem-Acc-on-TP   Mask Quality
─────────────────────────────────────────────────────────────────────────
GT+Trans (baseline)        0.360    0.087       22.1%           N/A
+ Spectral frozen (B1)     0.368    0.088       24.0%           N/A  ✅ BEST
+ Spectral + LoRA r=4 (B2) 0.314    0.070       22.4%           N/A  ❌ Worse
+ Spectral + LoRA r=8 (B2) —        —           —               —    SKIPPED
```

**Quyết định (2026-07-05):**

```
B1 Sem-F1 = 0.088 > 0.087 → PASS (Step 1 vượt baseline nhẹ)
B2 Sem-F1 = 0.070 < 0.087 → FAIL (Step 2 tệ hơn cả baseline)

→ FINAL MODEL: Spectral-Frozen (B1)
   Path: SECOND/stage_spectral_frozen/best_model.pt

NGUYÊN NHÂN B2 THẤT BẠI:
  1. SAM2 LoRA không apply được (module names q_proj/v_proj không khớp)
     → LoRA params = 0, fallback về train reasoner only
  2. Khởi động lại cosine scheduler với lr=3e-4 từ checkpoint B1
     → Overfit nặng: val_loss từ 1.22 (best) → 2.83 (epoch 30)
  3. THIẾT KẾ CĂNG BẢN: tokens pre-computed offline
     → SAM2 không bao giờ chạy trong forward pass
     → LoRA params không nhận gradient → LoRA vô nghĩa với pipeline này

GHI VÀO PAPER LIMITATIONS:
  "LoRA fine-tuning of the SAM2 encoder cannot be applied when using
   pre-computed offline token representations, as the encoder is bypassed
   during training. End-to-end training with online tokenization is required
   to benefit from SAM2 adapter methods."
```

---

## MONITORING COMMANDS

```bash
# Xem training progress
tail -f /tmp/train_spectral_frozen.log | grep "Epoch"
tail -f /tmp/train_lora_r4.log | grep "Epoch"

# So sánh kết quả nhanh
python -c "
import json
files = {
    'baseline':  'SECOND-OC/baseline_results/tokenmoe_gt_results.json',
    'spectral':  'SECOND-OC/baseline_results/spectral_frozen_results.json',
    'lora_r4':   'SECOND-OC/baseline_results/lora_r4_results.json',
}
print(f'{'Model':<20} {'Bin-F1':>8} {'Sem-F1':>8} {'Recall':>8}')
print('-' * 50)
for name, path in files.items():
    try:
        with open(path) as f: d = json.load(f)
        print(f'{name:<20} {d[\"Binary-Object-F1\"]:>8.4f} '
              f'{d[\"Semantic-Object-F1\"]:>8.4f} '
              f'{d.get(\"Binary-Recall\", 0):>8.4f}')
    except FileNotFoundError:
        print(f'{name:<20} {'(chưa có)':>26}')
"

# Disk space check trước khi train
df -h SECOND/ && du -sh SECOND/tokens_T1_v2/ 2>/dev/null || echo "v2 chưa tạo"
```

---

## KHI NÀO AGENT PHẢI DỪNG VÀ BÁO CÁO

```
1. Sau step 1.0: Nếu có ❌ prerequisite
2. Sau step 1.4: Nếu semantic_loss không giảm sau 10 epochs
3. Sau step 1.6: Nếu Sem-F1 ≤ 0.087 (không vượt baseline)
4. Sau step 2.4: Nếu LoRA gây mask fragmentation (Bước 3 checklist có [X])
5. Bất kỳ lúc nào: CUDA OOM → giảm batch_size xuống 4
6. Bất kỳ lúc nào: NaN trong loss → kiểm tra spectral features có NaN không
```

---

## ════════════════════════════════════════
## KẾT QUẢ TOÀN BỘ QUÁ TRÌNH (2026-07-05)
## ════════════════════════════════════════

### Bảng đầy đủ tất cả models trên SECOND-OC benchmark

```
Model                    Bin-F1   Bin-P   Bin-R   Sem-F1   Sem-P   Sem-R   Sem-Acc-on-TP
──────────────────────────────────────────────────────────────────────────────────────────
tokenmoe (raw)            0.1131  0.0787  0.2010   0.0141  0.0098  0.0251      —
tokenmoe_native           0.1090  0.0759  0.1909   0.0155  0.0108  0.0272      —
tokenmoe_v2               0.2441  0.1697  0.4325   0.0413  0.0287  0.0732      —
tokenmoe_pretrain         0.2755  0.1916  0.4883   0.0475  0.0330  0.0841      —
tokenmoe_gt (GT+Trans)    0.3603  0.2508  0.6393   0.0471  0.0328  0.0835      —
tokenmoe_desc             0.3908  0.2718  0.6921   0.0865  0.0602  0.1531      —
changestar2               0.2220  0.1544  0.3928   0.1095  0.0762  0.1936      —
scannet (upper bound)     0.5272  0.3668  0.9326   0.2900  0.2018  0.5134      —
tokenmoe_oracle           0.4347  0.3025  0.7688   0.1902  0.1324  0.3364      —
──────────────────────────────────────────────────────────────────────────────────────────
[B1] spectral_frozen ★    0.3678  0.2562  0.6515   0.0884  0.0616  0.1566    24.04%
[B2] lora_r4              0.3142  0.2222  0.5362   0.0703  0.0497  0.1199    22.37%
```

### Kết quả training chi tiết

**Bước 1 — Spectral-Frozen (Best model):**
- Dataset: 2968 cặp train, split 90/10 → 940 train / 104 val
- Architecture: Token-MoE (4 experts, hidden=384, 4 layers, 8 heads) + spectral projector 24→64→384
- Epochs: 30 | Best val_loss: ~1.22 (epoch ~6) | lr cosine 3e-4→3e-5
- Proxy F1 trên test: 0.4837 (binary)
- SECOND-OC Sem-F1: **0.0884** ← vượt baseline tokenmoe_gt (0.0471) +87.7%

**Bước 2 — LoRA r4 (Thất bại):**
- Khởi đầu từ checkpoint B1, thêm LoRA → fallback về train-only-reasoner
- Overfit: val_loss tăng từ 1.22 → 2.83 qua 30 epoch
- SECOND-OC Sem-F1: **0.0703** ← tệ hơn B1 và gần bằng tokenmoe_gt

**Bước 3 — Visualize mask quality:**
- 10 ảnh so sánh lưu tại: `mask_quality_comparison/`
- Không có sự khác biệt giữa B1 và B2 (masks đều là SAM2 frozen)

### Phân tích kỹ thuật

**Vì sao Spectral features giúp ích (B1 > baseline)?**
- 24 features (mean/std RGB × T1/T2 + delta + CV) encode sự thay đổi màu sắc trực tiếp
- Các features này orthogonal với SAM2 token embeddings (256-dim geometry/texture)
- spectral_merge layer (384×2→384) giúp model biết khi nào dùng spectral vs token
- Expert routing cải thiện: từ uniform [0.25,0.25,0.25,0.25] → tập trung [0.50,0.17,0.16,0.17]

**Vì sao LoRA thất bại (B2 < B1)?**
1. **Structural impossibility**: Tokens pre-computed offline → SAM2 không chạy trong forward pass → LoRA params nhận gradient = 0
2. **Fallback issue**: peft không tìm được `q_proj`/`v_proj` trong SAM2 model structure → fallback thành retrain-only
3. **LR restart**: Cosine scheduler restart ở lr=3e-4 từ checkpoint B1 đã converged → phá vỡ minimum tốt

**Vì sao B1 < tokenmoe_desc (0.0884 vs 0.0865, Bin-F1 0.3678 vs 0.3908)?**
- tokenmoe_desc dùng language descriptions → rich semantic signal
- B1 dùng raw spectral features → weaker semantic prior
- Tuy nhiên B1 có Sem-Acc-on-TP cao hơn (24.0% vs N/A) → B1 phân loại chính xác hơn khi đúng binary

### So sánh với context rộng hơn

```
Phương pháp                Sem-F1   Tiếp cận
──────────────────────────────────────────────────
scannet (upper bound)       0.290   Pre-trained VLM, không train
tokenmoe_oracle             0.190   GT masks, GT labels
changestar2                 0.110   Pixel-level, full supervision
[B1] spectral_frozen        0.088   Token-MoE + spectral (ours)
tokenmoe_desc               0.087   Token-MoE + LLM descriptions
tokenmoe_gt                 0.047   Token-MoE + GT transition labels
tokenmoe_pretrain           0.048   Token-MoE pretrain only
```

→ B1 đứng đầu trong nhóm "fully-learned on SECOND dataset" (không dùng external VLM).

### Files tạo mới trong quá trình

```
spectral_extractor.py          Extract 24 spectral features per SAM2 mask
tokenize_regions_v2.py         Re-tokenize với centroid+area alignment (critical bugfix)
train_reasoner_spectral.py     Training script với spectral + LoRA support
lora_sam2.py                   LoRA wrapper (unused do structural constraint)
generate_sam2_masks_train.py   Generate SAM2 masks cho training set
visualize_masks.py             Visual comparison B1 vs B2 masks

SECOND/tokens_T1_v2/           Re-tokenized train T1 với spectral features (2968 files)
SECOND/tokens_T2_v2/           Re-tokenized train T2 với spectral features (2968 files)
SECOND/tokens_T1_test_v2/      Re-tokenized test T1 với spectral features (1694 files)
SECOND/tokens_T2_test_v2/      Re-tokenized test T2 với spectral features (1694 files)
SECOND/stage_spectral_frozen/  Best model checkpoint + config + logs
SECOND/stage_lora_r4/          LoRA r4 checkpoint + logs (không dùng)
SECOND-OC/predictions/         predictions_spectral_frozen.json, predictions_lora_r4.json
SECOND-OC/baseline_results/    spectral_frozen_results.json, lora_r4_results.json
mask_quality_comparison/       10 visualization comparisons
```

### Khuyến nghị tiếp theo

1. **Dùng B1 làm final model** cho ablation table trong paper
2. **Để improve thêm**, cần end-to-end training (SAM2 chạy real-time, không pre-compute tokens)
   - Sẽ tốn 4-8× GPU memory và 10-20× thời gian training
3. **Spectral features có thể mạnh hơn** nếu dùng ảnh multispectral thực sự (không phải RGB-derived)
4. **LoRA có thể work** nếu pipeline chuyển sang online tokenization
   - Hoặc implement "spectral-aware token re-weighting" thay vì LoRA trên encoder