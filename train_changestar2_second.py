"""
train_changestar2_second.py
===========================
Train ChangeStar2 (torchange 0.0.4) trên SECOND dataset.

Dataset layout:
    SECOND/im1/        SECOND/im2/        (train images)
    SECOND/label1/     SECOND/label2/     (train labels – RGB)
    SECOND/test/im1/   SECOND/test/im2/   (test images)
    SECOND/test/label1/ SECOND/test/label2/

Output: checkpoints/changestar2_second_best.pth

Run with:
    conda run -n changestar2 python train_changestar2_second.py
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
SECOND_ROOT  = "SECOND"
BATCH_SIZE   = 8
LR           = 6e-5
EPOCHS       = 50
IMG_SIZE     = 256      # random crop from 512×512
DEVICE       = "cuda"
NUM_CLASSES  = 7        # SECOND: 7 semantic classes (0=background treated as ignore)
SAVE_PATH    = "checkpoints/changestar2_second_best.pth"
LOG_INTERVAL = 20       # batches

os.makedirs("checkpoints", exist_ok=True)

# ── SECOND RGB → class index ──────────────────────────────────────────────────
# Labels are RGB PNGs with 7 colors:
PALETTE = [
    ((0,   0,   0),   0),   # background / unlabeled  → ignore_index
    ((0,   128, 0),   1),   # tree
    ((128, 0,   0),   2),   # buildings
    ((0,   0,   255), 3),   # water
    ((128, 128, 128), 4),   # non_veg_ground (impervious surface)
    ((255, 255, 255), 5),   # playground
    ((0,   255, 0),   6),   # low_vegetation
]
_PAL_RGB = np.array([p[0] for p in PALETTE], dtype=np.float32)
_PAL_CLS = np.array([p[1] for p in PALETTE], dtype=np.int64)
IGNORE_INDEX = 0   # background → ignored in CE loss

def rgb_to_label(rgb_pil: Image.Image) -> np.ndarray:
    arr = np.array(rgb_pil, dtype=np.uint8)
    H, W = arr.shape[:2]
    flat = arr.reshape(-1, 3).astype(np.float32)
    # L2 distance to each palette entry (fast nearest colour)
    dists = ((flat[:, None, :] - _PAL_RGB[None, :, :]) ** 2).sum(-1)
    label = _PAL_CLS[dists.argmin(-1)]
    return label.reshape(H, W)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SECONDDataset(Dataset):
    def __init__(self, root: str, stems_file: str,
                 im1_dir: str, im2_dir: str,
                 lb1_dir: str, lb2_dir: str,
                 size: int = 256, augment: bool = True):
        self.im1_dir  = im1_dir
        self.im2_dir  = im2_dir
        self.lb1_dir  = lb1_dir
        self.lb2_dir  = lb2_dir
        self.size     = size
        self.augment  = augment

        with open(stems_file) as f:
            self.stems = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        im1 = Image.open(os.path.join(self.im1_dir, f"{stem}.png")).convert("RGB")
        im2 = Image.open(os.path.join(self.im2_dir, f"{stem}.png")).convert("RGB")
        lb1 = Image.open(os.path.join(self.lb1_dir, f"{stem}.png")).convert("RGB")
        lb2 = Image.open(os.path.join(self.lb2_dir, f"{stem}.png")).convert("RGB")

        W, H = im1.size

        if self.augment:
            # Random crop
            x = np.random.randint(0, max(1, W - self.size))
            y = np.random.randint(0, max(1, H - self.size))
            # Random horizontal flip
            hflip = np.random.rand() < 0.5
        else:
            x = (W - self.size) // 2
            y = (H - self.size) // 2
            hflip = False

        def crop_flip(img):
            img = img.crop((x, y, x + self.size, y + self.size))
            if hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return img

        im1, im2 = crop_flip(im1), crop_flip(im2)
        lb1, lb2 = crop_flip(lb1), crop_flip(lb2)

        def to_tensor(img: Image.Image) -> torch.Tensor:
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr.transpose(2, 0, 1))  # C,H,W

        lb1_t = torch.from_numpy(rgb_to_label(lb1)).long()
        lb2_t = torch.from_numpy(rgb_to_label(lb2)).long()

        return {
            "im1":    to_tensor(im1),
            "im2":    to_tensor(im2),
            "label1": lb1_t,
            "label2": lb2_t,
            "stem":   stem,
        }


# ── ChangeStar2 model via torchange config API ─────────────────────────────────
from torchange.models.changestar2 import ChangeStar2, Segmentation, get_detector, TargetGenerator, ChangeMixin2
import ever as er
import ever.module as M

class ChangeStar2SCD(ChangeStar2):
    def __init__(self, config):
        # Call ERModule.__init__ directly to bypass ChangeStar2.__init__
        er.ERModule.__init__(self, config)
        
        # Clean up the merged config to prevent TypeError inside AssymetricDecoder.__init__
        if self.config.segmentation.model_type == 'semantic_fpn':
            for key in ['fpn', 'fs_relation', 'fpn_decoder']:
                if key in self.config.segmentation.head:
                    del self.config.segmentation.head[key]
        
        segmentation = Segmentation(self.config.segmentation)
        classifier = M.ConvUpsampling(
            self.config.semantic_classifier.in_channels,
            self.config.semantic_classifier.out_channels,
            self.config.semantic_classifier.scale,
            3, 1, 1
        )
        detector = get_detector(**self.config.change_detector)
        name = self.config.target_generator.pop('name')
        target_generator = TargetGenerator(name, **self.config.target_generator)
        self.changemixin = ChangeMixin2(
            segmentation,
            classifier,
            detector,
            target_generator,
            self.config.loss
        )

def build_changestar2(num_classes: int, backbone: str = "resnet18") -> nn.Module:
    """
    Build ChangeStar2 using the ever/torchange config dict API.
    Uses semantic_fpn backbone (lighter than default farseg+resnet50).

    Backbone options: 'resnet18', 'resnet50'
    """
    resnet_type = backbone  # 'resnet18' or 'resnet50'
    in_ch_list  = (64, 128, 256, 512) if "18" in backbone else (256, 512, 1024, 2048)
    out_ch      = 128 if "18" in backbone else 256

    cfg = dict(
        segmentation=dict(
            model_type="semantic_fpn",
            backbone=dict(
                resnet_type=resnet_type,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
            ),
            neck=dict(
                in_channels_list=in_ch_list,
                out_channels=out_ch,
            ),
            head=dict(
                in_channels=out_ch,
                out_channels=out_ch,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                classifier_config=None,
            ),
        ),
        semantic_classifier=dict(
            in_channels=out_ch,
            out_channels=num_classes,
            scale=4.0,
        ),
        change_detector=dict(
            name="TSMTDM",
            in_channels=out_ch,
            scale=4.0,
            tsm_cfg=dict(dim=16, drop_path_prob=0.2, num_convs=4),
            tdm_cfg=dict(NConvNeXtBlock=4, PreNorm="LN"),  # lighter for resnet18
        ),
        target_generator=dict(
            name="sync_generate_target_v3",
            shuffle_prob=1.0,
        ),
        loss=dict(
            change=dict(
                symmetry_loss=True,
                bce=True,
                dice=False,
                weight=0.5,
                ignore_index=-1,
                log_bce_pos_neg_stat=False,
            ),
            semantic=dict(
                on=True,
                bce=False,
                dice=False,
                ignore_index=-1,
            ),
            change_type="multi_class",  # semantic change detection mode expects multi_class here
        ),
        pcm_m2m_inference=False,
    )

    model = ChangeStar2SCD(cfg)
    return model


# ── Loss (standalone, bypassing ChangeMixin2 training-mode output) ─────────────
def compute_loss(model_output, batch, device):
    """
    ChangeStar2 in training mode returns a loss_dict directly.
    We prepare the y dict required by torchange and run forward.
    """
    # We handle loss manually outside ChangeMixin2 for full control.
    # This function receives eval-mode outputs and computes losses.
    raise RuntimeError("Should not be called – use model in train mode which returns loss_dict")


# ── Prepare torchange-format y dict for training ───────────────────────────────
from torchange.models.changestar2 import Field

def make_y(batch, device):
    """
    torchange ChangeStar2 training expects:
        y[Field.MASK1]  = label1  (B, H, W) long
        y[Field.MASK2]  = label2  (B, H, W) long   ← will be overwritten by target_generator
        y[Field.XIMG1]  = im1     (B, C, H, W)     ← used by sync_generate_target_v3
        y[Field.XMASK1] = label1  (B, H, W) long   ← used by sync_generate_target_v3
    """
    lb1 = batch["label1"].to(device)
    lb2 = batch["label2"].to(device)

    return {
        Field.MASK1:  lb1,
        Field.MASK2:  lb2,
        Field.XIMG1:  batch["im1"].to(device),
        Field.XMASK1: lb1,
    }


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = SECONDDataset(
        root=args.data_root,
        stems_file=os.path.join(args.data_root, "train.txt"),
        im1_dir =os.path.join(args.data_root, "im1"),
        im2_dir =os.path.join(args.data_root, "im2"),
        lb1_dir =os.path.join(args.data_root, "label1"),
        lb2_dir =os.path.join(args.data_root, "label2"),
        size=args.img_size, augment=True,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
    print(f"Train samples: {len(train_ds)}, Batches/epoch: {len(train_dl)}")

    model = build_changestar2(NUM_CLASSES, backbone=args.backbone).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {total_params:.1f}M")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for step, batch in enumerate(pbar):
            im1 = batch["im1"].to(device)
            # ChangeStar2 in training mode expects x as im1 (3 channels).
            # The target_generator (sync_generate_target_v3) will pair it with a pseudo im2.
            x = im1
            y = make_y(batch, device)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss_dict = model(x, y)   # training mode → returns dict of losses
                total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.requires_grad)

            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            epoch_loss += total_loss.item()
            if step % LOG_INTERVAL == 0:
                loss_str = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items()
                                     if isinstance(v, torch.Tensor))
                pbar.set_postfix_str(loss_str)

        avg_loss = epoch_loss / len(train_dl)
        sched.step()
        lr_now = sched.get_last_lr()[0]
        print(f"[Epoch {epoch+1:03d}/{args.epochs}] loss={avg_loss:.4f}  lr={lr_now:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "loss": best_loss,
                "backbone": args.backbone,
                "num_classes": NUM_CLASSES,
            }, args.save_path)
            print(f"  → ✅ Saved best (loss={best_loss:.4f})")

    print(f"\n✅ Training done. Best loss={best_loss:.4f}  Checkpoint: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default=SECOND_ROOT)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--img-size",   type=int,   default=IMG_SIZE)
    parser.add_argument("--backbone",   default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--device",     default=DEVICE)
    parser.add_argument("--save-path",  default=SAVE_PATH)
    args = parser.parse_args()
    train(args)
