import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class RemoteCLIPRegionEncoder:
    """
    Extract per-region features using RemoteCLIP (ViT-L-14).
    For each region: crop bounding box -> preprocess (GPU-optimized) -> encode.
    """
    def __init__(
        self,
        weights_path: str = None,      # path to RemoteCLIP-ViT-L-14.pt
        device: str = "cuda",
        feature_dim: int = 768,        # ViT-L-14 output dim
        crop_size: int = 224,          # CLIP input size
        pad_ratio: float = 0.1,        # padding around bbox
    ):
        import open_clip
        self.device    = device
        self.feat_dim  = feature_dim
        self.crop_size = crop_size
        self.pad_ratio = pad_ratio

        # Load model
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="datacomp_xl_s13b_b90k"
        )

        # Load RemoteCLIP weights if provided
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            # Clean up prefix if needed
            new_state = {}
            for k, v in state.items():
                if k.startswith("visual."):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            self.model.visual.load_state_dict(new_state, strict=False)
            print(f"✅ RemoteCLIP weights loaded from: {weights_path}")

        self.model = self.model.visual.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_all_regions(
        self,
        image: np.ndarray,
        masks: np.ndarray,      # (N, H, W) bool
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode all region masks in a single image.
        Returns: (N, 768) float32
        """
        N = len(masks)
        features = np.zeros((N, self.feat_dim), dtype=np.float32)
        if N == 0:
            return features

        H, W = image.shape[:2]
        
        # Convert image to GPU tensor: (3, H, W), float, [0, 1]
        img_t = torch.from_numpy(image).permute(2, 0, 1).to(self.device).float() / 255.0
        
        # Normalization constants (CLIP defaults)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(3, 1, 1)

        crops = []
        for i in range(N):
            mask = masks[i]
            ys, xs = np.where(mask)
            if len(ys) == 0:
                crops.append(torch.zeros(3, self.crop_size, self.crop_size, device=self.device))
                continue

            pad_h = int((ys.max() - ys.min()) * self.pad_ratio)
            pad_w = int((xs.max() - xs.min()) * self.pad_ratio)

            y1 = max(0, ys.min() - pad_h)
            y2 = min(H, ys.max() + pad_h + 1)
            x1 = max(0, xs.min() - pad_w)
            x2 = min(W, xs.max() + pad_w + 1)

            crop_t = img_t[:, y1:y2, x1:x2]
            
            # Scale shorter side to crop_size (224)
            h_crop, w_crop = crop_t.shape[1], crop_t.shape[2]
            if h_crop < w_crop:
                h_new = self.crop_size
                w_new = int(round(w_crop * self.crop_size / h_crop))
            else:
                w_new = self.crop_size
                h_new = int(round(h_crop * self.crop_size / w_crop))

            # Resize
            crop_resized = F.interpolate(
                crop_t.unsqueeze(0),
                size=(h_new, w_new),
                mode="bicubic",
                align_corners=False
            ).squeeze(0)

            # Center Crop
            y_start = (h_new - self.crop_size) // 2
            x_start = (w_new - self.crop_size) // 2
            crop_cropped = crop_resized[:, y_start:y_start+self.crop_size, x_start:x_start+self.crop_size]

            # Normalize
            crop_normalized = (crop_cropped - mean) / std
            crops.append(crop_normalized)

        # Process in batches
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = torch.stack(crops[start:end])
            feats = self.model(batch)                    # (B, 768)
            feats = F.normalize(feats, dim=-1)           # L2 normalize
            features[start:end] = feats.cpu().numpy()

        return features

if __name__ == "__main__":
    encoder = RemoteCLIPRegionEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    print("✅ RemoteCLIP encoder initialized")
    print(f"   Feature dim: {encoder.feat_dim}")

    # Test with dummy input
    dummy_img   = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_masks = np.random.rand(10, 512, 512) > 0.9

    feats = encoder.encode_all_regions(dummy_img, dummy_masks)
    print(f"   Output shape: {feats.shape}")  # (10, 768)
    assert feats.shape == (10, 768)
    print("✅ Test passed")
