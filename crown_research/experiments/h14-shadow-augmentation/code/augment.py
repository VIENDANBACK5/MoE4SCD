"""Photometric & Shadow Augmentations for Domain Adaptation (H14).

Simulates:
1. Low-sun / low-angle tree casting shadows across the canopy (RandomShadow)
2. Different UAV camera sensor color response & white balance (RandomHSV)
3. Direct vs diffuse lighting / exposure variation (RandomGamma & Contrast)
"""
from __future__ import annotations

import random
import cv2
import numpy as np


class PhotometricAugmentor:
    """Combines geometric shadow simulation and photometric color jitter."""

    def __init__(
        self,
        shadow_prob: float = 0.60,
        hsv_prob: float = 0.70,
        gamma_prob: float = 0.60,
        contrast_prob: float = 0.50,
    ):
        self.shadow_prob = shadow_prob
        self.hsv_prob = hsv_prob
        self.gamma_prob = gamma_prob
        self.contrast_prob = contrast_prob

    def add_random_shadow(self, img_rgb: np.ndarray) -> np.ndarray:
        """Applies realistic blurred polygon shadows across a fraction of the image."""
        H, W, _ = img_rgb.shape
        num_shadows = random.randint(1, 3)
        img_out = img_rgb.astype(np.float32)

        for _ in range(num_shadows):
            # Create random polygon (3 to 6 vertices)
            num_verts = random.randint(3, 6)
            pts = []
            cx, cy = random.randint(0, W), random.randint(0, H)
            rad = random.randint(int(min(H, W) * 0.25), int(min(H, W) * 0.75))
            for _ in range(num_verts):
                angle = random.uniform(0, 2 * np.pi)
                r = rad * random.uniform(0.5, 1.3)
                px = np.clip(cx + r * np.cos(angle), 0, W - 1)
                py = np.clip(cy + r * np.sin(angle), 0, H - 1)
                pts.append([int(px), int(py)])

            pts = np.array(pts, dtype=np.int32)
            mask = np.zeros((H, W), dtype=np.float32)
            cv2.fillPoly(mask, [pts], 1.0)

            # Soften shadow boundary with Gaussian blur
            ksize = random.choice([25, 35, 45, 55])
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

            # Shadow darkness factor (0.35 to 0.70 of original brightness)
            darkness = random.uniform(0.35, 0.70)
            shadow_mult = 1.0 - mask * (1.0 - darkness)
            img_out *= shadow_mult[:, :, np.newaxis]

        return np.clip(img_out, 0, 255).astype(np.uint8)

    def jitter_hsv(self, img_rgb: np.ndarray) -> np.ndarray:
        """Jitters Hue, Saturation, and Value in HSV space."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Hue delta [-12, +12] degrees (OpenCV Hue is 0..180)
        h_shift = random.uniform(-10, 10)
        # Saturation mult [0.75, 1.25]
        s_scale = random.uniform(0.75, 1.25)
        # Value mult [0.80, 1.20]
        v_scale = random.uniform(0.80, 1.20)

        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def apply_gamma(self, img_rgb: np.ndarray) -> np.ndarray:
        """Applies random gamma correction."""
        gamma = random.uniform(0.75, 1.35)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img_rgb, table)

    def apply_contrast(self, img_rgb: np.ndarray) -> np.ndarray:
        """Adjusts contrast and brightness."""
        alpha = random.uniform(0.80, 1.25)
        beta = random.uniform(-15, 15)
        return np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        out = img_rgb.copy()
        if random.random() < self.shadow_prob:
            out = self.add_random_shadow(out)
        if random.random() < self.hsv_prob:
            out = self.jitter_hsv(out)
        if random.random() < self.gamma_prob:
            out = self.apply_gamma(out)
        if random.random() < self.contrast_prob:
            out = self.apply_contrast(out)
        return out
