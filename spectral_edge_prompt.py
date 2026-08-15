# spectral_edge_prompt.py
"""
Tính spectral edge map từ ảnh RGB.
Output: prompt points cho SAM2 (center points của homogeneous regions)

Sử dụng Sobel gradient trên từng channel RGB
→ Vùng gradient thấp = interior of objects → SAM2 foreground prompts
→ Vùng gradient cao = boundaries → SAM2 không prompt tại đây
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def compute_spectral_gradient(image_rgb):
    """
    Tính magnitude của spectral gradient.
    Gradient cao = ranh giới spectral giữa objects.

    Returns: (H, W) float32, normalized [0, 1]
    """
    image_f = image_rgb.astype(np.float32) / 255.0
    grad_mag = np.zeros(image_f.shape[:2], dtype=np.float32)

    for c in range(3):  # R, G, B
        channel = image_f[:, :, c]
        # Sobel gradient
        gx = ndimage.sobel(channel, axis=1)
        gy = ndimage.sobel(channel, axis=0)
        grad_mag += np.sqrt(gx**2 + gy**2)

    # Normalize
    grad_mag /= 3.0
    if grad_mag.max() > 0:
        grad_mag /= grad_mag.max()

    return grad_mag


def find_homogeneous_centers(grad_map, grid_size=16, low_grad_thresh=0.15):
    """
    Tìm các điểm trung tâm của vùng spectral đồng nhất.
    Đây là nơi tốt nhất để prompt SAM2.

    Args:
        grad_map: (H, W) spectral gradient magnitude
        grid_size: khoảng cách giữa các candidate points
        low_grad_thresh: gradient < này → interior of object

    Returns: list of (x, y) prompt points
    """
    H, W = grad_map.shape
    points = []

    for y in range(grid_size // 2, H, grid_size):
        for x in range(grid_size // 2, W, grid_size):
            # Lấy local region
            y1, y2 = max(0, y-grid_size//4), min(H, y+grid_size//4)
            x1, x2 = max(0, x-grid_size//4), min(W, x+grid_size//4)
            local_grad = grad_map[y1:y2, x1:x2].mean()

            # Chỉ prompt tại vùng có gradient thấp (interior)
            if local_grad < low_grad_thresh:
                points.append((x, y))

    return points


def generate_spectral_prompts(image_rgb, grid_size=16, grad_thresh=0.15):
    """
    Main function: từ ảnh RGB → prompt points cho SAM2.
    """
    grad_map = compute_spectral_gradient(image_rgb)
    points   = find_homogeneous_centers(grad_map, grid_size, grad_thresh)
    return points, grad_map


# --- Test ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = np.array(Image.open("SECOND/test/im1/00004.png").convert("RGB"))
    points, grad_map = generate_spectral_prompts(img, grid_size=16)

    print(f"Image size: {img.shape}")
    print(f"Spectral gradient range: [{grad_map.min():.3f}, {grad_map.max():.3f}]")
    print(f"Number of prompt points: {len(points)}")
    print(f"  (vs grid prompting 8×8 = {8*8} points)")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")

    axes[1].imshow(grad_map, cmap='hot')
    axes[1].set_title("Spectral Gradient\n(bright = boundary)")

    axes[2].imshow(img)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    axes[2].scatter(xs, ys, c='lime', s=20, alpha=0.7)
    axes[2].set_title(f"SAM2 Prompts\n({len(points)} points, spectral-guided)")

    plt.tight_layout()
    plt.savefig("spectral_prompts_00004.png", dpi=100)
    print("Saved: spectral_prompts_00004.png")
    print("✅ Test passed")
