import numpy as np
import cv2 as cv
from helpers.display import display_image

def compute_color_std(image: np.ndarray, mask: np.ndarray, visualize: bool = False) -> dict[str, float]:
    masked_img = np.zeros_like(image)
    masked_img[mask > 0] = image[mask > 0]

    masked_pixels = masked_img[mask > 0]

    if masked_pixels.size == 0:
        return 0.0

    std_r = np.std(masked_pixels[:, 0])
    std_g = np.std(masked_pixels[:, 1])
    std_b = np.std(masked_pixels[:, 2])
    std_mean = np.mean([std_r, std_g, std_b])

    if visualize:
        heatmap = np.std(masked_img.astype(np.float32), axis=2)
        heatmap_norm = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv.applyColorMap(heatmap_norm, cv.COLORMAP_JET)

        cv.putText(
            heatmap_color,
            f"Mean Color Std: {std_mean:.2f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        display_image(heatmap_color, title="Color Std Heatmap")

    return std_mean
