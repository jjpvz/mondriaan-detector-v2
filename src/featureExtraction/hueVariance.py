import cv2 as cv
import numpy as np

from helpers.display import display_image

def compute_hue_variance(image, visualize: bool = False) -> float:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    mask = (s > 30) & (v > 30)

    if np.sum(mask) == 0:
        return 0.0

    hue_variance = float(np.var(h[mask]))

    if visualize:
        annotated_img = image.copy()
        annotated_img[np.where(mask)] = [255, 0, 0]
        
        cv.putText(
            annotated_img,
            f"Hue Variance: {hue_variance:.2f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        display_image(annotated_img, title="Hue Variance Visualization")
    return hue_variance