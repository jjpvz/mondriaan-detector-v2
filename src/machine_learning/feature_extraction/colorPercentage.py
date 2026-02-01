import cv2 as cv
import numpy as np
from helpers.display import display_image

def compute_color_percentage(image, mask, visualize: bool = False):
    total_pixels = mask.size
    color_pixels = cv.countNonZero(mask)

    percentage = (color_pixels / total_pixels) * 100

    if visualize:
        annotated_img = image.copy()
        annotated_img[np.where(mask > 0)] = [0, 0, 255]

        text = f"{percentage:.2f}%"
        cv.putText(
            annotated_img,
            text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        display_image(annotated_img, title="Color Percentage Mask")

    return percentage