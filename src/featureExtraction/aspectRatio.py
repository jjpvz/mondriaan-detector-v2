import numpy as np
import cv2 as cv
from helpers.display import display_image

def compute_aspect_ratio(image, visualize: bool = False):
    height, width = image.shape[:2]

    if height == 0:
        return np.nan
    
    aspect_ratio = (width / height)
    image_center = (width / 2, height / 2)
    image_diagonal = np.sqrt(width ** 2 + height ** 2)

    if visualize:
        annotated_img = image.copy()
        text = f"Aspect Ratio: {aspect_ratio:.2f}"
        cv.putText(annotated_img, text, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        display_image(annotated_img)

    return aspect_ratio, image_center, image_diagonal