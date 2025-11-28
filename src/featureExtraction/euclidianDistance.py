import numpy as np
import cv2 as cv
from helpers.display import display_image

def compute_euclidian_distance(image, center_of_mass, image_center, image_diagonal, visualize: bool = False):
    if center_of_mass is None:
        return 0.0

    distance = np.linalg.norm(np.array(center_of_mass) - np.array(image_center)) / image_diagonal

    if visualize:
        annotated_img = image.copy()
        cx, cy = int(center_of_mass[0]), int(center_of_mass[1])
        icx, icy = int(image_center[0]), int(image_center[1])
        cv.circle(annotated_img, (cx, cy), 5, (0, 0, 255), -1)
        cv.circle(annotated_img, (icx, icy), 5, (255, 0, 0), -1)
        cv.line(annotated_img, (cx, cy), (icx, icy), (0, 255, 0), 2)
        display_image(annotated_img, title="Center of Mass Distance")

    return distance