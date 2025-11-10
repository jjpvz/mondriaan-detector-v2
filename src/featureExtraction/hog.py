import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from helpers.display import display_image

def compute_hog_features(image, visualize: bool = False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    hog = cv.HOGDescriptor()
    h = hog.compute(gray)

    if visualize:
        # Compute gradients
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=1)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=1)
        magnitude, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

        # Prepare visualization canvas
        vis = np.zeros_like(gray, dtype=np.float32)
        step = 6  # visualize every 8 pixels
        for i in range(0, gray.shape[0], step):
            for j in range(0, gray.shape[1], step):
                mag_cell = magnitude[i:i+step, j:j+step]
                ang_cell = angle[i:i+step, j:j+step]
                avg_ang = np.mean(ang_cell)
                avg_mag = np.mean(mag_cell)

                # Draw line representing average gradient
                center = (j + step//2, i + step//2)
                length = int(step//2 * avg_mag / (np.max(magnitude)+1e-5))
                angle_rad = np.deg2rad(avg_ang)
                x1 = int(center[0] - length * np.cos(angle_rad))
                y1 = int(center[1] - length * np.sin(angle_rad))
                x2 = int(center[0] + length * np.cos(angle_rad))
                y2 = int(center[1] + length * np.sin(angle_rad))
                cv.line(vis, (x1, y1), (x2, y2), color=255, thickness=1)

        # Convert single-channel to RGB for display
        vis_rgb = cv.cvtColor(vis.astype(np.uint8), cv.COLOR_GRAY2RGB)
        display_image(vis_rgb, title="HOG Visualization")

    return h
