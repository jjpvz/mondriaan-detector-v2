import cv2 as cv
import numpy as np
from helpers.histogram import plot_hue_histogram

def compute_unique_hues(image, threshold = 0.01, visualize: bool = False) -> int:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    
    mask = cv.bitwise_and(cv.inRange(s, 30, 255), cv.inRange(v, 30, 255))
    
    hist = cv.calcHist([h], [0], mask, [180], [0, 180]).flatten()
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    unique_count = int(np.sum(hist > threshold))

    if visualize:
        plot_hue_histogram(image, 0.01, str(unique_count))

    return unique_count
