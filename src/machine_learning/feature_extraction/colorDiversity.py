import cv2 as cv
import numpy as np
from scipy.stats import entropy

from helpers.histogram import plot_hue_histogram

def compute_color_diversity(image, s_threshold=30, v_threshold=30, visualize: bool = False) -> float:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    mask = cv.bitwise_and(cv.inRange(s, s_threshold, 255),
                          cv.inRange(v, v_threshold, 255))
    hist = cv.calcHist([h], [0], mask, [180], [0, 180]).flatten()
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    diversity = float(entropy(hist + 1e-6)) 

    if visualize:
        plot_hue_histogram(image, 0.01, str(diversity))

    return diversity