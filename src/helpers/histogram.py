import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plot_hue_histogram(img, threshold = 0.01, value: str = None):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    mask = cv.bitwise_and(cv.inRange(s, 30, 255), cv.inRange(v, 30, 255))

    hist = cv.calcHist([h], [0], mask, [180], [0, 180]).flatten()
    hist_norm = hist / hist.sum() if hist.sum() > 0 else hist

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(180), hist_norm, color='orange', width=1)
    plt.xlabel("Hue (0-179)")
    plt.ylabel("Normalized Frequency")
    plt.title("Hue Distribution")
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold {threshold}')
    plt.text(130, max(hist_norm)*0.9, f"{value}", fontsize=12, color='blue')
    plt.legend()
    plt.show()
