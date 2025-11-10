import cv2 as cv
import numpy as np

def compute_block_histogram(img, block_size=(50, 50), bins=16):
    h, w = img.shape[:2]
    features = []

    for y in range(0, h, block_size[1]):
        for x in range(0, w, block_size[0]):
            block = img[y:y+block_size[1], x:x+block_size[0]]
            # Convert to HSV for better color separation
            hsv = cv.cvtColor(block, cv.COLOR_BGR2HSV)
            hist_h = cv.calcHist([hsv], [0], None, [bins], [0, 180])
            hist_s = cv.calcHist([hsv], [1], None, [bins], [0, 256])
            hist_v = cv.calcHist([hsv], [2], None, [bins], [0, 256])
            
            # Normalize histograms and flatten
            hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            features.extend(hist)

    return np.array(features)
