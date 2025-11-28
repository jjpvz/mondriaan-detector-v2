import cv2 as cv
import numpy as np
from helpers.display import display_image

def compute_dct_features(image, block_size: int = 8, visualize: bool = False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0  # normaliseer

    dct = cv.dct(gray)

    if visualize:
        dct_vis = np.log(np.abs(dct) + 1)
        dct_vis = cv.normalize(dct_vis, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        display_image(cv.applyColorMap(dct_vis, cv.COLORMAP_JET), title="DCT Spectrum")

    dct_block = dct[:block_size, :block_size]
    features = dct_block.flatten()

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features
