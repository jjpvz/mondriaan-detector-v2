import cv2 as cv

def create_mask(img_rgb, h_range, s_min, v_min, invert=False):
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    masks = []

    for (h1, h2) in h_range:  # list of (minH, maxH) tuples
        lower = (h1, s_min, v_min)
        upper = (h2, 255, 255)
        masks.append(cv.inRange(hsv, lower, upper))
    mask = masks[0]

    for m in masks[1:]:
        mask = cv.bitwise_or(mask, m)

    # clean up mask with morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel, iterations=1)
    if invert:
        mask = cv.bitwise_not(mask)
    return mask