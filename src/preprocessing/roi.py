import cv2 as cv
import numpy as np

def getLargestContour(img_BW):
    contours, _ = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)

    return np.squeeze(contour)

def getContourExtremes(contour):
    left = contour[contour[:, 0].argmin()]
    right = contour[contour[:, 0].argmax()]
    top = contour[contour[:, 1].argmin()]
    bottom = contour[contour[:, 1].argmax()]

    return np.array((left, right, top, bottom))

def extract_roi(masked_img):
    gray = cv.cvtColor(masked_img, cv.COLOR_RGB2GRAY)
    contour = getLargestContour(gray)
    left, right, top, bottom = getContourExtremes(contour)
    x_min, x_max = left[0], right[0]
    y_min, y_max = top[1], bottom[1]
    cropped_img = masked_img[y_min:y_max, x_min:x_max]
    return cropped_img