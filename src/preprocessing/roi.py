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

def extract_roi(img, contour):
    left, right, top, bottom = getContourExtremes(contour)
    x_min, x_max = left[0], right[0]
    y_min, y_max = top[1], bottom[1]
    cropped_img = img[y_min:y_max, x_min:x_max]
    return cropped_img

def auto_canny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    # Check if image is already grayscale (1 channel) or needs conversion
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    else:
        gray = img
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(gray, lower, upper, L2gradient=True)
    return edged, (lower, upper)

def morph_close(binary, kernel_size, iterations):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=iterations)
    return closed

def morph_open(binary, kernel_size, iterations):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=iterations)
    return opened

def highpass_filter(bgr, radius=3, gain=3.0):
    #gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv.GaussianBlur(bgr, (21,21), radius)
    hp = bgr - blur + 127
   # hp = hp * gain
    hp_norm = cv.normalize(hp, None, alpha=0, beta=255,
                           norm_type=cv.NORM_MINMAX)
    return hp_norm.astype(np.uint8)

def unsharp_highpass(bgr, radius=3, amount=1.2): #Edge boost via unsharp mask in BGR. 
    blur = cv.GaussianBlur(bgr, (0, 0), radius) 
    return cv.addWeighted(bgr, 1 + amount, blur, -amount, 0)