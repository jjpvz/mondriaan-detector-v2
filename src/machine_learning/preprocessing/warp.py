import cv2 as cv
import numpy as np

from machine_learning.preprocessing.roi import auto_canny, getLargestContour, morph_close, unsharp_highpass

def warp_image(cropped_img):
    # Apply sharpening to RGB image first
    sharpened = unsharp_highpass(cropped_img)
    
    canny = auto_canny(sharpened)
    morph = morph_close(canny[0], kernel_size=3, iterations=1)
    contour_in_cropped = getLargestContour(morph)
    # src points from contour
    if contour_in_cropped is None:
        return cropped_img  # return original if no contour found
    src = get_quadrilateral_from_contour(contour_in_cropped)  # TL,TR,BR,BL
    
    if src is None:
        return cropped_img  # return original if no valid quadrilateral found

    # target dimensions
    (tl, tr, br, bl) = src
    w_top  = np.linalg.norm(tr - tl)
    w_bot  = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right= np.linalg.norm(br - tr)
    maxW = int(round(max(w_top, w_bot)))
    maxH = int(round(max(h_left, h_right)))

    # destination points and homography
    dst = np.array([
        [0,     0],
        [maxW-1,0],
        [maxW-1,maxH-1],
        [0,     maxH-1]
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(src, dst)

    # warping
    warped = cv.warpPerspective(cropped_img, M, (maxW, maxH),
                                flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_REPLICATE)
    return warped

def get_quadrilateral_from_contour(contour):
    if contour is None or len(contour) == 0:
        return None
    
    # get corners from contour
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return order_quad_points(approx.reshape(-1, 2))
    # fallback: minAreaRect if no quad found
    rect = cv.minAreaRect(contour.reshape(-1,1,2))
    box = cv.boxPoints(rect)
    return order_quad_points(box)

def order_quad_points(pts):
    if pts is None or len(pts) < 4:
        return None
    
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)