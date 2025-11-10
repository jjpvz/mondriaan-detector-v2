from preprocessing.roi import getLargestContour
import cv2 as cv
import numpy as np

def warp_image(cropped_img):
    gray_img = cv.cvtColor(cropped_img, cv.COLOR_RGB2GRAY)
    contour_in_cropped = getLargestContour(gray_img)
    # src points from contour
    src = get_quadrilateral_from_contour(contour_in_cropped)  # TL,TR,BR,BL

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
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)