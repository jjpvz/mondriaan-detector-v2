from helpers.mask import create_mask
import cv2 as cv

def remove_background(img):
    mask = create_mask(img, [(40, 95)], 50, 50, True)
    masked_img = cv.bitwise_and(img, img, mask=mask)

    return masked_img