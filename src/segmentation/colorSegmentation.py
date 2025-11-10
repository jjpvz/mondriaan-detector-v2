import cv2 as cv
from helpers.display import display_image
from helpers.mask import create_mask

def segment_colors(img, visualize: bool = False):
    k = cv.getStructuringElement(cv.MORPH_RECT, (15,15))

    red_mask = create_mask(img, [(0, 18), (165, 180)], 110, 70)

    if visualize:
        display_image(red_mask)

    yellow_mask = create_mask(img, [(18, 42)], 70, 90)

    if visualize:
        display_image(yellow_mask)
    blue_mask_temp = create_mask(img, [(105, 135)], 100, 60)
    blue_mask = cv.morphologyEx(blue_mask_temp, cv.MORPH_OPEN, k, iterations=1)

    if visualize:
        display_image(blue_mask)
        
    return red_mask, yellow_mask, blue_mask