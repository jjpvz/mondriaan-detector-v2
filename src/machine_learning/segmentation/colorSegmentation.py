import cv2 as cv
from helpers.display import display_image
from helpers.mask import create_mask

def segment_colors(img, visualize: bool = False):
    if visualize:
        display_image(img)
   
    k = cv.getStructuringElement(cv.MORPH_RECT, (15,15))

    red_mask = create_mask(img, [(0, 15), (168, 180)], 100, 90)

    if visualize:
        display_image(red_mask)

    yellow_mask = create_mask(img, [(16, 28)], 100, 120)

    if visualize:
        display_image(yellow_mask)
    blue_mask_temp = create_mask(img, [(100, 130)], 100, 60)
    blue_mask = cv.morphologyEx(blue_mask_temp, cv.MORPH_OPEN, k, iterations=1)

    if visualize:
        display_image(blue_mask)
        
    return red_mask, yellow_mask, blue_mask