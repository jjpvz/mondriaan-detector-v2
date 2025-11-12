from helpers.display import display_image
from preprocessing.warp import warp_image
from preprocessing.roi import *


def preprocess_image(img_rgb, visualize: bool = False):
    sharped = unsharp_highpass(img_rgb)

    if visualize:
        display_image(sharped)

    canny = auto_canny(sharped)

    if visualize:
        display_image(canny[0])

    morph = morph_close(canny[0], kernel_size=3, iterations=6)

    if visualize:
        display_image(morph)
    contour = getLargestContour(morph)
    cropped_img = extract_roi(img_rgb, contour)

    if visualize:
        display_image(cropped_img)
        
    warped_img = warp_image(cropped_img)

    if visualize:
        display_image(warped_img)
    
    return warped_img