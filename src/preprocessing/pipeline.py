from helpers.display import display_image
from preprocessing.warp import warp_image
from preprocessing.roi import extract_roi
from preprocessing.backgroundRemoval import remove_background

def preprocess_image(img_rgb, visualize: bool = False):
    masked_img = remove_background(img_rgb)

    if visualize:
        display_image(masked_img)

    cropped_img = extract_roi(masked_img)

    if visualize:
        display_image(cropped_img)
        
    warped_img = warp_image(cropped_img)

    if visualize:
        display_image(warped_img)
    
    return warped_img