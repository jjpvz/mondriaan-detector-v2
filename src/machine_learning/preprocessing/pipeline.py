from helpers.display import display_image
from machine_learning.preprocessing.lightCorrection import licht_correction
from machine_learning.preprocessing.roi import auto_canny, extract_rotated_roi, getLargestContour, morph_close, remove_horizontal_borders, unsharp_highpass
from machine_learning.preprocessing.warp import warp_image
from machine_learning.preprocessing.whitebalance import apply_fixed_wb, gains

def preprocess_image(img_rgb, visualize: bool = False):
    if visualize:
        display_image(img_rgb)
   
    balanced_img = apply_fixed_wb(img_rgb, gains)

    if visualize:
        display_image(balanced_img)

    sharped = unsharp_highpass(balanced_img)

    if visualize:
        display_image(sharped)

    canny = auto_canny(sharped)

    if visualize:
        display_image(canny[0])
        

    no_borders = remove_horizontal_borders(canny[0], perc=0.02)

    morph = morph_close(no_borders, kernel_size=3, iterations=3)

    if visualize:
        display_image(morph)

    contour = getLargestContour(morph)
    
    cropped_img = extract_rotated_roi(img_rgb, contour)

    if visualize:
        display_image(cropped_img)
        
    warped_img = warp_image(cropped_img)

    if visualize:
        display_image(warped_img)

    light_corrected = licht_correction(warped_img)

    if visualize:
        display_image(light_corrected)

    return light_corrected