import cv2 as cv

from helpers.display import display_image

def compute_center_of_mass(image, mask, min_area=500, visualize: bool = False):
    color_pixels = int(cv.countNonZero(mask))

    if color_pixels < min_area:
        return None

    m = cv.moments(mask, binaryImage=True)

    if m["m00"] == 0:
        return None

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    if visualize:
        annotated_img = image.copy()
        cv.circle(annotated_img, (int(cx), int(cy)), radius=5, color=(0, 0, 255), thickness=-1)
        display_image(annotated_img, title="Center of Mass")

    return (float(cx), float(cy))