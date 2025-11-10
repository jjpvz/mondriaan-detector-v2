import cv2 as cv
import numpy as np

from helpers.display import display_image

def count_number_of_colored_squares(image, red_mask, yellow_mask, blue_mask, min_area: int = 0, visualize: bool = False):
    combined_mask = cv.bitwise_or(red_mask, yellow_mask)
    combined_mask = cv.bitwise_or(combined_mask, blue_mask)

    num_labels, labels = cv.connectedComponents(combined_mask)

    image = image.copy()
    count = 0
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area >= min_area:
            count += 1

            if visualize:
                ys, xs = np.where(labels == i)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if visualize:
        display_image(image, title="Colored Squares")
    
    return count

def count_number_of_squares(img, min_area: int = 0, visualize: bool = False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    combined_mask = cv.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    
    num_labels, labels = cv.connectedComponents(combined_mask)

    img = img.copy()
    count = 0
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area >= min_area:
            count += 1

            if visualize:
                ys, xs = np.where(labels == i)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if visualize:
        display_image(img, title="Colored Squares")

    return count
