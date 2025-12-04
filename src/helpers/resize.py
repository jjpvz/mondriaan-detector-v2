import cv2 as cv
import numpy as np

def resize_image(img, standard_width, standard_height):
    # check orientation and rotate if neccesary

    h, w = img.shape[:2]
    
    # Validate image dimensions
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    
    if h > w:  # portrait image
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
        
    # calculate scale to fit within standard dimensions
    scale = min(standard_width / w, standard_height / h)

    # New dimensions after scaling
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Ensure new dimensions are at least 1 pixel
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Resize with same aspect ratio
    resized = cv.resize(img, (new_w, new_h))
        
    # make canvas with standard dimensions (black instead of white)
    canvas = np.full((standard_height, standard_width, 3), (0, 0, 0), dtype=np.uint8)

    # Center the image on the canvas
    y_offset = (standard_height - new_h) // 2
    x_offset = (standard_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas