import numpy as np
import cv2 as cv

from preprocessing.pipeline import preprocess_image

def prepare_data_for_dl(images):
    class_names = sorted(list(set(item[2] for item in images)))
    class_to_int = {name: i for i, name in enumerate(class_names)}
    
    X = []
    y = []
    
    for image_raw, _, class_name in images:
        image_processed = preprocess_image(image_raw, False)

        resized_img = cv.resize(image_processed, (244, 244), interpolation=cv.INTER_AREA)

        image_normalized = resized_img.astype('float32') / 255.0
        
        X.append(image_normalized)
        y.append(class_to_int[class_name])
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y, class_names