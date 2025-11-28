import cv2 as cv
import numpy as np
from helpers.display import display_image

def compute_orb_features(image, visualize: bool = False, nfeatures: int = 500):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures=nfeatures)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if visualize:
        img_kp = cv.drawKeypoints(image, keypoints, None, color=(0,255,0),
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        display_image(img_kp, title="ORB Keypoints")

    if descriptors is None or len(descriptors) == 0:
        feature_vector = np.zeros(64) 
    else:
        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        feature_vector = np.concatenate([mean_desc, std_desc])

    return feature_vector
