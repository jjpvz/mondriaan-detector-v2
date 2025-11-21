from analyze.pipeline import analyze_features
from dataAcquisition.load import load_images
from featureExtraction.blockHistogram import compute_block_histogram
from featureExtraction.dct import compute_dct_features
from featureExtraction.hog import compute_hog_features
from featureExtraction.orb import compute_orb_features
from preprocessing.pipeline import preprocess_image
from segmentation.colorSegmentation import segment_colors
from featureExtraction.pipeline import extract_features
import pandas as pd
import cv2 as cv

if __name__ == "__main__":
    images = load_images("fullset")
    
    features_list = [] 
    hog_features_list = []
    dct_features_list = []
    orb_features_list = []

    hist_features_list = []

    for i, (image, name, class_name) in enumerate(images):

        image = preprocess_image(image, True)
        
        red_mask, yellow_mask, blue_mask = segment_colors(image, False)

        #features_list.append(extract_features(i, name, class_name, image, red_mask, yellow_mask, blue_mask))
        
        # hist_features_list.append(compute_block_histogram(image))
        #dct_features_list.append(compute_dct_features(image, 8, True))
        # hog_features_list.append(compute_hog_features(image, False))
        # orb_features_list.append(compute_orb_features(image, False))

    dataframe = pd.DataFrame(features_list)
    print(dataframe)
    
    analyze_features(dataframe, hog_features_list, dct_features_list, orb_features_list, hist_features_list)