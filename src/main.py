from analyze.pipeline import analyze_features
from dataAcquisition.load import load_images
from deepLearning.createModel import create_cnn_model
from deepLearning.prepareData import prepare_data_for_dl
from featureExtraction.blockHistogram import compute_block_histogram
from featureExtraction.dct import compute_dct_features
from featureExtraction.hog import compute_hog_features
from featureExtraction.orb import compute_orb_features
from helpers.display import display_image
from preprocessing.pipeline import preprocess_image
from segmentation.colorSegmentation import segment_colors
from featureExtraction.pipeline import extract_features
import pandas as pd
import cv2 as cv
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def apply_machine_learning():
    features_list = [] 
    hog_features_list = []
    dct_features_list = []
    orb_features_list = []

    hist_features_list = []

    for i, (image, name, class_name) in enumerate(images):

        image = preprocess_image(image, False)
        
        red_mask, yellow_mask, blue_mask = segment_colors(image, False)

        features_list.append(extract_features(i, name, class_name, image, red_mask, yellow_mask, blue_mask))
        
        # hist_features_list.append(compute_block_histogram(image))
        #dct_features_list.append(compute_dct_features(image, 8, True))
        # hog_features_list.append(compute_hog_features(image, False))
        # orb_features_list.append(compute_orb_features(image, False))

    dataframe = pd.DataFrame(features_list)
    print(dataframe)
    
    # analyze_features(dataframe, hog_features_list, dct_features_list, orb_features_list, hist_features_list)

def apply_deep_learning(images):

    X, y, class_names = prepare_data_for_dl(images)

    input_shape = X.shape[1:] 
    num_classes = len(class_names)

    print(f"Totale dataset vorm (X): {X.shape}") 
    print(f"Aantal klassen: {num_classes}")

    model = create_cnn_model(input_shape, num_classes)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Gebruik SC_Crossentropy voor integer labels
                metrics=['accuracy'])

    model.summary()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=20,         
        validation_split=0.1,
        shuffle=True)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest Nauwkeurigheid: {test_acc:.4f}')

    predictions = model.predict(X_test)   
    most_probable_predictions = np.argmax(predictions, axis=1)

    print("\nVoorbeeld van voorspellingen op de testset:")
    for i in range(5):
        print(f"Werkelijke label: {class_names[y_test[i]]} (Index: {y_test[i]})")
        print(f"Voorspeld label: {class_names[most_probable_predictions[i]]} (Index: {most_probable_predictions[i]})\n")


if __name__ == "__main__":
    images = load_images("fullset")

   # apply_deep_learning(images)
    apply_machine_learning()