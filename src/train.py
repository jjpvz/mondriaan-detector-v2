import argparse
import keras
import pandas as pd
import joblib
import configparser

from deep_learning.analyze.print_specification_report import print_specification_report
from deep_learning.augment_images import augment_images as augment_images_for_dl
from deep_learning.analyze.plot_confusion_matrix import plot_confusion_matrix
from deep_learning.analyze.plot_learning_curves import plot_learning_curves
from deep_learning.analyze.get_predictions import get_predictions
from deep_learning.train_model import train_model
from deep_learning.get_model import get_optimal_model

from machine_learning.data_acquisition.data_augmentation import augment_images
from machine_learning.data_acquisition.load import load_images
from machine_learning.feature_extraction.pipeline import extract_features
from machine_learning.preprocessing.pipeline import preprocess_image
from machine_learning.segmentation.colorSegmentation import segment_colors
from machine_learning.train_model import gridsearch_RF, train_random_forest

def train_ml_model():
    images = load_images("fullset")

    features_list = [] 

    augmented_images = augment_images(
        images, 
        num_aug_per_image=10,
    )

    total_images = len(augmented_images)
    print(f"\nFeature extraction gestart voor {total_images} afbeeldingen...")
      
    for i, (image, name, class_name) in enumerate(augmented_images):
        image = preprocess_image(image, False)
        red_mask, yellow_mask, blue_mask = segment_colors(image, False)
        features_list.append(extract_features(i, name, class_name, image, red_mask, yellow_mask, blue_mask))
        
        if total_images > 0:
            progress = ((i + 1) / total_images) * 100
            print(f"\rFeatures extraheren: {progress:.1f}% ({i + 1}/{total_images})", end="", flush=True)
        
    dataframe = pd.DataFrame(features_list)

    dataframe.to_csv(config['General']['csv_path'], index=False)

    dataframe = pd.read_csv(config['General']['csv_path'])

    gridsearch_RF(dataframe)

    model_rm = train_random_forest(dataframe)

    joblib.dump(model_rm, config['General']['RF_Model_path'])

def train_dl_model():
    config = configparser.ConfigParser()
    config.read('config.ini')
    fullset = config['General']['fullset_path']
    model_path = config['General']['dl_path']

    image_size = (224, 224)
    batch_size = 32

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        fr"{fullset}",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    augmentation = augment_images_for_dl()

    input_shape = image_size + (3,)
    num_classes = len(train_ds.class_names)

    model = get_optimal_model(input_shape, num_classes, augmentation)

    history = train_model(model, train_ds, val_ds)
    model.save(model_path)

    y_val, y_pred = get_predictions(model, val_ds)

    print_specification_report(y_val, y_pred, train_ds)
    plot_learning_curves(history)
    plot_confusion_matrix(y_val, y_pred, train_ds)

def train_tl_model():
    print("not implemented yet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML en DL modellen.")
    parser.add_argument(
        'model', 
        choices=['ml', 'dl', 'ts'], 
        help="Welk model wil je trainen? 'ml' voor Random Forest, 'dl' voor CNN, of 'tl' voor MobileNetV2."
    )

    args = parser.parse_args()

    if args.model == 'ml':
        train_ml_model()

    if args.model == 'dl':
        train_dl_model()

    if args.model == 'tl':
        train_tl_model()