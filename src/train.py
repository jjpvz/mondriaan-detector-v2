import argparse
import keras
import numpy as np
import pandas as pd
import joblib
import configparser
import tensorflow as tf

from sklearn.utils import compute_class_weight

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

from transfer_learning.transfer_learning_model import finetune_mobilenetv2, transfer_learning_model

def train_ml_model():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
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

    joblib.dump(model_rm, config['General']['ml_path'])

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
    config = configparser.ConfigParser()
    config.read('config.ini')
    fullset = config['General']['fullset_path']
    model_path = config['General']['tl_path']

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

    class_labels = np.concatenate([y for x, y in train_ds], axis=0)
    class_weights_array = compute_class_weight( class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
    class_weights = dict(enumerate(class_weights_array))

    model = transfer_learning_model(
            input_shape=input_shape,
            num_classes=num_classes,
            augmentation=augmentation,
            dense_units=256,
            dropout=0.3,
            lr=1e-4
        )
    
    callbacks_initial = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=20,
        callbacks=callbacks_initial,
        class_weight=class_weights
    )
    
    model = finetune_mobilenetv2(model, fine_tune_at=120, lr=1e-5)
    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=30,
        callbacks=callbacks_ft,
        class_weight=class_weights
    )

    model.save(model_path)

    y_val, y_pred = get_predictions(model, val_ds)

    print_specification_report(y_val, y_pred, train_ds)
    plot_confusion_matrix(y_val, y_pred, train_ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML en DL modellen.")
    parser.add_argument(
        'model', 
        choices=['ml', 'dl', 'tl'], 
        help="Welk model wil je trainen? 'ml' voor Random Forest, 'dl' voor CNN, of 'tl' voor MobileNetV2."
    )

    args = parser.parse_args()

    if args.model == 'ml':
        train_ml_model()

    if args.model == 'dl':
        train_dl_model()

    if args.model == 'tl':
        train_tl_model()