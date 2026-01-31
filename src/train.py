import argparse
import pandas as pd
import joblib
import configparser
import os

from machine_learning.data_acquisition.data_augmentation import augment_images
from machine_learning.data_acquisition.load import load_images
from machine_learning.feature_extraction.pipeline import extract_features
from machine_learning.preprocessing.pipeline import preprocess_image
from machine_learning.segmentation.colorSegmentation import segment_colors
from machine_learning.train_model import gridsearch_RF, train_random_forest

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "..", "config.ini")
config.read(config_path)

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
    print("not supported")
    # X, y, class_names = prepare_data_for_dl(images)

    # input_shape = X.shape[1:] 
    # num_classes = len(class_names)

    # model = create_cnn_model(input_shape, num_classes)
    # model = create_transfer_model(num_classes, input_shape=input_shape)

    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #             metrics=['accuracy'])

    # model.summary()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # history = model.fit(
    #     X_train, 
    #     y_train, 
    #     epochs=20,         
    #     validation_split=0.1,
    #     shuffle=True)
    #     # --- Learning curve ---
    # plt.figure(figsize=(10, 5))
    # plt.plot(history.history['loss'], label='Training loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    # plt.plot(history.history['accuracy'], label='Training accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss / Accuracy")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print(f'\nTest Nauwkeurigheid: {test_acc:.4f}')

    # predictions = model.predict(X_test)   
    # most_probable_predictions = np.argmax(predictions, axis=1)

    # print("\nVoorbeeld van voorspellingen op de testset:")
    # for i in range(5):
    #     print(f"Werkelijke label: {class_names[y_test[i]]} (Index: {y_test[i]})")
    #     print(f"Voorspeld label: {class_names[most_probable_predictions[i]]} (Index: {most_probable_predictions[i]})\n")
    
    # # --- Confusion matrix ---
    # cm = confusion_matrix(y_test, most_probable_predictions, normalize='true')

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(
    #     cm,
    #     annot=True,
    #     cmap="Blues",
    #     xticklabels=class_names,
    #     yticklabels=class_names
    # )
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Normalized Confusion Matrix (testset)")
    # plt.tight_layout()
    # plt.show()

    # model.save("mobilenetv2_mondriaan_model.keras")

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

    if args.model == 'dl' or args.model == 'ts':
        train_dl_model()