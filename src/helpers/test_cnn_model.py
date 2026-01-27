from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow import keras

from helpers.Gui import show_directory_selection_window, show_prediction_window
from helpers.display import display_image
from preprocessing.pipeline import preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
IMAGE_SIZE = (224, 224)


def _select_model_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Selecteer een CNN model",
        filetypes=[
            ("Keras model", "*.keras *.h5 *.hdf5"),
            ("Alle bestanden", "*.*"),
        ],
    )

    root.destroy()

    return Path(file_path) if file_path else None


def _iter_image_files(directory: Path) -> list[Path]:
    return [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _prepare_cnn_input(img_bgr: np.ndarray) -> np.ndarray:
    resized = cv.resize(img_bgr, IMAGE_SIZE, interpolation=cv.INTER_AREA)
    img_rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    ready_img = img_rgb.astype("float32")
    return np.expand_dims(ready_img, axis=0)


def _predict_single(model, img_bgr: np.ndarray) -> tuple[int, float]:
    batch = _prepare_cnn_input(img_bgr)
    predictions = model.predict(batch, verbose=0)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx])
    return predicted_idx, confidence


def test_cnn_model_with_gui(class_names: list[str] | None = None, model_path: str | Path | None = None) -> None:
    model_file = Path(model_path) if model_path else _select_model_file()
    if not model_file:
        print("Geen model geselecteerd. Afgebroken.")
        return

    print(f"CNN model laden: {model_file}")
    model = keras.models.load_model(model_file)
    print("Model geladen. Selecteer een map met afbeeldingen.")

    directory = show_directory_selection_window()
    if not directory:
        print("Geen map geselecteerd. Afgebroken.")
        return

    folder = Path(directory)
    image_files = _iter_image_files(folder)

    if not image_files:
        print("Geen afbeeldingen gevonden in de gekozen map.")
        return

    for image_path in sorted(image_files):
        img = cv.imread(str(image_path))
        if img is None:
            print(f"Kon afbeelding niet laden: {image_path}")
            continue

        predicted_idx, confidence = _predict_single(model, img)
        predicted_label = class_names[predicted_idx] if class_names else predicted_idx
        print(f"Voorspelling: {predicted_label} (index: {predicted_idx}) - Zekerheid: {confidence * 100:.2f}%")

        show_prediction_window(img, predicted_label, confidence)
