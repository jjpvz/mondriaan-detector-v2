from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2 as cv
import numpy as np
from tensorflow import keras

from helpers.Gui import show_directory_selection_window, show_prediction_window
from preprocessing.pipeline import preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
IMAGE_SIZE = (224, 224)


def _iter_image_files(directory: Path) -> list[Path]:
	return [
		p
		for p in directory.rglob("*")
		if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
	]


def _select_model_path() -> str | None:
	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	file_path = filedialog.askopenfilename(
		title="Selecteer CNN model",
		filetypes=[
			("Keras model", "*.keras *.h5 *.hdf5"),
			("Alle bestanden", "*.*"),
		],
	)
	root.destroy()
	return file_path or None


def _prepare_cnn_input(img_bgr: np.ndarray) -> np.ndarray:
	img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
	processed = preprocess_image(img_rgb, False)
	resized = cv.resize(processed, IMAGE_SIZE, interpolation=cv.INTER_AREA)
	normalized = resized.astype("float32") / 255.0
	return np.expand_dims(normalized, axis=0)


def _predict_single(model, img_bgr: np.ndarray) -> tuple[int, float]:
	X = _prepare_cnn_input(img_bgr)
	predictions = model.predict(X, verbose=0)
	predicted_idx = int(np.argmax(predictions[0]))
	confidence = float(predictions[0][predicted_idx])
	return predicted_idx, confidence


def test_cnn_model_with_gui(model_path: str | None = None, class_names: list[str] | None = None) -> None:
	if model_path is None:
		model_path = _select_model_path()
		if not model_path:
			print("Geen model geselecteerd. Afgebroken.")
			return

	print("CNN model laden...")
	model = keras.models.load_model(model_path)
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

	for image_path in image_files:
		img = cv.imread(str(image_path))
		if img is None:
			print(f"Kon afbeelding niet laden: {image_path}")
			continue

		predicted_idx, confidence = _predict_single(model, img)
		if class_names and 0 <= predicted_idx < len(class_names):
			print(f"Voorspelling: {class_names[predicted_idx]} (index: {predicted_idx}) - Zekerheid: {confidence * 100:.2f}%")
		else:
			print(f"Voorspelling: index {predicted_idx} - Zekerheid: {confidence * 100:.2f}%")

		show_prediction_window(img, predicted_idx, confidence)


if __name__ == "__main__":
	test_cnn_model_with_gui()
