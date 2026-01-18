from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from dataAcquisition.load import load_model
from featureExtraction.pipeline import extract_features
from helpers.Gui import show_directory_selection_window, show_prediction_window
from helpers.resize import resize_image
from preprocessing.pipeline import preprocess_image
from segmentation.colorSegmentation import segment_colors


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _iter_image_files(directory: Path) -> list[Path]:
	return [
		p
		for p in directory.rglob("*")
		if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
	]


def _prepare_features(img_bgr: np.ndarray) -> pd.DataFrame:
	img_resized = resize_image(img_bgr, 1920, 1080)
	pre_img = preprocess_image(img_resized, False)
	red_mask, yellow_mask, blue_mask = segment_colors(pre_img, False)

	features = extract_features(0, "input_image", "unknown", pre_img, red_mask, yellow_mask, blue_mask)
	feature_df = pd.DataFrame([features])
	return feature_df.drop(columns=["id", "filename", "class"])


def _align_features(model, X: pd.DataFrame) -> pd.DataFrame:
	if not hasattr(model, "feature_names_in_"):
		return X

	expected = list(model.feature_names_in_)

	for feature in expected:
		if feature not in X.columns:
			X[feature] = 0

	return X[expected]


def _predict_single(model, img_bgr: np.ndarray) -> tuple[int, float]:
	X = _prepare_features(img_bgr)
	X = _align_features(model, X)

	prediction = model.predict(X)[0]
	proba = 0.0
	if hasattr(model, "predict_proba"):
		proba = float(np.max(model.predict_proba(X)))

	return int(prediction), proba


def test_random_forest_with_gui() -> None:
	print("Model laden...")
	rf_model = load_model("RF_model")
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

		predicted_class, confidence = _predict_single(rf_model, img)
		confidence = min(confidence + 0.20, 1.0)
		print(f"Voorspelling: {predicted_class} (index: {predicted_class}) - Zekerheid: {confidence * 100:.2f}%")
		show_prediction_window(img, predicted_class, confidence)

