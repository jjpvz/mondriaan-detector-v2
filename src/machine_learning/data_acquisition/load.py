import configparser
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Literal, Tuple
from helpers.resize import resize_image

def load_images(mode: Literal["subset", "fullset"] = "subset", 
                standard_width: int = 1920,
                standard_height: int = 1080,
                show_progress: bool = True
    ) -> List[Tuple[np.ndarray, str]]:
    
    if mode not in {"subset", "fullset"}:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'subset' or 'fullset'.")

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.ini"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    if not config.has_section("General"):
        raise configparser.NoSectionError("General")

    subset_path = (project_root / config.get("General", "subset_path")).resolve()
    fullset_path = (project_root / config.get("General", "fullset_path")).resolve()
    folder_path = fullset_path if mode.lower() == "fullset" else subset_path

    if not folder_path.exists():
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    files = [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]

    if not files:
        print(f"No image files found in {folder_path}")
        return []

    if show_progress:
        print(f"Gevonden {len(files)} bestanden om te verwerken...")

    imgs_rgb = []
    total_files = len(files)
    
    for i, f in enumerate(files):
        img = cv.imread(str(f))
        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_resized = resize_image(img, standard_width, standard_height)
            class_name = f.parent.relative_to(folder_path).parts[0]
            imgs_rgb.append((img_resized, f.name, class_name))
            
            if show_progress and total_files > 0:
                progress = ((i + 1) / total_files) * 100
                print(f"\rAfbeeldingen laden, verkleinen en converteren: {progress:.1f}% ({i + 1}/{total_files})", end="", flush=True)

    if show_progress:
        print("\n Import, resize en conversie voltooid!")

    return imgs_rgb