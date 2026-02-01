import os
import cv2 as cv
import numpy as np
from typing import List, Tuple

ImageTuple = Tuple[np.ndarray, str, str]  # (image, filename, class_name)

def _make_new_filename(filename: str, suffix: str) -> str:
    """Insert a suffix before the file extension."""
    name, ext = os.path.splitext(filename)
    return f"{name}{suffix}{ext}"

def _random_exposure(
    img: np.ndarray,
    brightness_range=(-20, 20),
    contrast_range=(0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random brightness and contrast adjustments."""
    if rng is None:
        rng = np.random.default_rng()

    beta = float(rng.uniform(*brightness_range))
    alpha = float(rng.uniform(*contrast_range))

    out = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def _random_white_balance(
    img: np.ndarray,
    temp_range=(3600, 4100),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply random white balance adjustment by simulating color temperature.
    
    Parameters
    ----------
    img : np.ndarray
        Input RGB image.
    temp_range : tuple
        Temperature range in Kelvin (min, max). Default (3600, 4100).
    rng : np.random.Generator
        Random generator.
    
    Returns
    -------
    np.ndarray
        White balance adjusted image.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Random temperature in Kelvin
    temp = float(rng.uniform(*temp_range))
    
    # Simplified color temperature to RGB conversion
    # Lower temp = warmer (more red), higher temp = cooler (more blue)
    temp = temp / 100.0
    
    if temp <= 66:
        r = 255
        g = temp - 2
        g = 99.4708025861 * np.log(g) - 161.1195681661
        if temp <= 19:
            b = 0
        else:
            b = temp - 10
            b = 138.5177312231 * np.log(b) - 305.0447927307
    else:
        r = temp - 55
        r = 351.97690566805693 * (r ** -0.13320475922)
        g = temp - 50
        g = 325.4494125711974 * (g ** -0.07943437555)
        b = 255
    
    # Normalize to 0-1 range
    r = np.clip(r, 0, 255) / 255.0
    g = np.clip(g, 0, 255) / 255.0
    b = np.clip(b, 0, 255) / 255.0
    
    # Apply to image channels
    img_float = img.astype(np.float32)
    img_float[:, :, 0] *= r  # Red channel
    img_float[:, :, 1] *= g  # Green channel
    img_float[:, :, 2] *= b  # Blue channel
    
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

def augment_images(
    images: List[ImageTuple],
    num_aug_per_image: int = 3,
    seed: int | None = 42,
    show_progress: bool = True,
) -> List[ImageTuple]:
    """
    Generate augmented image variants using simple OpenCV operations.
    No resizing - keeps original image dimensions.

    Parameters
    ----------
    images : list of (img, filename, class_name)
        Output from load_images(...)
    num_aug_per_image : int
        Number of extra variants per original image.
    seed : int | None
        Random seed for reproducibility.
    show_progress : bool
        If True, show progress in console.

    Returns
    -------
    List[ImageTuple]
        List containing original + augmented images.
    """
    rng = np.random.default_rng(seed)
    augmented: List[ImageTuple] = []

    rot_options = [0, 1, 2, 3]       # number of 90° rotations
    flip_options = [None, 0, 1]      # None = no flip, 0 = vertical, 1 = horizontal

    total_images = len(images)
    total_to_generate = total_images * (1 + num_aug_per_image)
    
    if show_progress:
        print(f"Data augmentatie gestart: {total_images} originele afbeeldingen")
        print(f"Genereren van {num_aug_per_image} varianten per afbeelding...")
    
    processed = 0

    for img_idx, (img, filename, class_name) in enumerate(images):
        # Add original
        augmented.append((img, filename, class_name))
        processed += 1

        if show_progress and total_to_generate > 0:
            progress = (processed / total_to_generate) * 100
            print(f"\rAfbeeldingen augmenteren: {progress:.1f}% ({processed}/{total_to_generate})", end="", flush=True)

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            continue

        created = 0

        while created < num_aug_per_image:
            try:
                img_aug = img.copy()

                # 1) Random 90° rotation
                k = rot_options[rng.integers(0, len(rot_options))]
                if k != 0:
                    img_aug = np.rot90(img_aug, k)

                # 2) Random flip
                flip_code = flip_options[rng.integers(0, len(flip_options))]
                if flip_code is not None:
                    img_aug = cv.flip(img_aug, flipCode=flip_code)

                # 3) Random brightness/contrast
                if rng.random() > 0.3:  # 70% chance
                    img_aug = _random_exposure(
                        img_aug,
                        brightness_range=(-20, 20),
                        contrast_range=(0.9, 1.1),
                        rng=rng
                    )

                # 4) Random white balance (color temperature)
                if rng.random() > 0.3:  # 70% chance
                    img_aug = _random_white_balance(
                        img_aug,
                        temp_range=(3600, 4100),
                        rng=rng
                    )

                suffix = f"_aug{created+1}"
                new_filename = _make_new_filename(filename, suffix)

                augmented.append((img_aug, new_filename, class_name))
                processed += 1
                created += 1
                
                if show_progress and total_to_generate > 0:
                    progress = (processed / total_to_generate) * 100
                    print(f"\rAfbeeldingen augmenteren: {progress:.1f}% ({processed}/{total_to_generate})", end="", flush=True)
            
            except (cv.error, ValueError):
                continue

    if show_progress:
        print(f"\n Data augmentatie voltooid! Totaal: {len(augmented)} afbeeldingen ({total_images} origineel + {len(augmented) - total_images} gegenereerd)")

    return augmented