import cv2 as cv
import numpy as np
import math

def licht_correction(img_rgb, blur_sigma=30, clip_hist_percent=1.0, target_mean=0.75, eps=1e-5):
    """
    Corrects uneven lighting in an image using homomorphic filtering and gamma correction.
    
    This function addresses lighting variations by:
    1. Separating illumination from reflectance using homomorphic filtering
    2. Normalizing the brightness distribution with contrast stretching
    3. Adjusting overall brightness to a target level using gamma correction
    
    Parameters:
    -----------
    img_rgb : numpy.ndarray
        Input image in RGB color space
    blur_sigma : float, default=30
        Sigma value for Gaussian blur kernel. Higher values create stronger smoothing
        of the illumination component (range: 10-50 typical)
    clip_hist_percent : float, default=1.0
        Percentage of histogram to clip from both ends for contrast stretching.
        Higher values increase contrast but may lose detail (range: 0.0-5.0)
    target_mean : float, default=0.8
        Target mean brightness level after correction (range: 0.0-1.0)
        0.5 = medium brightness, lower = darker, higher = brighter
    eps : float, default=1e-5
        Small epsilon value to prevent division by zero
    
    Returns:
    --------
    numpy.ndarray
        Light-corrected image in RGB color space
    """
    
    # Convert to HSV color space to work with the brightness (V) channel
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV) .astype(np.float32)   
    h, s, v = cv.split(hsv)

    # Normalize V channel to [0, 1] range
    v_norm = v / 255.0

    # Apply Gaussian blur to estimate the illumination component
    blurred = cv.GaussianBlur(v_norm, (0,0), blur_sigma)

    # Homomorphic filtering: divide original by blurred to remove illumination variations
    v_homografh = v_norm / (blurred + eps)

    # Normalize to [0, 1] range
    v_homografh = v_homografh / (v_homografh.max() + eps)

    # Apply contrast stretching by clipping histogram percentiles
    if clip_hist_percent > 0.0:
       low_p = clip_hist_percent
       high_p = 100.0 - clip_hist_percent
       v_low = np.percentile(v_homografh, low_p)
       v_high = np.percentile(v_homografh, high_p)
       if v_high > v_low + eps:
           # Stretch the histogram between the clipped percentiles
           v_cs = (v_homografh - v_low) / (v_high - v_low)
           v_cs = np.clip(v_cs, 0.0, 1.0)
       else:
           v_cs = np.clip(v_homografh, 0.0, 1.0)

    else:
        v_cs = np.clip(v_homografh, 0.0, 1.0)

    # Calculate gamma value to adjust mean brightness to target
    cur_mean = float(v_cs.mean())
    if cur_mean > eps:
        gamma = math.log(target_mean + eps) / math.log(cur_mean + eps)
    else:
        gamma = 1.0
    
    # Apply gamma correction to achieve target brightness
    v_gamma = np.power(v_cs,gamma)
    v_gamma = np.clip(v_gamma, 0.0, 1.0)

    # Scale back to 0-255 range and merge with original H and S channels
    v_out = (v_gamma * 255.0).astype(np.float32)
    hsv_corr = cv.merge([h, s, v_out])
    
    # Convert back to RGB color space
    img_corrected = cv.cvtColor(hsv_corr.astype(np.uint8), cv.COLOR_HSV2RGB)

    return img_corrected
