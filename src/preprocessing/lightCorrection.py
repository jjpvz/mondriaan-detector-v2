import cv2 as cv
import numpy as np
import math

def licht_correction(img_rgb, blur_sigma=30, clip_hist_percent=1.0, target_mean=0.5, eps=1e-5):

    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV) .astype(np.float32)   
    h, s, v = cv.split(hsv)

    v_norm = v / 255.0

    blurred = cv.GaussianBlur(v_norm, (0,0), blur_sigma)

    v_homografh = v_norm / (blurred + eps)

    v_homografh = v_homografh / (v_homografh.max() + eps)

    if clip_hist_percent > 0.0:
       low_p = clip_hist_percent
       high_p = 100.0 - clip_hist_percent
       v_low = np.percentile(v_homografh, low_p)
       v_high = np.percentile(v_homografh, high_p)
       if v_high > v_low + eps:
           v_cs = (v_homografh - v_low) / (v_high - v_low)
           v_cs = np.clip(v_cs, 0.0, 1.0)
       else:
           v_cs = np.clip(v_homografh, 0.0, 1.0)

    else:
        v_cs = np.clip(v_homografh, 0.0, 1.0)

    cur_mean = float(v_cs.mean())
    if cur_mean > eps:
        gamma = math.log(target_mean + eps) / math.log(cur_mean + eps)
    else:
        gamma = 1.0
    
    v_gamma = np.power(v_cs,gamma)
    v_gamma = np.clip(v_gamma, 0.0, 1.0)

    v_out = (v_gamma * 255.0).astype(np.float32)
    hsv_corr = cv.merge([h, s, v_out])
    img_corrected = cv.cvtColor(hsv_corr.astype(np.uint8), cv.COLOR_HSV2RGB)

    return img_corrected
