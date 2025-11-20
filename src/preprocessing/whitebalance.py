import cv2 as cv
import numpy as np

calibrate = False

# Pre-calculated white balance gains from calibration procedure, will be written over if calibrate=True
gains = np.array([1.0361509629354595, 1.003614566478534, 0.9629354435696218], dtype=np.float32)

if calibrate:
    img = cv.imread("C:\GIT\mondriaan-detector-v2\Whitebalance.jpg")
    img3800k = cv.imread("C:/GIT/mondriaan-detector-v2/3800k.JPG")

    if img is None:
        print("Error: could not load image.")
        exit()

    if img3800k is None:
        print("Error: could not load 3800k image.")
        exit()

    x1, y1 = 30, 30
    x2, y2 = 150, 150

    roi3800k = img3800k[y1:y2, x1:x2]
    roi = img[y1:y2, x1:x2]

    imshow = cv.imshow("ROI", roi)
    imshow3800k = cv.imshow("ROI 3800K", roi3800k)
    cv.waitKey(0)
    cv.destroyAllWindows()

    b_mean3800k, g_mean3800k, r_mean3800k = roi3800k.reshape(-1, 3).mean(axis=0)
    print("3800K Mean B,G,R:", b_mean3800k, g_mean3800k, r_mean3800k)
    b_mean, g_mean, r_mean = roi.reshape(-1, 3).mean(axis=0)
    print("Mean B,G,R:", b_mean, g_mean, r_mean)

    gray_target = (b_mean + g_mean + r_mean) / 3
    gray_target_3800k = (b_mean3800k + g_mean3800k + r_mean3800k) / 3

    gain_b = gray_target / b_mean
    gain_g = gray_target / g_mean
    gain_r = gray_target / r_mean

    gain_b_3800k = gray_target_3800k / b_mean3800k
    gain_g_3800k = gray_target_3800k / g_mean3800k
    gain_r_3800k = gray_target_3800k / r_mean3800k

    print("3800K WB gains:", gain_b_3800k, gain_g_3800k, gain_r_3800k)

    print("WB gains:", gain_b, gain_g, gain_r)

    mean_gain_b = (gain_b + gain_b_3800k) / 2
    mean_gain_g = (gain_g + gain_g_3800k) / 2
    mean_gain_r = (gain_r + gain_r_3800k) / 2
    print("Mean WB gains:", mean_gain_b, mean_gain_g, mean_gain_r)

    gains = np.array([mean_gain_b, mean_gain_g, mean_gain_r], dtype=np.float32)


# Apply fixed white balance gains to an image
# input RGB image and numpy array of gains for B, G, R channels
def apply_fixed_wb(img, gains):
    bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    img32 = bgr_img.astype(np.float32)
    img32 = img32 * gains 
    img32 = np.clip(img32, 0, 255)
    img32 = cv.cvtColor(img32, cv.COLOR_BGR2RGB)
    return img32.astype(np.uint8)