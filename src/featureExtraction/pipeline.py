from featureExtraction.aspectRatio import compute_aspect_ratio
from featureExtraction.blockHistogram import compute_block_histogram
from featureExtraction.centerOfMass import compute_center_of_mass
from featureExtraction.dct import compute_dct_features
from featureExtraction.euclidianDistance import compute_euclidian_distance
from featureExtraction.colorPercentage import compute_color_percentage
from featureExtraction.hog import compute_hog_features
from featureExtraction.hueVariance import compute_hue_variance
from featureExtraction.colorDiversity import compute_color_diversity
from featureExtraction.uniqueHues import compute_unique_hues
from featureExtraction.standardDeviation import compute_color_std
from featureExtraction.numberOfSquares import count_number_of_colored_squares, count_number_of_squares
from featureExtraction.orb import compute_orb_features

def extract_features(i, name, class_name, img, red_mask, yellow_mask, blue_mask) -> dict[str, float]:
    aspect_ratio, center, diagonal = compute_aspect_ratio(img, False) 

    red_com = compute_center_of_mass(img, red_mask, False)
    yellow_com = compute_center_of_mass(img, yellow_mask, False)
    blue_com = compute_center_of_mass(img, blue_mask, False)

    red_dist = compute_euclidian_distance(img, red_com, center, diagonal, False)
    yellow_dist = compute_euclidian_distance(img, yellow_com, center, diagonal, False)
    blue_dist = compute_euclidian_distance(img, blue_com, center, diagonal, False)

    red_pct = compute_color_percentage(img, red_mask, False)
    yellow_pct = compute_color_percentage(img, yellow_mask, False)
    blue_pct = compute_color_percentage(img, blue_mask, False)

    unique_hues = compute_unique_hues(img, 0.01, False)
    color_diversity = compute_color_diversity(img, 30, 30, False)
    hue_variance = compute_hue_variance(img, False)

    red_std = compute_color_std(img, red_mask, False)
    yellow_std = compute_color_std(img, yellow_mask, False)
    blue_std = compute_color_std(img, blue_mask, False)

    number_of_colored_squares = count_number_of_colored_squares(img, red_mask, yellow_mask, blue_mask, 0, False)
    number_of_squares = count_number_of_squares(img, 0, False)
    orb = compute_orb_features(img, False)
    hog_vec = compute_hog_features(img, False).flatten()
    dct_vec = compute_dct_features(img)
    hist_vec = compute_block_histogram(img)


    orb_features = {f"orb_{k}": float(v) for k, v in enumerate(orb)}
    hog_features = {f"hog_{k}": float(v) for k, v in enumerate(hog_vec)}
    dct_features = {f"dct_{k}": float(v) for k, v in enumerate(dct_vec)}
    hist_features = {f"hist_{k}": float(v) for k, v in enumerate(hist_vec)}

    features = {
        "id": "mondriaan " + str(i + 1),
        "filename": name,
        "class": class_name,
        "aspect_ratio": aspect_ratio,
        "red_dist": red_dist,
        "yellow_dist": yellow_dist,
        "blue_dist": blue_dist,
        "red_pct": red_pct,
        "yellow_pct": yellow_pct,
        "blue_pct": blue_pct,
        "unique_hues": unique_hues,
        "color_diversity": color_diversity,
        "hue_variance": hue_variance,
        "red_std": red_std,
        "yellow_std": yellow_std,
        "blue_std": blue_std,
        "number_of_colored_squares": number_of_colored_squares,
        "number_of_squares": number_of_squares,
        **orb_features,
        **hog_features,
        **dct_features,
        **hist_features,

    }

    return features