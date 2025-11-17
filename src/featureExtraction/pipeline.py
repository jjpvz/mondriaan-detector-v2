from featureExtraction.aspectRatio import compute_aspect_ratio
from featureExtraction.centerOfMass import compute_center_of_mass
from featureExtraction.euclidianDistance import compute_euclidian_distance
from featureExtraction.colorPercentage import compute_color_percentage
from featureExtraction.hueVariance import compute_hue_variance
from featureExtraction.colorDiversity import compute_color_diversity
from featureExtraction.uniqueHues import compute_unique_hues
from featureExtraction.standardDeviation import compute_color_std
from featureExtraction.numberOfSquares import count_number_of_colored_squares, count_number_of_squares

def extract_features(i, name, class_name, img, red_mask, yellow_mask, blue_mask) -> dict[str, float]:
    aspect_ratio, center, diagonal = compute_aspect_ratio(img, True)

    red_com = compute_center_of_mass(img, red_mask, True)
    yellow_com = compute_center_of_mass(img, yellow_mask, True)
    blue_com = compute_center_of_mass(img, blue_mask, True)

    red_dist = compute_euclidian_distance(img, red_com, center, diagonal, True)
    yellow_dist = compute_euclidian_distance(img, yellow_com, center, diagonal, True)
    blue_dist = compute_euclidian_distance(img, blue_com, center, diagonal, True)

    red_pct = compute_color_percentage(img, red_mask, True)
    yellow_pct = compute_color_percentage(img, yellow_mask, True)
    blue_pct = compute_color_percentage(img, blue_mask, True)

    unique_hues = compute_unique_hues(img, 0.01, True)
    color_diversity = compute_color_diversity(img, 30, 30, True)
    hue_variance = compute_hue_variance(img, True)

    red_std = compute_color_std(img, red_mask, True)
    yellow_std = compute_color_std(img, yellow_mask, True)
    blue_std = compute_color_std(img, blue_mask, True)

    number_of_colored_squares = count_number_of_colored_squares(img, red_mask, yellow_mask, blue_mask, 0, True)
    number_of_squares = count_number_of_squares(img, 0, True)

    features = {
        "id": i,
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
    }

    return features