from analyze.boxplot import plot_feature_boxplot
from analyze.histogram import plot_feature_distribution
from analyze.pca import plot_descriptor_pca
from analyze.scatterplot import plot_feature_scatter

def analyze_features(dataframe, hog_features, dct_features, orb_features, hist_features):
    # print(dataframe)

    features_to_use = dataframe.drop(columns=['id', 'filename', 'class'])
    # print(dataframe.describe())

    # show_correlation_heatmap(features_to_use)

    # for _, feat in enumerate(features_to_use):
    #     plot_feature_distribution(dataframe, feat)

    # for _, feat in enumerate(features_to_use):
    #     plot_feature_boxplot(dataframe, feat)

    # Check if the total number of squares correlates with hue variation.
    # plot_feature_scatter(dataframe, "number_of_squares", "unique_hues")

    # Detect whether the relative amount of a color correlates with its spatial placement.
    # plot_feature_scatter(dataframe, "red_pct", "red_dist")

    # Shows if more colored squares correlate with dominance of a primary color.
    # plot_feature_scatter(dataframe, "number_of_colored_squares", "red_pct")
    # plot_feature_scatter(dataframe, "number_of_colored_squares", "blue_pct")
    # plot_feature_scatter(dataframe, "number_of_colored_squares", "yellow_pct")

    # Class labels from your feature DataFrame
    labels = dataframe['class'].values

    # Plot PCA for each descriptor
    plot_descriptor_pca(hog_features, labels, "HOG")
    plot_descriptor_pca(dct_features, labels, "DCT")
    plot_descriptor_pca(orb_features, labels, "ORB")
    plot_descriptor_pca(hist_features, labels, "Block Histogram")