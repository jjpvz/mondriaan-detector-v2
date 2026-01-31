from analyze.boxplot import plot_feature_boxplot
from analyze.correlation_heatmap import show_correlation_heatmap
from analyze.histogram import plot_feature_distribution
from analyze.scatterplot import plot_feature_scatter

def analyze_features(dataframe):
    print(dataframe)
    print(dataframe.describe())

    features_to_use = dataframe.drop(columns=['id', 'filename', 'class'])

    show_correlation_heatmap(features_to_use)

    for _, feat in enumerate(features_to_use):
        plot_feature_distribution(dataframe, feat)

    for _, feat in enumerate(features_to_use):
        plot_feature_boxplot(dataframe, feat)

    # Check if the total number of squares correlates with hue variation.
    plot_feature_scatter(dataframe, "number_of_squares", "unique_hues")

    # Detect whether the relative amount of a color correlates with its spatial placement.
    plot_feature_scatter(dataframe, "red_pct", "red_dist")

    # Shows if more colored squares correlate with dominance of a primary color.
    plot_feature_scatter(dataframe, "number_of_colored_squares", "red_pct")
    plot_feature_scatter(dataframe, "number_of_colored_squares", "blue_pct")
    plot_feature_scatter(dataframe, "number_of_colored_squares", "yellow_pct")