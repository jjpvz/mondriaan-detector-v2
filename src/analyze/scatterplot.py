import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_scatter(df, feature_x, feature_y, hue='class', figsize=(6,5)):
    plt.figure(figsize=figsize)
    sns.scatterplot(x=feature_x, y=feature_y, hue=hue, data=df, palette='Set2', s=50, alpha=0.7)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
