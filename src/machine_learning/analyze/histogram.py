import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distribution(df, feature_name, bins=20, kde=True, figsize=(6,5)):
    plt.figure(figsize=figsize)
    sns.histplot(df[feature_name], bins=bins, kde=kde, color='skyblue', edgecolor='black')
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(f"Distribution of {feature_name}")
    plt.tight_layout()
    plt.show()
