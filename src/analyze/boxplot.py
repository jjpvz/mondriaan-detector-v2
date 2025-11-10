import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_boxplot(df, feature_name):
    plt.figure(figsize=(6,5))
    sns.boxplot(x='class', y=feature_name, data=df)
    plt.xlabel('Class')
    plt.ylabel(feature_name)
    plt.title(f"{feature_name} by Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
