import seaborn as sns
import matplotlib.pyplot as plt

def show_correlation_heatmap(df):
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.show()
