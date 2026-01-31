'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains functions to plot a confusion matrix

how to use:
1. plot_confusion_matrix(y_val, y_pred, train_ds)
    - y_val: true labels
    - y_pred: predicted labels
    - train_ds: dataset containing class names
    - Displays a normalized confusion matrix heatmap

'''
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_val, y_pred, train_ds):
    cm = confusion_matrix(y_val, y_pred, normalize='true')

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=train_ds.class_names,
                yticklabels=train_ds.class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()