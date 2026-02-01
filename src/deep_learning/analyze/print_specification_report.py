'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains functions to print a detailed classification report

how to use:
1. print_specification_report(y_val, y_pred, train_ds)
    - y_val: true labels
    - y_pred: predicted labels
    - train_ds: dataset containing class names
    - Prints precision, recall, F1-score per class and weighted F1-score

'''
from sklearn.metrics import classification_report, f1_score
import numpy as np
import numpy as np

def print_specification_report(y_val, y_pred, train_ds):
    print("\n--- Model Performance Report ---")
    present_labels = np.unique(np.concatenate((y_val, y_pred)))
    present_names = [train_ds.class_names[i] for i in present_labels]

    report = classification_report(y_val, y_pred, labels=present_labels, target_names=present_names)
    print(report)

    f1 = f1_score(y_val, y_pred, labels=present_labels, average='weighted', zero_division=0)
    print(f"Gemiddelde F1-score: {f1:.4f}")