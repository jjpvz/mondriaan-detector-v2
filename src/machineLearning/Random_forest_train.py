from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from helpers.learningcurve import plot_learning_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_random_forest(df):
    label_encoder = LabelEncoder()
    df["class_encoded"] = label_encoder.fit_transform(df["class"])

    X = df.drop(columns=["id", "filename", "class", "class_encoded"])
    Y = df["class_encoded"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=42, 
        stratify=Y)

    ML_model = RandomForestClassifier(
        n_estimators = 200,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        n_jobs = -1,
        random_state = 42
    )

    ML_model.fit(X_train, Y_train)

    y_pred = ML_model.predict(X_test)

    
    print("\nClassification report:")
    print(classification_report(Y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    # perform cross-validation to evaluate model stability
    cv_scores = cross_val_score(ML_model, X_train, Y_train, cv=5, scoring="accuracy")

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    plt = plot_learning_curve(ML_model, X_train, Y_train, cv=cv, n_jobs=-1)
    plt.show()

    return ML_model


def gridsearch_RF(df):

    label_encoder = LabelEncoder()
    df["class_encoded"] = label_encoder.fit_transform(df["class"])

    X = df.drop(columns=["id", "filename", "class", "class_encoded"])
    Y = df["class_encoded"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=42, 
        stratify=Y)

    rf = RandomForestClassifier(
    n_jobs=-1,
    random_state=42
)
    
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,  # return to 5 when dataset is bigger trough data augmentation
        n_jobs=-1,
        scoring="accuracy",
        verbose=2
    )

    grid_search.fit(X_train, Y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return None

