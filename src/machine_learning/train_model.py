from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helpers.learningcurve import plot_learning_curve

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
        n_estimators = 618,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 2,
        n_jobs = -1,
        random_state = 42,
        bootstrap = True,
        criterion = "entropy",
        class_weight = "balanced",
        ccp_alpha = 0.001,
        max_features= "sqrt",
        max_samples= None
    )

    ML_model.fit(X_train, Y_train)

    y_pred = ML_model.predict(X_test)
    y_proba = ML_model.predict_proba(X_test)
    
    prob_score = np.max(y_proba, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    results_df = pd.DataFrame({
        "True_Label": label_encoder.inverse_transform(Y_test),
        "Predicted_Label": y_pred_labels,
        "Prediction_Probability": prob_score
    })

    print(results_df.head(10))

    print("\nClassification report:")
    print(classification_report(Y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    learning_curve_plt = plot_learning_curve(
        ML_model, X_train, Y_train, 
        cv=cv, 
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )  
    learning_curve_plt.show()
    
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
        n_jobs=1,
        random_state=42
    )
    
    param_grid = {
        "n_estimators": [618],
        "max_depth": [10,None],
        "min_samples_split": [1,2,4,6],
        "min_samples_leaf": [1,2,5,8],
        "max_features": ["sqrt",0.5],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced"],
        "ccp_alpha": [0.001, 0.002],
        "max_samples": [None],
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,  
        scoring="accuracy",
        verbose=2
    )

    grid_search.fit(X_train, Y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(Y_train, y_train_pred)

    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(Y_test, y_test_pred)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print("Classification report (test):")
    print(classification_report(Y_test, y_test_pred))
    print("Confusion matrix (test):")
    print(confusion_matrix(Y_test, y_test_pred))

    plot_learning_curve(best_model, X_train, Y_train, cv=5, n_jobs=-1)
    plt.show()

    return None

