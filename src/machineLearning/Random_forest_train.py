from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
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
        n_estimators = 1000,
        max_depth = None,
        min_samples_split = 3,
        min_samples_leaf = 1,
        n_jobs = -1,
        random_state = 42,
        bootstrap = True,
        criterion = "entropy",
        class_weight = "balanced",
        ccp_alpha = 0.001,
        max_features="sqrt"
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

    # perform cross-validation to evaluate model stability
    cv_scores = cross_val_score(ML_model, X_train, Y_train, cv=5, scoring="accuracy")

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    # Learning curve - reduced splits and sequential processing to avoid memory issues
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)  # Reduced from 100 to 10
    learning_curve_plt = plot_learning_curve(ML_model, X_train, Y_train, cv=cv, n_jobs=-1)  
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
        "n_estimators": [1],
        "max_depth": [None],
        "min_samples_split": [ 3,4,5],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"],
        "bootstrap": [True],
        "criterion": ["entropy"],
        "class_weight":["balanced"],
        "ccp_alpha": [0.001]
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

        # ---- ==> hier begint de overfitting-check <== ----
    best_model = grid_search.best_estimator_

    # Train-accuracy
    y_train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(Y_train, y_train_pred)

    # Test-accuracy
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

