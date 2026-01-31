from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from helpers.learningcurve import plot_learning_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.learningcurve import plot_learning_curve

def train_lightgbm(df):
    label_encoder = LabelEncoder()
    df["class_encoded"] = label_encoder.fit_transform(df["class"])

    X = df.drop(columns=["id", "filename", "class", "class_encoded"])
    Y = df["class_encoded"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    # ---- GPU LightGBM model ----
    ML_model = LGBMClassifier(
        objective="multiclass",         # of "binary"
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.1,
        class_weight="balanced",
        
        # GPU instellingen:
        device="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        max_bin=63,

        n_jobs=-1,
        random_state=42
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
    plt.title("Confusion Matrix - LightGBM (GPU)")
    plt.show()

    # CV-scores
    cv_scores = cross_val_score(ML_model, X_train, Y_train, cv=5, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    plot_learning_curve(ML_model, X_train, Y_train, cv=cv, n_jobs=-1).show()

    return ML_model


def gridsearch_LGBM(df):
    label_encoder = LabelEncoder()
    df["class_encoded"] = label_encoder.fit_transform(df["class"])

    X = df.drop(columns=["id", "filename", "class", "class_encoded"])
    Y = df["class_encoded"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    base_model = LGBMClassifier(
        objective="multiclass",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",

        # GPU-settings:
        device="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        max_bin=63
    )

    param_grid = {
        "n_estimators": [200, 500, 1000],
        "learning_rate": [0.1, 0.05, 0.01],
        "max_depth": [3, 5, 7, -1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [0.0, 0.1],
        "class_weight": ["balanced", None]
    }

    grid_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        verbose=2
    )

    grid_search.fit(X_train, Y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print(f"Train accuracy: {accuracy_score(Y_train, y_train_pred):.4f}")
    print(f"Test  accuracy: {accuracy_score(Y_test, y_test_pred):.4f}")

    print("Classification report:")
    print(classification_report(Y_test, y_test_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, y_test_pred))

    plot_learning_curve(best_model, X_train, Y_train, cv=5, n_jobs=-1).show()

    return None
