
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

FEATURE_DIR = Path("data/processed/csp_features")
SPLIT_DIR = Path("data/processed/splits")


def main():
    X_train = np.load(FEATURE_DIR / "F_train.npy")
    X_test = np.load(FEATURE_DIR / "F_test.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    y_test = np.load(SPLIT_DIR / "y_test.npy")

    print("Train feature shape:", X_train.shape)
    print("Test feature shape:", X_test.shape)
    print("Train feature std:", float(X_train.std()))
    print("Train feature min/max:", float(X_train.min()), float(X_train.max()))

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, class_weight="balanced"),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n===== SVM Baseline =====")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Macro-F1:",
        f"{f1_score(y_test, predictions, average='macro'):.4f}",
    )
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
