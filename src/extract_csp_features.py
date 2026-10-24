
from itertools import combinations
from pathlib import Path

import numpy as np

from csp import BinaryCSP

DATA_DIR = Path("data/processed")
SPLIT_DIR = DATA_DIR / "splits"
OUTPUT_DIR = DATA_DIR / "csp_features"

N_COMPONENTS = 4


def extract_pairwise_features(X_train, y_train, X_test):
    digit_pairs = list(combinations(range(10), 2))

    train_features = []
    test_features = []

    for class_a, class_b in digit_pairs:
        print(f"Fitting CSP for digits {class_a} vs {class_b}")

        model = BinaryCSP(n_components=N_COMPONENTS)
        model.fit(X_train, y_train, class_a, class_b)

        # Transform every trial using this pair's learned filters.
        train_pair = model.transform(X_train)
        test_pair = model.transform(X_test)

        train_features.append(train_pair)
        test_features.append(test_pair)

    # Combine features from all 45 pairs.
    F_train = np.concatenate(train_features, axis=1)
    F_test = np.concatenate(test_features, axis=1)

    return F_train, F_test


def main():
    X_train = np.load(SPLIT_DIR / "X_train.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    X_test = np.load(SPLIT_DIR / "X_test.npy")

    F_train, F_test = extract_pairwise_features(
        X_train, y_train, X_test
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "F_train.npy", F_train)
    np.save(OUTPUT_DIR / "F_test.npy", F_test)

    print("\nCSP feature extraction completed.")
    print("Training feature shape:", F_train.shape)
    print("Testing feature shape:", F_test.shape)
    print("Expected feature count:", 45 * N_COMPONENTS)
    print("Training features finite:", np.isfinite(F_train).all())
    print("Testing features finite:", np.isfinite(F_test).all())


if __name__ == "__main__":
    main()
