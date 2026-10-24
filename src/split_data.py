
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/processed")
OUTPUT_DIR = DATA_DIR / "splits"


def main():
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")

    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", X[train_idx])
    np.save(OUTPUT_DIR / "y_train.npy", y[train_idx])
    np.save(OUTPUT_DIR / "X_test.npy", X[test_idx])
    np.save(OUTPUT_DIR / "y_test.npy", y[test_idx])

    print("Train EEG:", X[train_idx].shape)
    print("Test EEG:", X[test_idx].shape)
    print("Train labels:", y[train_idx].shape)
    print("Test labels:", y[test_idx].shape)

    print("\nTraining trials per digit:")
    print(np.bincount(y[train_idx], minlength=10))

    print("\nTesting trials per digit:")
    print(np.bincount(y[test_idx], minlength=10))

    print("\nTrain/test overlap:", len(set(train_idx) & set(test_idx)))


if __name__ == "__main__":
    main()
