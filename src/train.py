
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path("data/processed")
SPLIT_DIR = DATA_DIR / "splits"
FEATURE_DIR = DATA_DIR / "csp_features"
OUTPUT_DIR = Path("outputs")

SEED = 42
tf.keras.utils.set_random_seed(SEED)


def build_model(input_dim, num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    F_train = np.load(FEATURE_DIR / "F_train.npy").astype(np.float32)
    F_test = np.load(FEATURE_DIR / "F_test.npy").astype(np.float32)

    y_train = np.load(SPLIT_DIR / "y_train.npy")
    y_test = np.load(SPLIT_DIR / "y_test.npy")

    # Compute weights from training labels only.
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    model = build_model(F_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        F_train,
        y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(F_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print("\n===== Test Results =====")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=np.arange(10),
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=np.arange(10)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / "csp_mlp.keras")
    np.save(OUTPUT_DIR / "test_predictions.npy", predictions)

    print("\nSaved model to outputs/csp_mlp.keras")


if __name__ == "__main__":
    main()
