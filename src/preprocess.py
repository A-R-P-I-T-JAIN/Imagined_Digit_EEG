
from pathlib import Path

import numpy as np

DATA_PATH = Path("data/MindBigData64_Mnist2022-EEGv0.016.txt")
OUTPUT_DIR = Path("data/processed")

N_CHANNELS = 64
N_SAMPLES = 400
EXPECTED_FIELDS = 26388
EEG_START = EXPECTED_FIELDS - (64 * 400)


def load_dataset(path):
    eeg_trials = []
    labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            fields = line.split(",")

            if len(fields) != EXPECTED_FIELDS:
                raise ValueError(
                    f"Line {line_number}: expected "
                    f"{EXPECTED_FIELDS} fields, got {len(fields)}"
                )

            label = int(fields[2])

            # Keep only digit classes 0 through 9.
            if label not in range(10):
                continue

            eeg = np.asarray(
                fields[EEG_START:],
                dtype=np.float32
            )

            if eeg.size != N_CHANNELS * N_SAMPLES:
                raise ValueError(
                    f"Line {line_number}: unexpected EEG size "
                    f"{eeg.size}"
                )

            if not np.isfinite(eeg).all():
                raise ValueError(
                    f"Line {line_number}: EEG contains NaN or infinity"
                )

            # Convert 25,600 values to 64 channels × 400 samples.
            eeg = eeg.reshape(N_CHANNELS, N_SAMPLES)

            eeg_trials.append(eeg)
            labels.append(label)

    X = np.stack(eeg_trials)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def debug_row_structure(path):
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            fields = line.split(",")

            print(f"Line: {line_number}")
            print(f"Total fields: {len(fields)}")
            print(f"Label (index 2): {fields[2]}")
            print(f"Fields before EEG: {len(fields) - 25600}")
            print("Candidate final 5 fields:", fields[-5:])
            print("Candidate EEG start index:", len(fields) - 25600)
            break



def main():
    X, y = load_dataset(DATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    print("Preprocessing input validation completed.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("EEG dtype:", X.dtype)
    print("Label dtype:", y.dtype)
    print("EEG values finite:", np.isfinite(X).all())
    print("Labels:", np.unique(y))
    print("Trials per digit:", np.bincount(y, minlength=10))


if __name__ == "__main__":
    main()
