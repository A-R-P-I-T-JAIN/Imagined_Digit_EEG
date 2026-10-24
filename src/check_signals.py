
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/processed")
X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")


def main():
    print("===== Raw EEG Diagnostics =====")
    print("Shape:", X.shape)
    print("Data type:", X.dtype)
    print("All finite:", np.isfinite(X).all())
    print("Overall min:", float(X.min()))
    print("Overall max:", float(X.max()))
    print("Overall mean:", float(X.mean()))
    print("Overall std:", float(X.std()))

    # Statistics for individual trials.
    trial_std = X.std(axis=(1, 2))
    trial_mean = X.mean(axis=(1, 2))

    print("\nTrial standard deviation:")
    print("Min:", float(trial_std.min()))
    print("Median:", float(np.median(trial_std)))
    print("Max:", float(trial_std.max()))

    print("\nTrial mean:")
    print("Min:", float(trial_mean.min()))
    print("Median:", float(np.median(trial_mean)))
    print("Max:", float(trial_mean.max()))

    print("\n===== Per-Digit Statistics =====")
    for digit in range(10):
        signals = X[y == digit]

        print(
            f"Digit {digit}: "
            f"trials={len(signals)}, "
            f"mean={signals.mean():.6g}, "
            f"std={signals.std():.6g}, "
            f"median trial std={np.median(signals.std(axis=(1, 2))):.6g}"
        )

    # Check whether channels have constant or nearly constant values.
    channel_std = X.std(axis=(0, 2))
    print("\nChannels with std < 1e-8:", int(np.sum(channel_std < 1e-8)))
    print("Channel std min:", float(channel_std.min()))
    print("Channel std max:", float(channel_std.max()))


if __name__ == "__main__":
    main()
