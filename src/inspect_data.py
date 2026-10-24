
from pathlib import Path
from collections import Counter

DATA_PATH = Path(
    "data/MindBigData64_Mnist2022-EEGv0.016.txt"
)

EXPECTED_COLUMNS = 26388
digit_counts = Counter()
total_rows = 0
invalid_rows = 0

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

with DATA_PATH.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        line = line.strip()

        if not line:
            continue

        total_rows += 1
        parts = line.split(",")

        if len(parts) != EXPECTED_COLUMNS:
            invalid_rows += 1

            if invalid_rows <= 5:
                print(
                    f"Invalid row {line_number}: "
                    f"{len(parts)} columns"
                )
            continue

        try:
            label = int(parts[2])
        except ValueError:
            invalid_rows += 1
            continue

        if 0 <= label <= 9:
            digit_counts[label] += 1

print("\nDataset inspection")
print("------------------")
print("File:", DATA_PATH.name)
print("Total non-empty rows:", total_rows)
print("Invalid rows:", invalid_rows)
print("Valid digit trials:", sum(digit_counts.values()))

print("\nDigit distribution:")
for digit in range(10):
    print(f"Digit {digit}: {digit_counts[digit]}")

print("\nExpected values per row:", EXPECTED_COLUMNS)
print("EEG channels:", 64)
print("Samples per channel:", 400)
print("Sampling rate: 200 Hz")
