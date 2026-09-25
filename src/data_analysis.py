import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"


# Load TSV files
def load_dataset(file_path):
    return pd.read_csv(file_path, sep="\t", dtype=str).fillna("")


def analyze_source(file_path, source_name):
    df = load_dataset(file_path)

    print(f"\n{'=' * 60}")
    print(f"{source_name} ANALYSIS")
    print(f"{'=' * 60}")

    print(f"Total records: {len(df)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print((df == "").sum())

    print("\nCountry distribution:")
    if "country" in df.columns:
        print(df["country"].value_counts(dropna=False))

    print("\nSample records:")
    print(df.head(5).to_string(index=False))

    return df


def analyze_ground_truth(file_path):
    df = load_dataset(file_path)

    print(f"\n{'=' * 60}")
    print("GROUND TRUTH ANALYSIS")
    print(f"{'=' * 60}")

    print(f"Total Source 1 entities: {len(df)}")

    # Count matching IDs for each Source 1 entity
    df["match_count"] = df["matched_entity_ids"].apply(
        lambda x: len([item for item in x.split(",") if item.strip()])
    )

    print("\nMatch count distribution:")
    print(df["match_count"].value_counts().sort_index())

    print("\nSingletons:")
    print((df["match_count"] == 0).sum())

    print("\nEntities with at least one match:")
    print((df["match_count"] > 0).sum())

    print("\nAverage matches per Source 1 entity:")
    print(df["match_count"].mean())


def main():
    print("BUSINESS ENTITY RESOLUTION")
    print("DATASET ANALYSIS")

    # Training datasets
    analyze_source(
        TRAIN_DIR / "train_source1.tsv",
        "TRAIN SOURCE 1"
    )

    analyze_source(
        TRAIN_DIR / "train_source2.tsv",
        "TRAIN SOURCE 2"
    )

    analyze_source(
        TRAIN_DIR / "train_source3.tsv",
        "TRAIN SOURCE 3"
    )

    analyze_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    # Test datasets
    analyze_source(
        TEST_DIR / "test_source1.tsv",
        "TEST SOURCE 1"
    )

    analyze_source(
        TEST_DIR / "test_source2.tsv",
        "TEST SOURCE 2"
    )

    analyze_source(
        TEST_DIR / "test_source3.tsv",
        "TEST SOURCE 3"
    )


if __name__ == "__main__":
    main()