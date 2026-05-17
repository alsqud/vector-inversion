import argparse
from pathlib import Path

from src.data_utils import check_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset files and basic split sizes."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/defense_terms_clean_v2",
        help="Path to the dataset directory containing train.txt, val.txt, and test.txt.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=== Dataset Check ===")
    print(f"Dataset directory: {data_dir}")

    try:
        result = check_dataset(data_dir)

        print("\n=== Split Sizes ===")
        print(f"Train size: {result['train_size']}")
        print(f"Validation size: {result['val_size']}")
        print(f"Test size: {result['test_size']}")

        print("\n=== Test Samples ===")
        for idx, sample in enumerate(result["test_sample"], start=1):
            print(f"{idx}. {sample}")

        print("\nDataset check completed.")

    except FileNotFoundError as error:
        print("\nDataset check failed.")
        print(error)
        print("\nExpected file structure:")
        print("data/defense_terms_clean_v2/train.txt")
        print("data/defense_terms_clean_v2/val.txt")
        print("data/defense_terms_clean_v2/test.txt")


if __name__ == "__main__":
    main()