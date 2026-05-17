import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def normalize_summary_columns(df):
    """
    Rename mean columns to simple metric names for plotting and paper tables.
    """
    rename_map = {
        "Hit@1_mean": "Hit@1",
        "Hit@5_mean": "Hit@5",
        "MRR_mean": "MRR",
        "Cosine_Mean_mean": "Cosine_Mean",
        "Char_ROUGE_L_F1_mean": "Char_ROUGE_L_F1",
        "Keyword_Recall_mean": "Keyword_Recall",
        "Term_Hit_Rate_mean": "Term_Hit_Rate",
    }

    return df.rename(columns=rename_map)


def main():
    parser = argparse.ArgumentParser(
        description="Merge seed-repeat summary CSV files."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the default YAML configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    result_dir = Path(config["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    baseline_noise_path = result_dir / "seed_repeat_baseline_noise_summary.csv"
    pca_pcanoise_path = result_dir / "seed_repeat_pca_pcanoise_summary.csv"

    summary_files = []

    if baseline_noise_path.exists():
        summary_files.append(baseline_noise_path)
    else:
        print(f"Missing file: {baseline_noise_path}")

    if pca_pcanoise_path.exists():
        summary_files.append(pca_pcanoise_path)
    else:
        print(f"Missing file: {pca_pcanoise_path}")

    if len(summary_files) == 0:
        raise FileNotFoundError(
            "No summary CSV files found. Run scripts 02 and 03 first."
        )

    dataframes = []

    for path in summary_files:
        print(f"Loading: {path}")
        df = pd.read_csv(path)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = normalize_summary_columns(merged_df)

    output_path = result_dir / "seed_repeat_summary_merged.csv"
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nSaved merged summary to:")
    print(output_path)

    print("\n=== Merged Summary ===")
    print(merged_df)


if __name__ == "__main__":
    main()