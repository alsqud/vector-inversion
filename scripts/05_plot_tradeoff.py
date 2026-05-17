import argparse
from pathlib import Path

import yaml

from src.plotting import plot_tradeoff


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate a retrieval-inversion trade-off figure."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the default YAML configuration file.",
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Path to the merged summary CSV file.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for the trade-off figure.",
    )

    parser.add_argument(
        "--x_metric",
        type=str,
        default="Hit@1",
        help="Metric used for the x-axis.",
    )

    parser.add_argument(
        "--y_metric",
        type=str,
        default="Cosine_Mean",
        help="Metric used for the y-axis.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    result_dir = Path(config["paths"]["result_dir"])
    figure_dir = Path(config["paths"]["figure_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_csv is None:
        summary_csv = result_dir / "seed_repeat_summary_merged.csv"
    else:
        summary_csv = Path(args.summary_csv)

    if args.output_path is None:
        output_path = figure_dir / "tradeoff_hit1_cosine.png"
    else:
        output_path = Path(args.output_path)

    print("=== Plot Trade-off Figure ===")
    print(f"Summary CSV: {summary_csv}")
    print(f"Output path: {output_path}")
    print(f"X metric: {args.x_metric}")
    print(f"Y metric: {args.y_metric}")

    plot_tradeoff(
        summary_csv=summary_csv,
        output_path=output_path,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        label_col="Condition",
    )

    print("\nSaved trade-off figure to:")
    print(output_path)


if __name__ == "__main__":
    main()