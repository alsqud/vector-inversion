from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_summary_csv(summary_csv):
    """
    Load a summary CSV file containing retrieval and inversion metrics.
    """
    summary_csv = Path(summary_csv)

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    return pd.read_csv(summary_csv)


def plot_tradeoff(
    summary_csv,
    output_path,
    x_metric="Hit@1",
    y_metric="Cosine_Mean",
    label_col="Condition",
):
    """
    Plot the trade-off between retrieval performance and inversion risk.

    Args:
        summary_csv: path to summary CSV
        output_path: output figure path
        x_metric: metric for retrieval performance
        y_metric: metric for inversion risk
        label_col: column used for point labels
    """
    df = load_summary_csv(summary_csv)

    required_columns = [x_metric, y_metric, label_col]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in summary CSV: {column}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df[x_metric],
        df[y_metric],
        s=80,
    )

    for _, row in df.iterrows():
        plt.annotate(
            str(row[label_col]),
            (row[x_metric], row[y_metric]),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=9,
        )

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title("Retrieval Performance and Inversion Risk Trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_bar(
    summary_csv,
    output_path,
    metric,
    label_col="Condition",
):
    """
    Plot a bar chart for a selected metric.
    """
    df = load_summary_csv(summary_csv)

    required_columns = [metric, label_col]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in summary CSV: {column}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.bar(
        df[label_col],
        df[metric],
    )

    plt.xlabel(label_col)
    plt.ylabel(metric)
    plt.title(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()