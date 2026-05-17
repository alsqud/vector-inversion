import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval_pipeline import load_cached_embeddings
from src.protection import add_gaussian_noise
from src.retrieval import evaluate_retrieval


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def evaluate_baseline_or_noise(
    model_type,
    protection_type,
    sigma,
    seed,
    embedding_cache_dir,
):
    query_embeddings, corpus_embeddings, _ = load_cached_embeddings(
        cache_dir=embedding_cache_dir,
        model_type=model_type,
    )

    if protection_type == "baseline":
        protected_corpus_embeddings = corpus_embeddings

    elif protection_type == "noise":
        protected_corpus_embeddings = add_gaussian_noise(
            embeddings=corpus_embeddings,
            sigma=sigma,
            seed=seed,
        )

    else:
        raise ValueError(f"Unsupported protection type: {protection_type}")

    retrieval_result = evaluate_retrieval(
        query_embeddings=query_embeddings,
        corpus_embeddings=protected_corpus_embeddings,
    )

    row = {
        "Model": model_type,
        "Protection": protection_type,
        "Sigma": sigma,
        "Seed": seed,
        "Condition": make_condition_name(model_type, protection_type, sigma),
    }

    row.update(retrieval_result)

    return row


def make_condition_name(model_type, protection_type, sigma):
    if protection_type == "baseline":
        return f"{model_type}-Baseline"

    if protection_type == "noise":
        return f"{model_type}-Noise({sigma})"

    return f"{model_type}-{protection_type}"


def summarize_results(result_df):
    metric_cols = ["Hit@1", "Hit@5", "MRR"]

    summary_rows = []

    for condition, group in result_df.groupby("Condition"):
        row = {
            "Condition": condition,
            "Model": group["Model"].iloc[0],
            "Protection": group["Protection"].iloc[0],
            "Sigma": group["Sigma"].iloc[0],
            "Num_Runs": len(group),
        }

        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run seed-repeat retrieval evaluation for baseline and noise conditions."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the default YAML configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    embedding_cache_dir = config["paths"]["embedding_cache_dir"]
    result_dir = Path(config["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    seeds = config["seed_repeat"]["seeds"]
    conditions = config["seed_repeat"]["baseline_noise_conditions"]

    rows = []

    print("=== Seed Repeat Evaluation: Baseline and Noise ===")

    for condition in conditions:
        model_type = condition["model"]
        protection_type = condition["protection"]
        sigma = float(condition.get("sigma", 0.0))

        for seed in seeds:
            print(
                f"Running: model={model_type}, protection={protection_type}, sigma={sigma}, seed={seed}"
            )

            row = evaluate_baseline_or_noise(
                model_type=model_type,
                protection_type=protection_type,
                sigma=sigma,
                seed=seed,
                embedding_cache_dir=embedding_cache_dir,
            )

            rows.append(row)

    result_df = pd.DataFrame(rows)

    raw_output_path = result_dir / "seed_repeat_baseline_noise_raw.csv"
    result_df.to_csv(raw_output_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_results(result_df)

    summary_output_path = result_dir / "seed_repeat_baseline_noise_summary.csv"
    summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

    print("\nSaved raw results to:")
    print(raw_output_path)

    print("\nSaved summary results to:")
    print(summary_output_path)

    print("\n=== Summary ===")
    print(summary_df)


if __name__ == "__main__":
    main()