import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.eval_pipeline import load_cached_embeddings, load_train_text_embeddings
from src.protection import (
    fit_pca,
    apply_pca_for_retrieval,
    apply_pca_noise_for_retrieval,
)
from src.retrieval import evaluate_retrieval


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def make_condition_name(model_type, protection_type, pca_dim, sigma):
    if protection_type == "pca":
        return f"{model_type}-PCA{pca_dim}"

    if protection_type == "pca_noise":
        return f"{model_type}-PCA{pca_dim}+Noise({sigma})"

    return f"{model_type}-{protection_type}"


def evaluate_pca_or_pca_noise(
    model_type,
    protection_type,
    pca_dim,
    sigma,
    seed,
    embedding_cache_dir,
):
    query_embeddings, corpus_embeddings, _ = load_cached_embeddings(
        cache_dir=embedding_cache_dir,
        model_type=model_type,
    )

    train_text_embeddings = load_train_text_embeddings(
        cache_dir=embedding_cache_dir,
        model_type=model_type,
    )

    pca, explained_variance = fit_pca(
        train_embeddings=train_text_embeddings,
        pca_dim=pca_dim,
        seed=seed,
    )

    if protection_type == "pca":
        protected_query_embeddings, protected_corpus_embeddings = apply_pca_for_retrieval(
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
            pca=pca,
        )

    elif protection_type == "pca_noise":
        protected_query_embeddings, protected_corpus_embeddings = apply_pca_noise_for_retrieval(
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
            pca=pca,
            sigma=sigma,
            seed=seed,
        )

    else:
        raise ValueError(f"Unsupported protection type: {protection_type}")

    retrieval_result = evaluate_retrieval(
        query_embeddings=protected_query_embeddings,
        corpus_embeddings=protected_corpus_embeddings,
    )

    row = {
        "Model": model_type,
        "Protection": protection_type,
        "PCA_Dim": pca_dim,
        "Sigma": sigma,
        "Seed": seed,
        "Explained_Variance": explained_variance,
        "Condition": make_condition_name(
            model_type=model_type,
            protection_type=protection_type,
            pca_dim=pca_dim,
            sigma=sigma,
        ),
    }

    row.update(retrieval_result)

    return row


def summarize_results(result_df):
    metric_cols = ["Hit@1", "Hit@5", "MRR", "Explained_Variance"]

    summary_rows = []

    for condition, group in result_df.groupby("Condition"):
        row = {
            "Condition": condition,
            "Model": group["Model"].iloc[0],
            "Protection": group["Protection"].iloc[0],
            "PCA_Dim": group["PCA_Dim"].iloc[0],
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
        description="Run seed-repeat retrieval evaluation for PCA and PCA+Noise conditions."
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
    conditions = config["seed_repeat"]["pca_conditions"]

    rows = []

    print("=== Seed Repeat Evaluation: PCA and PCA+Noise ===")

    for condition in conditions:
        model_type = condition["model"]
        protection_type = condition["protection"]
        pca_dim = int(condition.get("pca_dim", 256))
        sigma = float(condition.get("sigma", 0.0))

        for seed in seeds:
            print(
                f"Running: model={model_type}, protection={protection_type}, "
                f"pca_dim={pca_dim}, sigma={sigma}, seed={seed}"
            )

            row = evaluate_pca_or_pca_noise(
                model_type=model_type,
                protection_type=protection_type,
                pca_dim=pca_dim,
                sigma=sigma,
                seed=seed,
                embedding_cache_dir=embedding_cache_dir,
            )

            rows.append(row)

    result_df = pd.DataFrame(rows)

    raw_output_path = result_dir / "seed_repeat_pca_pcanoise_raw.csv"
    result_df.to_csv(raw_output_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_results(result_df)

    summary_output_path = result_dir / "seed_repeat_pca_pcanoise_summary.csv"
    summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

    print("\nSaved raw results to:")
    print(raw_output_path)

    print("\nSaved summary results to:")
    print(summary_output_path)

    print("\n=== Summary ===")
    print(summary_df)


if __name__ == "__main__":
    main()