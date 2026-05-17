import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import (
    normalize_text,
    truncate_keep_space,
    truncate_no_space,
    char_rouge_l_f1,
    keyword_recall_one,
    term_hit_one,
)


def load_cached_embeddings(cache_dir, model_type):
    """
    Load cached embeddings for E0 or E1.

    Expected files:
        E0_public_test_query_terms.npy
        E0_public_test_corpus_term_def.npy
        E0_public_test_texts.npy

        E1_finetuned_test_query_terms.npy
        E1_finetuned_test_corpus_term_def.npy
        E1_finetuned_test_texts.npy
    """
    cache_dir = Path(cache_dir)

    if model_type == "E0":
        prefix = "E0_public"
    elif model_type == "E1":
        prefix = "E1_finetuned"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    query_path = cache_dir / f"{prefix}_test_query_terms.npy"
    corpus_path = cache_dir / f"{prefix}_test_corpus_term_def.npy"
    text_path = cache_dir / f"{prefix}_test_texts.npy"

    for path in [query_path, corpus_path, text_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding cache file: {path}")

    query_embeddings = np.load(str(query_path)).astype(np.float32)
    corpus_embeddings = np.load(str(corpus_path)).astype(np.float32)
    text_embeddings = np.load(str(text_path)).astype(np.float32)

    return query_embeddings, corpus_embeddings, text_embeddings


def load_train_text_embeddings(cache_dir, model_type):
    """
    Load training text embeddings for PCA fitting.
    """
    cache_dir = Path(cache_dir)

    if model_type == "E0":
        prefix = "E0_public"
    elif model_type == "E1":
        prefix = "E1_finetuned"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_path = cache_dir / f"{prefix}_train_texts.npy"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training embedding cache file: {train_path}")

    return np.load(str(train_path)).astype(np.float32)


def save_predictions_json(predictions, output_path):
    """
    Save generated predictions to a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"predictions": predictions},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_predictions_json(prediction_path):
    """
    Load generated predictions from a JSON file.
    """
    prediction_path = Path(prediction_path)

    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["predictions"]


def evaluate_inversion_predictions(
    reference_texts,
    predicted_texts,
    eval_embedder=None,
    reference_embeddings=None,
    truncate_ratio=1.2,
    save_path=None,
):
    """
    Evaluate inversion predictions.

    If eval_embedder and reference_embeddings are provided,
    Cosine Mean is also calculated.

    Args:
        reference_texts: list of original texts
        predicted_texts: list of reconstructed texts
        eval_embedder: optional SentenceTransformer model
        reference_embeddings: optional reference embeddings
        truncate_ratio: truncation ratio for generated text
        save_path: optional CSV output path

    Returns:
        Dictionary of evaluation metrics.
    """
    reference_texts = [normalize_text(text) for text in reference_texts]
    predicted_raw = [normalize_text(text) for text in predicted_texts]

    predicted_cut_keep_space = [
        truncate_keep_space(ref, pred, ratio=truncate_ratio)
        for ref, pred in zip(reference_texts, predicted_raw)
    ]

    predicted_cut_no_space = [
        truncate_no_space(ref, pred, ratio=truncate_ratio)
        for ref, pred in zip(reference_texts, predicted_raw)
    ]

    rouge_scores = np.array(
        [
            char_rouge_l_f1(ref, pred)
            for ref, pred in zip(reference_texts, predicted_cut_no_space)
        ],
        dtype=np.float32,
    )

    keyword_scores = np.array(
        [
            keyword_recall_one(ref, pred)
            for ref, pred in zip(reference_texts, predicted_cut_keep_space)
        ],
        dtype=np.float32,
    )

    term_hit_scores = np.array(
        [
            term_hit_one(ref, pred)
            for ref, pred in zip(reference_texts, predicted_cut_keep_space)
        ],
        dtype=np.float32,
    )

    result = {
        "Char_ROUGE_L_F1": float(np.nanmean(rouge_scores)),
        "Keyword_Recall": float(np.nanmean(keyword_scores)),
        "Term_Hit_Rate": float(np.nanmean(term_hit_scores)),
    }

    cosine_scores = None

    if eval_embedder is not None and reference_embeddings is not None:
        predicted_embeddings = eval_embedder.encode(
            predicted_cut_keep_space,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        cosine_scores = np.sum(
            reference_embeddings * predicted_embeddings,
            axis=1,
        )

        result["Cosine_Mean"] = float(np.mean(cosine_scores))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        output_df = pd.DataFrame(
            {
                "reference": reference_texts,
                "prediction_raw": predicted_raw,
                "prediction_cut": predicted_cut_keep_space,
                "char_rouge_l_f1": rouge_scores,
                "keyword_recall": keyword_scores,
                "term_hit": term_hit_scores,
            }
        )

        if cosine_scores is not None:
            output_df["cosine"] = cosine_scores

        output_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return result