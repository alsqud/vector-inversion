# Reproducibility Checklist

This checklist summarizes the files, settings, and outputs required to reproduce the paper results.

## 1. Repository Files

The following repository files should be available:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `LICENSE`
- `config/default.yaml`
- `config/paths.example.yaml`
- `data/README.md`
- `docs/experiment_protocol.md`
- `docs/model_parameters.md`
- `docs/reproducibility_checklist.md`

## 2. Dataset Files

The full dataset is not included in this repository.

To reproduce the full results, place the dataset files as follows:

```text
data/defense_terms_clean_v2/train.txt
data/defense_terms_clean_v2/val.txt
data/defense_terms_clean_v2/test.txt
```

Expected split sizes:

| Split | Expected Count |
|---|---:|
| Train | 25,620 |
| Validation | 3,203 |
| Test | 3,203 |

Each line must follow:

```text
term: definition
```

## 3. Model and Checkpoint Files

The following model checkpoint is required for full inversion evaluation:

```text
checkpoints/best_model.pt
```

This file is not tracked by Git because it is a large model checkpoint.

## 4. Embedding Cache Files

The following embedding cache files may be required depending on the reproduction mode:

```text
outputs/embedding_cache/E0_public_test_query_terms.npy
outputs/embedding_cache/E0_public_test_corpus_term_def.npy
outputs/embedding_cache/E0_public_test_texts.npy
outputs/embedding_cache/E0_public_train_texts.npy

outputs/embedding_cache/E1_finetuned_test_query_terms.npy
outputs/embedding_cache/E1_finetuned_test_corpus_term_def.npy
outputs/embedding_cache/E1_finetuned_test_texts.npy
outputs/embedding_cache/E1_finetuned_train_texts.npy
```

These files are not tracked by Git because they are large generated artifacts.

## 5. Main Experimental Conditions

The representative experimental conditions are:

| Condition | Description |
|---|---|
| E0-Baseline | Public embedding model without protection |
| E1-Baseline | Defense-domain fine-tuned embedding model without protection |
| E0-Noise(0.03) | Gaussian noise applied to E0 embeddings |
| E1-Noise(0.03) | Gaussian noise applied to E1 embeddings |
| E0-PCA256 | PCA dimensionality reduction applied to E0 embeddings |
| E1-PCA256 | PCA dimensionality reduction applied to E1 embeddings |
| E0-PCA256+Noise(0.03) | PCA and Gaussian noise applied to E0 embeddings |
| E1-PCA256+Noise(0.03) | PCA and Gaussian noise applied to E1 embeddings |

## 6. Random Seeds

The repeated experiments use the following random seeds:

```text
42, 123, 2024
```

## 7. Retrieval Metrics

The retrieval evaluation reports:

- Hit@1
- Hit@5
- MRR

## 8. Inversion Metrics

The inversion evaluation reports:

- Cosine Mean
- Char ROUGE-L F1
- Keyword Recall
- Term Hit Rate

## 9. Security and Redistribution Notice

The full dataset, model checkpoints, embedding cache files, and prediction-level outputs are not included in this repository due to redistribution, file size, and security considerations.

Only sample data format files, configuration files, documentation, and reproducibility code are intended to be tracked by Git.