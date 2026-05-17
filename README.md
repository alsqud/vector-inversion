# Vector Inversion

This repository provides reproducibility code for experiments on embedding inversion risks and vector protection trade-offs in a Korean defense-domain RAG setting.

## Purpose

This project evaluates whether sentence embeddings used in Retrieval-Augmented Generation systems may leak semantic information through embedding inversion attacks.

The experiments compare:

1. A public Korean sentence embedding model
2. A defense-domain fine-tuned embedding model
3. Gaussian noise protection
4. PCA dimensionality reduction
5. PCA + Gaussian noise protection

## Main Models

| Component | Model |
|---|---|
| Public embedding model E0 | `jhgan/ko-sbert-sts` |
| Fine-tuned embedding model E1 | Defense-domain positive-pair fine-tuned E0 |
| Inversion attacker generator | `gogamza/kobart-base-v2` |
| Attacker architecture | Embedding-to-KoBART prefix projection |

## Dataset Format

Each line follows:

```text
term: definition
```

Example:

```text
공개첩보중분류코드: 공개첩보의 대분류를 세분화한 코드
```

The full dataset is not included in this repository due to redistribution and security restrictions.

Sample-format files are provided under:

```text
data/sample/
```

To reproduce the full paper results, place the full dataset files as follows:

```text
data/defense_terms_clean_v2/train.txt
data/defense_terms_clean_v2/val.txt
data/defense_terms_clean_v2/test.txt
```

## Repository Structure

```text
vector-inversion/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── config/
│   ├── default.yaml
│   └── paths.example.yaml
├── data/
│   ├── README.md
│   └── sample/
├── src/
├── scripts/
├── results/
├── figures/
└── docs/
```

## Required External Files

The following files are required to fully reproduce the reported results:

```text
data/defense_terms_clean_v2/train.txt
data/defense_terms_clean_v2/val.txt
data/defense_terms_clean_v2/test.txt

checkpoints/best_model.pt

outputs/embedding_cache/E0_public_test_query_terms.npy
outputs/embedding_cache/E0_public_test_corpus_term_def.npy
outputs/embedding_cache/E0_public_test_texts.npy
outputs/embedding_cache/E0_public_train_texts.npy

outputs/embedding_cache/E1_finetuned_test_query_terms.npy
outputs/embedding_cache/E1_finetuned_test_corpus_term_def.npy
outputs/embedding_cache/E1_finetuned_test_texts.npy
outputs/embedding_cache/E1_finetuned_train_texts.npy
```

Large checkpoints, full datasets, embedding caches, and prediction-level outputs are not tracked by Git.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduction Steps

### 1. Check environment

```bash
python scripts/00_check_environment.py
```

### 2. Check dataset format

```bash
python scripts/01_check_data.py --data_dir data/defense_terms_clean_v2
```

### 3. Run baseline and noise seed-repeat evaluation

```bash
python scripts/02_seed_repeat_eval.py --config config/default.yaml
```

### 4. Run PCA and PCA+Noise seed-repeat evaluation

```bash
python scripts/03_pca_pcanoise_repeat_eval.py --config config/default.yaml
```

### 5. Merge seed-repeat results

```bash
python scripts/04_merge_seed_results.py --config config/default.yaml
```

### 6. Generate trade-off figure

```bash
python scripts/05_plot_tradeoff.py --config config/default.yaml
```

### 7. Generate representative case tables

```bash
python scripts/06_generate_case_tables.py --input_csv results/predictions/example_predictions.csv --output_csv results/representative_cases.csv
```

## Main Metrics

### Retrieval Metrics

- Hit@1
- Hit@5
- MRR

### Inversion Metrics

- Cosine Mean
- Char ROUGE-L F1
- Keyword Recall
- Term Hit Rate

## Security and Redistribution Notice

The full dataset, model checkpoints, embedding cache files, and prediction-level outputs are not included in this repository due to redistribution, file size, and security considerations.

This repository is intended to provide:

- Reproducibility code
- Configuration files
- Sample data format
- Evaluation scripts
- Documentation for reproducing the experimental setting

## Citation

Citation information will be added after publication.

## License

Code is released under the MIT License. Dataset and model checkpoints may have separate licenses or redistribution restrictions.