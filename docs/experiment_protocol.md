# Experiment Protocol

This document summarizes the experimental protocol used in the paper.

## 1. Dataset Preparation

The dataset consists of Korean defense-domain term-definition pairs.

Each line follows:

```text
term: definition
```

The full dataset is split into:

| Split | Count | Usage |
|---|---:|---|
| Train | 25,620 | Embedding fine-tuning and inversion attacker training |
| Validation | 3,203 | Checkpoint selection |
| Test | 3,203 | Final evaluation |

The full dataset is not included in this repository. To reproduce the full results, place the dataset files under:

```text
data/defense_terms_clean_v2/train.txt
data/defense_terms_clean_v2/val.txt
data/defense_terms_clean_v2/test.txt
```

## 2. Embedding Models

Two embedding settings are evaluated.

| Model | Description |
|---|---|
| E0 | Public Korean sentence embedding model |
| E1 | Defense-domain fine-tuned embedding model |

E0 uses:

```text
jhgan/ko-sbert-sts
```

E1 is obtained by fine-tuning E0 using defense-domain positive pairs.

## 3. Retrieval Evaluation

Retrieval is evaluated using term names as queries and term-definition pairs as the corpus.

The retrieval metrics are:

- Hit@1
- Hit@5
- MRR

## 4. Embedding Inversion Evaluation

A KoBART-based embedding-to-text attacker is trained using E0 embeddings and original text pairs.

The generator model is:

```text
gogamza/kobart-base-v2
```

The main inversion metrics are:

- Cosine Mean
- Char ROUGE-L F1
- Keyword Recall
- Term Hit Rate

## 5. Transfer Attack Setting

For E1, the E0-trained inversion attacker is reused without retraining.

This setting is a non-adaptive transfer attack. It does not imply that E1 is secure against adaptive attackers trained directly on E1 embeddings.

## 6. Vector Protection Settings

The following vector protection settings are evaluated:

| Protection | Description |
|---|---|
| Noise | Gaussian noise added to embeddings |
| PCA | Dimensionality reduction using PCA |
| PCA + Noise | PCA followed by Gaussian noise |

Representative settings:

```text
noise sigma = 0.03
PCA dimension = 256
```

## 7. Repeated Runs

Experiments are repeated using three random seeds:

```text
42, 123, 2024
```

The reported results are based on mean and standard deviation across repeated runs.