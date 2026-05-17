# Model Parameters

This document summarizes the main model and training parameters used in the experiments.

## 1. Embedding Models

| Model ID | Description | Base Model |
|---|---|---|
| E0 | Public embedding model | jhgan/ko-sbert-sts |
| E1 | Defense-domain fine-tuned embedding model | jhgan/ko-sbert-sts |

## 2. Embedding Fine-tuning Setting

E1 is fine-tuned from E0 using defense-domain term-definition positive pairs.

| Parameter | Value |
|---|---:|
| Training examples | 25,620 |
| Epochs | 3 |
| Batch size | 64 |
| Steps per epoch | 401 |
| Total steps | 1,203 |
| Warmup steps | 120 |
| Loss function | MultipleNegativesRankingLoss |

## 3. Inversion Attacker Model

The inversion attacker is based on an embedding-to-text generation structure.

| Parameter | Value |
|---|---|
| Generator model | gogamza/kobart-base-v2 |
| Architecture | Embedding-to-KoBART prefix projection |
| Input embedding dimension | 768 |
| Prefix length | 16 |
| Max generation length | 96 |
| Batch size | 32 |
| Best checkpoint | best_model.pt |
| Best checkpoint epoch | 13 |

## 4. Text Generation Parameters

| Parameter | Value |
|---|---:|
| Number of beams | 4 |
| No repeat n-gram size | 3 |
| Repetition penalty | 1.15 |
| Length penalty | 1.0 |
| Early stopping | true |

## 5. Evaluation Metrics

### Retrieval Metrics

- Hit@1
- Hit@5
- MRR

### Inversion Metrics

- Cosine Mean
- Char ROUGE-L F1
- Keyword Recall
- Term Hit Rate

## 6. Vector Protection Settings

| Protection Setting | Main Parameter |
|---|---|
| Gaussian noise | sigma = 0.03 |
| PCA | dimension = 256 |
| PCA + Gaussian noise | dimension = 256, sigma = 0.03 |

## 7. Repeated Seeds

The experiments are repeated using the following random seeds:

```text
42, 123, 2024
```

The final reported values are based on mean and standard deviation across repeated runs.