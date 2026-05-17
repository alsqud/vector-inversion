# Data

This directory describes the dataset format used in the paper.

The dataset consists of Korean defense-domain term-definition pairs.

## Format

Each line must follow the format below:

```text
term: definition
```

Example:

```text
공개첩보중분류코드: 공개첩보의 대분류를 세분화한 코드
```

The text before the first colon is treated as the term, and the text after the first colon is treated as the definition.

## Splits Used in the Paper

| Split | Count | Usage |
|---|---:|---|
| Train | 25,620 | Embedding model fine-tuning and inversion attacker training |
| Validation | 3,203 | Checkpoint selection |
| Test | 3,203 | Final retrieval and inversion evaluation |

## Full Dataset Placement

The full dataset is not included in this repository due to redistribution and security restrictions.

To reproduce the full paper results, place the full dataset files as follows:

```text
data/defense_terms_clean_v2/train.txt
data/defense_terms_clean_v2/val.txt
data/defense_terms_clean_v2/test.txt
```

## Sample Files

Sample files with the same input format are provided under:

```text
data/sample/
```

These sample files are only provided to illustrate the input data format. They are not the full dataset used in the paper.