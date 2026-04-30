# RNA Family Classification Demo

This project is a lightweight web demo for RNA family classification.

## What it does

- Upload RNA sequences in FASTA or CSV format
- Choose between a stable model and a higher-coverage experimental model
- Predict Rfam family labels for each sequence
- Show top-k probabilities, confidence, and risk flags
- Download prediction results as CSV

## Models

- `stable_1669`
  - Stable demonstration model
  - Trained on 1,669 Rfam families
- `high_coverage_2238`
  - Higher-coverage experimental model
  - Trained on 2,238 Rfam families

## Repository scope

Model checkpoints, processed Rfam datasets, raw Rfam FASTA files, and training run outputs are not included in this GitHub repository.

## Environment note

Mamba inference depends on `mamba-ssm`. In the development setup used for this project, the recommended runtime is WSL with CUDA support.
