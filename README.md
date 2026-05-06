# RNA Family Classification Demo

Upload RNA sequences in FASTA or CSV format and classify them into Rfam families using a trained BiGRU sequence model.

This project is a lightweight RNA family classification demo/tool. It packages data preparation results, trained sequence models, a prediction script, and a Flask web interface into a single deployable workflow.

## Final deployed model

The deployed demo uses a BiGRU classifier trained on 2,238 Rfam families.

- Test accuracy: `0.9794`
- Test macro-F1: `0.9746`
- Maximum input length: `1024 nt`

This model was selected because it provides the best balance between family coverage, prediction quality, and deployment simplicity. It uses only standard PyTorch modules and does not require `mamba-ssm` or CUDA-specific dependencies.

The final deployment model is tracked with Git LFS. After cloning the repository, make sure Git LFS files are pulled:

```bash
git lfs install
git lfs pull
```

## Docker quick start

```bash
docker build -t rna-family-classifier .
docker run --rm -p 5000:5000 rna-family-classifier
```

Or:

```bash
docker compose up --build
```

Open the demo at:

```text
http://127.0.0.1:5000
```

## Docker deployment status

Docker CPU-only deployment has been tested successfully.

- `docker build`: PASS
- `docker run`: PASS
- container CPU inference: PASS
- Flask upload prediction: PASS

The deployed container runs on CPU by default:

- No CUDA required
- No `mamba-ssm` required
- Only one model is deployed: `2238-family BiGRU`

## Required model files

Before running the demo, place the model files at:

- `runs_rfam_2238_bigru/bigru_run/best_model.pt`
- `processed_rfam_near_full_v2_1_min20_1024/label_mapping.json`

The demo also expects:

- `data/rfam_family_metadata.csv`

If the model checkpoint or label mapping is not present locally, the app will report a missing runtime file error instead of running prediction.

Docker deployment also requires these files to exist at the same paths before `docker build` and `docker run`.

If the checkpoint is too large for normal GitHub tracking, keep it outside normal source control and provide it through a release asset, local placement, or another artifact distribution path.

If the model file is missing after clone, run:

```bash
git lfs pull
```

## Input formats

Supported inputs:

- FASTA
- CSV with a `sequence` column
- Optional `sequence_id` column

## Output fields

Prediction outputs include:

- `sequence_id`
- `predicted_family`
- `confidence`
- top-k family probabilities
- `risk_flag`
- downloadable CSV

## Development notes

Mamba, CNN, Transformer, and ensemble models were evaluated during development, but the Docker demo deploys only the 2238-family BiGRU model for simplicity and portability.
