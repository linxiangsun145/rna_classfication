# Rfam Multi-Stage Experiment Plan

## Project Goal

This project is building a practical RNA family classification tool, not a paper-first research codebase.

The target product direction is:

```text
upload FASTA / CSV
→ run a trained RNA family classifier
→ output predicted RNA family labels
→ show results in a frontend
→ support result download
```

The long-term model goal is to cover the full Rfam family space, with the final target being a classifier that can handle up to `4360` families.

The current roadmap is:

```text
50-family validated experiment
→ scale to larger Rfam subsets
→ train a practical 4360-family model
→ add inference pipeline
→ add backend + upload frontend demo
```

This is not primarily a publication project. The main objective is to build a usable, demo-friendly RNA family classification system.

## Current Status

The project already has the core training and evaluation loop in place:

- `scripts/download_and_build_rfam_dataset.py`
  - Download Rfam FASTA files and build large multi-family datasets
- `scripts/build_rfam_dataset.py`
  - Build smaller local-family datasets from existing FASTA files
- `scripts/validate_rfam_dataset.py`
  - Validate split integrity, labels, duplicates, and summary consistency
- `datasets/rna_dataset.py`
  - PyTorch dataset and dataloader support
- `models/cnn_classifier.py`
  - CNN baseline classifier
- `models/mamba_classifier.py`
  - Mamba classifier for sequence modeling
- `scripts/train_classifier.py`
  - Unified training entry for CNN and Mamba
- `scripts/analyze_results.py`
  - Structured experiment result analysis

Completed experiment milestone:

- 50-family Rfam experiment has been completed end-to-end
- Dataset build, validation, training, and analysis all ran successfully
- Mamba outperformed CNN on the 50-family benchmark

## Expansion Strategy

Do not jump directly from 50 families to 4360 families.

Use staged scaling:

```text
50 → 100 → 500 → 1000 → 4360
```

Recommended purpose of each stage:

- `100-family`
  - Smoke test for scaling beyond the initial benchmark
- `500-family`
  - Medium-scale benchmark to expose class growth issues
- `1000-family`
  - Stress test for runtime, memory, and long-tail behavior
- `4360-family`
  - Final production-oriented training stage for the demo model

This staged route lowers risk, makes failures easier to interpret, and keeps later demo work grounded in a stable training pipeline.

## Standard Pipeline

Each stage should follow the same workflow:

```text
build dataset
→ validate dataset
→ train CNN baseline
→ train Mamba
→ analyze results
→ decide next step
```

Do not skip validation or analysis. The project should move forward only when each stage has both trainable outputs and interpretable results.

## Command Templates

### 100-family

Dataset build:

```bash
python scripts/download_and_build_rfam_dataset.py ^
  --raw_dir data/rfam_fasta ^
  --output_dir processed_rfam_100 ^
  --start_id 1 ^
  --end_id 4360 ^
  --max_families 100 ^
  --min_samples_per_family 1000 ^
  --max_samples_per_family 2000
```

Validation:

```bash
python scripts/validate_rfam_dataset.py --data_dir processed_rfam_100
```

CNN:

```bash
python scripts/train_classifier.py ^
  --model cnn ^
  --data_dir processed_rfam_100 ^
  --output_dir runs_rfam_100 ^
  --epochs 10 ^
  --batch_size 64 ^
  --max_len 256 ^
  --lr 1e-3 ^
  --device cuda
```

Mamba:

```bash
python scripts/train_classifier.py \
  --model mamba \
  --data_dir processed_rfam_100 \
  --output_dir runs_rfam_100 \
  --epochs 10 \
  --batch_size 32 \
  --max_len 256 \
  --lr 1e-3 \
  --device cuda
```

Analysis:

```bash
python scripts/analyze_results.py --runs_dir runs_rfam_100
```

### 500-family

Recommended parameters:

```text
max_families = 500
min_samples_per_family = 500
max_samples_per_family = 1000
max_len = 512
batch_size = 16 or 32
```

Dataset build:

```bash
python scripts/download_and_build_rfam_dataset.py ^
  --raw_dir data/rfam_fasta ^
  --output_dir processed_rfam_500 ^
  --start_id 1 ^
  --end_id 4360 ^
  --max_families 500 ^
  --min_samples_per_family 500 ^
  --max_samples_per_family 1000 ^
  --max_len 1000
```

Validation:

```bash
python scripts/validate_rfam_dataset.py --data_dir processed_rfam_500
```

CNN:

```bash
python scripts/train_classifier.py ^
  --model cnn ^
  --data_dir processed_rfam_500 ^
  --output_dir runs_rfam_500 ^
  --epochs 10 ^
  --batch_size 32 ^
  --max_len 512 ^
  --lr 1e-3 ^
  --device cuda
```

Mamba:

```bash
python scripts/train_classifier.py \
  --model mamba \
  --data_dir processed_rfam_500 \
  --output_dir runs_rfam_500 \
  --epochs 10 \
  --batch_size 16 \
  --max_len 512 \
  --lr 1e-3 \
  --device cuda
```

Analysis:

```bash
python scripts/analyze_results.py --runs_dir runs_rfam_500
```

### 1000-family

Recommended parameters:

```text
max_families = 1000
min_samples_per_family = 300
max_samples_per_family = 800
max_len = 512
batch_size = 16
```

Dataset build:

```bash
python scripts/download_and_build_rfam_dataset.py ^
  --raw_dir data/rfam_fasta ^
  --output_dir processed_rfam_1000 ^
  --start_id 1 ^
  --end_id 4360 ^
  --max_families 1000 ^
  --min_samples_per_family 300 ^
  --max_samples_per_family 800 ^
  --max_len 1000
```

Validation:

```bash
python scripts/validate_rfam_dataset.py --data_dir processed_rfam_1000
```

CNN:

```bash
python scripts/train_classifier.py ^
  --model cnn ^
  --data_dir processed_rfam_1000 ^
  --output_dir runs_rfam_1000 ^
  --epochs 10 ^
  --batch_size 16 ^
  --max_len 512 ^
  --lr 1e-3 ^
  --device cuda
```

Mamba:

```bash
python scripts/train_classifier.py \
  --model mamba \
  --data_dir processed_rfam_1000 \
  --output_dir runs_rfam_1000 \
  --epochs 10 \
  --batch_size 16 \
  --max_len 512 \
  --lr 1e-3 \
  --device cuda
```

Analysis:

```bash
python scripts/analyze_results.py --runs_dir runs_rfam_1000
```

### 4360-family

This is the final stage, not the starting point.

Recommended conservative parameters:

```text
max_families = 4360
min_samples_per_family = 50 or 100
max_samples_per_family = 500
max_len = 512
batch_size = 8 or 16
```

Dataset build:

```bash
python scripts/download_and_build_rfam_dataset.py ^
  --raw_dir data/rfam_fasta ^
  --output_dir processed_rfam_4360 ^
  --start_id 1 ^
  --end_id 4360 ^
  --max_families 4360 ^
  --min_samples_per_family 100 ^
  --max_samples_per_family 500 ^
  --max_len 1000
```

Validation:

```bash
python scripts/validate_rfam_dataset.py --data_dir processed_rfam_4360
```

CNN:

```bash
python scripts/train_classifier.py ^
  --model cnn ^
  --data_dir processed_rfam_4360 ^
  --output_dir runs_rfam_4360 ^
  --epochs 10 ^
  --batch_size 16 ^
  --max_len 512 ^
  --lr 1e-3 ^
  --device cuda
```

Mamba:

```bash
python scripts/train_classifier.py \
  --model mamba \
  --data_dir processed_rfam_4360 \
  --output_dir runs_rfam_4360 \
  --epochs 10 \
  --batch_size 8 \
  --max_len 512 \
  --lr 1e-3 \
  --device cuda
```

Analysis:

```bash
python scripts/analyze_results.py --runs_dir runs_rfam_4360
```

## Success Criteria

At the end of each stage, check:

- dataset validation reports `PASS`
- train/val/test all contain the expected set of classes
- macro-F1 is stable and not collapsing
- Mamba vs CNN comparison is meaningful
- no severe long-tail category issue is dominating evaluation
- `min_samples_per_family` and `max_samples_per_family` still make sense for the next stage

What to watch closely:

- a large drop in macro-F1 while accuracy stays high
- many classes with near-zero recall
- confusion concentrated on a few related families
- training instability or repeated CUDA OOM

## When to Move to the Next Stage

Do not advance to the next stage until all of the following are true:

```text
1. dataset validation PASS
2. training completes without CUDA OOM
3. metrics.json / confusion_matrix.csv / per_class_metrics.json / length_bucket_metrics.json all generated
4. analysis_report generated
5. no severe class imbalance issue blocking evaluation
```

If a stage fails any of these checks, adjust data construction before scaling again. The most likely knobs are:

- `min_samples_per_family`
- `max_samples_per_family`
- `batch_size`
- `max_len`

## Future Demo Plan

After the `4360-family` model is trained and validated, move into tool-building rather than more benchmark expansion.

Recommended next implementation order:

```text
predict_sequences.py
→ Flask / FastAPI backend
→ upload FASTA / CSV frontend
→ result table
→ confidence / top3 predictions
→ download predictions.csv
```

Suggested deliverables after the final model:

- batch inference script for FASTA and CSV
- simple API wrapper around the trained model
- upload page for user inputs
- prediction table with family label and confidence
- downloadable CSV result export

That phase should focus on usability, robustness, and demo clarity rather than deeper benchmark complexity.

## Recommended Next Step

The next stage should be `100-family`.

Reason:

- it is the lowest-risk scale-up from the completed 50-family experiment
- it will quickly show whether the pipeline stays stable under larger class count
- it is much easier to debug than jumping directly to 500 or 1000 families
- it keeps the project aligned with the final demo goal instead of turning into open-ended research exploration
