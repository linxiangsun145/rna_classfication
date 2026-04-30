# Near-Full-Family V2 Plan

## Project Scope

This project is a practical RNA family classification demo/tool, not a paper-first research project.

The product goal remains:

```text
upload FASTA / CSV
-> run a trained RNA family classifier
-> return predicted RNA family labels
-> show results in a frontend
-> support CSV download
```

The model should be described as a **near-full-family Rfam classifier**, not a strict "complete 4360-family classifier".

## Updated Coverage Position

Use the following wording going forward:

```text
near-full-family Rfam classifier
covering current FASTA-accessible families
recommended training target:
4226 families
max_len = 1024
strict_global_dedup
```

Why this is the right target:

- `133` RF accessions do not have corresponding FASTA files in the current upstream/local `fasta_files` collection.
- Increasing `max_len` from `512` to `1024` recovers all `12` families that were previously `too_long_only`.
- Increasing further to `2048` or `4096` does not recover additional families.
- `strict_global_dedup` loses only `1` family compared with family-preserving dedup, while keeping labels cleaner.

## Recommended Near-Full-Family V2 Configuration

Use this as the default near-full-family v2 build/training target:

```text
raw_dir = data/rfam_fasta
output_dir = processed_rfam_near_full_1024
start_id = 1
end_id = 4360
max_families = 5000
min_samples_per_family = 1
max_samples_per_family = 500
max_len = 1024
dedup = strict_global_dedup
build_only = true
```

Note:

- The current `download_and_build_rfam_dataset.py` does not expose a `--dedup_strategy` flag.
- The current default behavior is effectively **strict global dedup**.
- Do not modify that script yet for this stage; keep the training entry stable.

## Standard Execution Order

For near-full-family v2, always follow this order:

```text
build dataset
-> validate dataset
-> train Mamba in WSL
-> analyze results
-> run inference smoke tests
-> only then consider demo integration
```

Do not update the Flask demo model path until the near-full-family v2 model is fully validated.

## Dataset Build Command

```bash
python scripts/download_and_build_rfam_dataset.py \
  --raw_dir data/rfam_fasta \
  --output_dir processed_rfam_near_full_1024 \
  --start_id 1 \
  --end_id 4360 \
  --max_families 5000 \
  --min_samples_per_family 1 \
  --max_samples_per_family 500 \
  --max_len 1024 \
  --build_only
```

Expected outcome:

- selected families should be close to `4226`
- `train.csv`, `val.csv`, `test.csv` should exist
- `label_mapping.json` should exist
- `dataset_summary.json` should exist

## Dataset Validation Command

Validation must pass before training:

```bash
python scripts/validate_rfam_dataset.py \
  --data_dir processed_rfam_near_full_1024
```

If validation does not pass, do not start training.

## Mamba Training Command

Run training in WSL, not the Windows Python environment:

```bash
wsl -d Ubuntu
cd /mnt/d/vibe_coding/rna_classfication
source .wsl_mamba_env/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

python scripts/train_classifier.py \
  --model mamba \
  --data_dir processed_rfam_near_full_1024 \
  --output_dir runs_rfam_near_full_1024 \
  --max_len 1024 \
  --batch_size 8 \
  --epochs 10 \
  --lr 1e-3 \
  --device cuda
```

Batch-size guidance:

- start with `batch_size = 8`
- if CUDA OOM happens, drop to `batch_size = 4`
- if memory is clearly stable, then consider testing `batch_size = 16`

The move from `max_len = 512` to `1024` will significantly increase memory pressure and training time.

## Analysis Command

After training:

```bash
python scripts/analyze_results.py \
  --runs_dir runs_rfam_near_full_1024
```

If only Mamba is trained and CNN is omitted, partial analysis is acceptable. The current analysis flow already supports that mode and should be used as-is.

## Inference Smoke Test Commands

After training and analysis, validate the trained near-full-family v2 model with both FASTA and CSV inference.

FASTA:

```bash
python scripts/predict_rna_family.py \
  --input examples/test_sequences.fasta \
  --output predictions/test_predictions_near_full_fasta.csv \
  --model_path runs_rfam_near_full_1024/mamba_run/best_model.pt \
  --label_mapping processed_rfam_near_full_1024/label_mapping.json \
  --model_type mamba \
  --max_len 1024 \
  --batch_size 32 \
  --top_k 5 \
  --device cuda
```

CSV:

```bash
python scripts/predict_rna_family.py \
  --input examples/test_sequences.csv \
  --output predictions/test_predictions_near_full_csv.csv \
  --model_path runs_rfam_near_full_1024/mamba_run/best_model.pt \
  --label_mapping processed_rfam_near_full_1024/label_mapping.json \
  --model_type mamba \
  --max_len 1024 \
  --batch_size 32 \
  --top_k 5 \
  --device cuda
```

## Success Criteria

Dataset build success:

- selected families should be close to `4226`
- train/val/test files all exist
- `label_mapping.json` exists

Validation success:

- `validate_rfam_dataset.py` returns `PASS`

Training success:

- `best_model.pt` is generated
- `metrics.json` is generated
- `test_macro_f1` is available

Recommended interpretation:

```text
macro-F1 >= 0.90:
  minimum acceptable near-full-family v2 model

macro-F1 >= 0.95:
  good near-full-family v2 result

macro-F1 >= 0.97:
  very strong result
```

If `macro-F1 < 0.90`:

- do not change the model first
- inspect low-support families
- inspect length-bucket performance
- inspect confusion clusters
- inspect `max_samples_per_family`

## Risk Notes

Near-full-family v2 introduces a very different difficulty profile from the current `1669-family` demo model.

Main risks:

- many more low-support families will appear
- overall macro-F1 may be lower than the `1669-family` model
- `max_len = 1024` increases runtime and memory cost
- some families may be too small to support high-confidence predictions

Frontend/demo implications:

- keep `confidence`
- keep `top-k`
- keep `risk_flag`
- avoid presenting all predictions as equally reliable

## Demo Integration Guidance

If the near-full-family v2 model trains successfully and passes validation, then the next step is to switch demo defaults to:

```text
runs_rfam_near_full_1024/mamba_run/best_model.pt
processed_rfam_near_full_1024/label_mapping.json
max_len = 1024
top_k = 5
```

Do **not** modify the Flask demo in this stage.

## Immediate Next Step

The next step is:

```text
build processed_rfam_near_full_1024
-> validate it
-> only then start the WSL Mamba training run
```

Do not jump directly to demo changes before the dataset and training outputs are stabilized.
