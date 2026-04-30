# Near-Full-Family V2 Command Sheet

## Positioning

Use this runbook for the near-full-family Rfam classifier:

```text
4226 FASTA-accessible families
max_len = 1024
strict_global_dedup
```

Do not refer to this stage as a true "full 4360-family" model.

## 1. Build Dataset

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

## 2. Validate Dataset

```bash
python scripts/validate_rfam_dataset.py \
  --data_dir processed_rfam_near_full_1024
```

Validation must pass before training.

## 3. Train Mamba in WSL

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

OOM fallback order:

```text
8 -> 4
```

If memory is clearly stable:

```text
8 -> 16
```

## 4. Analyze Results

```bash
python scripts/analyze_results.py \
  --runs_dir runs_rfam_near_full_1024
```

If only Mamba is trained, partial analysis is acceptable.

## 5. FASTA Inference Smoke Test

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

## 6. CSV Inference Smoke Test

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

## 7. Success Thresholds

```text
macro-F1 >= 0.90  minimum acceptable
macro-F1 >= 0.95  good
macro-F1 >= 0.97  very good
```

If macro-F1 is below `0.90`, check:

- low-support families
- length buckets
- confusion clusters
- `max_samples_per_family`

Do not change the model structure first.

## 8. Do Not Do Yet

Do not switch the Flask demo defaults yet.

Only after:

```text
dataset build success
-> validation PASS
-> best_model.pt exists
-> metrics.json exists
-> inference smoke tests pass
```

should the demo be updated to point at:

```text
runs_rfam_near_full_1024/mamba_run/best_model.pt
processed_rfam_near_full_1024/label_mapping.json
max_len = 1024
top_k = 5
```
