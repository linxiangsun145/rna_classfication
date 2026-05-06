#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any


SEQUENCE_COLUMN_CANDIDATES = ("sequence", "seq", "rna_sequence")
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LEN = 512
DEFAULT_TOP_K = 3
KNOWN_CONFUSING_GROUPS = [
    {"RF02540", "RF02541", "RF02542", "RF02543", "RF01960"},
    {"RF04337", "RF04338", "RF04339", "RF04340"},
    {"RF04346", "RF04347"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RNA family classification inference on FASTA or CSV inputs."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA or CSV file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--label_mapping",
        type=Path,
        required=True,
        help="Path to label_mapping.json.",
    )
    parser.add_argument(
        "--model_type",
        choices=["mamba", "cnn", "gru", "bigru"],
        default="bigru",
        help="Model type to load if checkpoint config does not specify one.",
    )
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of top predictions to return.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device, for example cuda or cpu.",
    )
    return parser.parse_args()


def ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def require_dependencies() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import torch
    except ImportError as exc:
        raise ImportError(
            "Inference requires torch and pandas. Install them before running "
            "scripts/predict_rna_family.py."
        ) from exc
    return pd, torch


def resolve_device(torch: Any, device_name: str) -> Any:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, falling back to cpu")
        device_name = "cpu"
    return torch.device(device_name)


def open_text_file(file_path: Path):
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")
    return file_path.open("r", encoding="utf-8")


def clean_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).upper().split()).replace("T", "U")
    normalized = [base if base in {"A", "C", "G", "U", "N"} else "N" for base in sequence]
    return "".join(normalized)


def parse_fasta(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    header: str | None = None
    seq_lines: list[str] = []
    auto_index = 0

    def flush_record(current_header: str | None, current_lines: list[str]) -> None:
        nonlocal auto_index
        if current_header is None:
            return
        raw_sequence = "".join(current_lines)
        sequence = clean_sequence(raw_sequence)
        token = current_header.strip().split()[0] if current_header.strip() else ""
        if token:
            sequence_id = token
        else:
            auto_index += 1
            sequence_id = f"seq_{auto_index}"
        if not sequence:
            warnings.append(f"[WARN] Skipping empty sequence after cleaning: {sequence_id}")
            return
        records.append({"sequence_id": sequence_id, "sequence": sequence})

    with open_text_file(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record(header, seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

    flush_record(header, seq_lines)
    return records, warnings


def parse_csv(path: Path, pd: Any) -> tuple[list[dict[str, Any]], list[str]]:
    dataframe = pd.read_csv(path)
    sequence_column = next((col for col in SEQUENCE_COLUMN_CANDIDATES if col in dataframe.columns), None)
    if sequence_column is None:
        raise ValueError(
            f"CSV must contain one of {list(SEQUENCE_COLUMN_CANDIDATES)} columns."
        )

    id_column = None
    for candidate in ("sequence_id", "id"):
        if candidate in dataframe.columns:
            id_column = candidate
            break

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    auto_index = 0

    for _, row in dataframe.iterrows():
        auto_index += 1
        raw_id = row[id_column] if id_column is not None else None
        if pd.isna(raw_id) or str(raw_id).strip() == "":
            sequence_id = f"seq_{auto_index}"
        else:
            sequence_id = str(raw_id).strip()

        raw_sequence = "" if pd.isna(row[sequence_column]) else str(row[sequence_column])
        sequence = clean_sequence(raw_sequence)
        if not sequence:
            warnings.append(f"[WARN] Skipping empty sequence after cleaning: {sequence_id}")
            continue
        records.append({"sequence_id": sequence_id, "sequence": sequence})

    return records, warnings


def load_input_records(path: Path, pd: Any) -> tuple[list[dict[str, Any]], list[str]]:
    suffixes = path.suffixes
    if ".csv" in suffixes or path.suffix.lower() == ".csv":
        return parse_csv(path, pd)
    return parse_fasta(path)


def load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "family_to_label" in payload:
        family_to_label = {str(family): int(label) for family, label in payload["family_to_label"].items()}
    else:
        family_to_label = {str(family): int(label) for family, label in payload.items()}

    if "label_to_family" in payload:
        label_to_family = {int(label): str(family) for label, family in payload["label_to_family"].items()}
    else:
        label_to_family = {label: family for family, label in family_to_label.items()}

    return family_to_label, label_to_family


def encode_sequence(sequence: str, max_len: int) -> tuple[list[int], list[int], bool]:
    from datasets.rna_dataset import encode_sequence as dataset_encode_sequence

    truncated = len(sequence) > max_len
    input_ids, attention_mask = dataset_encode_sequence(sequence, max_len)
    return input_ids, attention_mask, truncated


def infer_num_classes(label_to_family: dict[int, str], state_dict: dict[str, Any]) -> int:
    if label_to_family:
        return len(label_to_family)
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is not None:
        return int(classifier_weight.shape[0])
    raise ValueError("Unable to infer num_classes from label mapping or checkpoint.")


def extract_state_dict_and_config(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], dict(checkpoint.get("config", {}))
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"], dict(checkpoint.get("config", {}))
        if any(key.endswith("weight") or key.endswith("bias") for key in checkpoint.keys()):
            return checkpoint, {}
    raise ValueError("Unsupported checkpoint format.")


def build_model(
    model_type: str,
    num_classes: int,
    config: dict[str, Any],
) -> Any:
    if model_type in {"gru", "bigru"}:
        from models.gru_classifier import GRUClassifier

        return GRUClassifier(
            num_classes=num_classes,
            embed_dim=int(config.get("gru_embed_dim", config.get("embed_dim", 128))),
            hidden_dim=int(config.get("gru_hidden_dim", config.get("hidden_dim", 128))),
            num_layers=int(config.get("gru_num_layers", config.get("num_layers", 1))),
            dropout=float(config.get("gru_dropout", config.get("dropout", 0.2))),
            bidirectional=bool(config.get("bidirectional", model_type == "bigru")),
        )

    if model_type == "cnn":
        from models.cnn_classifier import CNNClassifier

        return CNNClassifier(
            num_classes=num_classes,
            embedding_dim=int(config.get("embedding_dim", 128)),
            conv_channels=int(config.get("conv_channels", 128)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.2)),
        )

    if model_type == "mamba":
        try:
            from models.mamba_classifier import MambaClassifier
        except ImportError as exc:
            raise ImportError(
                "Unable to import the Mamba model. On Windows this usually means mamba-ssm "
                "is unavailable; run inference in the WSL environment used for training."
            ) from exc

        return MambaClassifier(
            num_classes=num_classes,
            d_model=int(config.get("d_model", 128)),
            n_layers=int(config.get("n_layers", 2)),
            d_state=int(config.get("d_state", 16)),
            d_conv=int(config.get("d_conv", 4)),
            expand=int(config.get("expand", 2)),
            dropout=float(config.get("dropout", 0.2)),
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def load_model(
    model_path: Path,
    model_type: str,
    num_classes: int,
    device: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict, checkpoint_config = extract_state_dict_and_config(checkpoint)
    effective_model_type = str(checkpoint_config.get("model", model_type))
    model = build_model(effective_model_type, num_classes, checkpoint_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint_config


def predict_batch(
    model: Any,
    batch_records: list[dict[str, Any]],
    max_len: int,
    top_k: int,
    device: Any,
    torch: Any,
    label_to_family: dict[int, str],
) -> list[dict[str, Any]]:
    encoded_input_ids: list[list[int]] = []
    encoded_attention_masks: list[list[int]] = []
    prepared_records: list[dict[str, Any]] = []

    for record in batch_records:
        input_ids, attention_mask, was_truncated = encode_sequence(record["sequence"], max_len)
        encoded_input_ids.append(input_ids)
        encoded_attention_masks.append(attention_mask)
        prepared_records.append(
            {
                **record,
                "seq_len": len(record["sequence"]),
                "was_truncated": was_truncated,
            }
        )

    with torch.no_grad():
        input_tensor = torch.tensor(encoded_input_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(encoded_attention_masks, dtype=torch.long, device=device)
        logits = model(input_ids=input_tensor, attention_mask=mask_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        k = max(top_k, 3)
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=-1)

    predictions: list[dict[str, Any]] = []
    for record, probs_row, indices_row in zip(prepared_records, top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        top_predictions = []
        for label, prob in zip(indices_row, probs_row):
            top_predictions.append(
                {
                    "label": int(label),
                    "family": label_to_family.get(int(label), f"UNKNOWN_{label}"),
                    "prob": float(prob),
                }
            )

        result = {
            **record,
            "predicted_label": top_predictions[0]["label"],
            "predicted_family": top_predictions[0]["family"],
            "confidence": top_predictions[0]["prob"],
            "top_predictions": top_predictions,
        }
        predictions.append(result)

    return predictions


def is_confusing_family_group(family_id: str) -> bool:
    return any(family_id in group for group in KNOWN_CONFUSING_GROUPS)


def assign_risk_flags(prediction: dict[str, Any], max_len: int) -> str:
    flags: list[str] = []
    if float(prediction["confidence"]) < 0.70:
        flags.append("low_confidence")
    if int(prediction["seq_len"]) > max_len:
        flags.append("truncated_sequence")
    if is_confusing_family_group(str(prediction["predicted_family"])):
        flags.append("confusing_family_group")
    return ";".join(flags) if flags else "ok"


def write_predictions(output_path: Path, predictions: list[dict[str, Any]], top_k: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    k = max(top_k, 3)
    fieldnames = [
        "sequence_id",
        "sequence",
        "seq_len",
        "was_truncated",
        "predicted_label",
        "predicted_family",
        "confidence",
    ]
    for rank in range(1, k + 1):
        fieldnames.extend([f"top{rank}_family", f"top{rank}_prob"])
    fieldnames.append("risk_flag")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            row = {
                "sequence_id": prediction["sequence_id"],
                "sequence": prediction["sequence"],
                "seq_len": prediction["seq_len"],
                "was_truncated": prediction["was_truncated"],
                "predicted_label": prediction["predicted_label"],
                "predicted_family": prediction["predicted_family"],
                "confidence": prediction["confidence"],
            }
            top_predictions = prediction["top_predictions"]
            for rank in range(1, k + 1):
                if rank <= len(top_predictions):
                    row[f"top{rank}_family"] = top_predictions[rank - 1]["family"]
                    row[f"top{rank}_prob"] = top_predictions[rank - 1]["prob"]
                else:
                    row[f"top{rank}_family"] = ""
                    row[f"top{rank}_prob"] = ""
            row["risk_flag"] = prediction["risk_flag"]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    ensure_project_root_on_path()

    try:
        pd, torch = require_dependencies()
        device = resolve_device(torch, args.device)
        _, label_to_family = load_label_mapping(args.label_mapping)
        records, warnings = load_input_records(args.input, pd)
        if not records:
            raise ValueError("No valid sequences were found in the input file.")

        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict, checkpoint_config = extract_state_dict_and_config(checkpoint)
        num_classes = infer_num_classes(label_to_family, state_dict)
        model_type = str(checkpoint_config.get("model", args.model_type))
        model = build_model(model_type, num_classes, checkpoint_config)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        effective_max_len = int(checkpoint_config.get("max_len", args.max_len))
        all_predictions: list[dict[str, Any]] = []
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            batch_predictions = predict_batch(
                model=model,
                batch_records=batch_records,
                max_len=effective_max_len,
                top_k=args.top_k,
                device=device,
                torch=torch,
                label_to_family=label_to_family,
            )
            for prediction in batch_predictions:
                prediction["risk_flag"] = assign_risk_flags(prediction, effective_max_len)
            all_predictions.extend(batch_predictions)

        write_predictions(args.output, all_predictions, args.top_k)

        for warning in warnings:
            print(warning)
        print(f"[OK] Loaded {len(records)} sequences from {args.input}")
        print(f"[OK] Model type = {model_type}")
        print(f"[OK] Device = {device}")
        print(f"[OK] Wrote predictions to {args.output}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
