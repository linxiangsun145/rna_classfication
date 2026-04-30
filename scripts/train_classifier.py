#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RNA sequence classifiers on processed Rfam CSV data."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("processed"),
        help="Directory containing processed train/val/test CSV files.",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "mamba"],
        default="cnn",
        help="Model backbone to train.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs"),
        help="Directory to save checkpoints and metrics.",
    )
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="CNN embedding dimension.",
    )
    parser.add_argument(
        "--conv_channels",
        type=int,
        default=128,
        help="CNN convolution output channels.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="CNN kernel size.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Mamba hidden size.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of Mamba blocks.",
    )
    parser.add_argument(
        "--d_state",
        type=int,
        default=16,
        help="Mamba state size.",
    )
    parser.add_argument(
        "--d_conv",
        type=int,
        default=4,
        help="Mamba local convolution width.",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=2,
        help="Mamba expansion factor.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--loss_type",
        choices=["ce", "weighted_ce", "focal", "class_balanced_focal"],
        default="ce",
        help="Training loss type.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for cross-entropy variants.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss variants.",
    )
    parser.add_argument(
        "--cb_beta",
        type=float,
        default=0.9999,
        help="Beta parameter for class-balanced focal loss.",
    )
    parser.add_argument(
        "--max_class_weight",
        type=float,
        default=10.0,
        help="Upper bound applied to computed class weights.",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Use WeightedRandomSampler for the train loader.",
    )
    parser.add_argument(
        "--sampler_power",
        type=float,
        default=0.5,
        help="Power used in inverse-frequency train sampling weights.",
    )
    parser.add_argument(
        "--classifier_head",
        choices=["linear", "mlp"],
        default="linear",
        help="Classifier head type for the Mamba model.",
    )
    parser.add_argument(
        "--head_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for the optional MLP classifier head.",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="Dropout used inside the optional MLP classifier head.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)


def ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def require_core_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise ImportError(
            "Training requires torch, pandas, and scikit-learn. Install them before running "
            "scripts/train_classifier.py."
        ) from exc

    return (
        torch,
        pd,
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )


def resolve_device(torch: Any, device_name: str) -> Any:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, falling back to cpu")
        device_name = "cpu"
    return torch.device(device_name)


def create_dataloaders(
    args: argparse.Namespace, torch: Any
) -> tuple[dict[str, Any], dict[str, int], dict[str, Counter]]:
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from datasets.rna_dataset import RNADataset, load_label_mapping

    datasets = {
        "train": RNADataset(args.data_dir / "train.csv", max_len=args.max_len),
        "val": RNADataset(args.data_dir / "val.csv", max_len=args.max_len),
        "test": RNADataset(args.data_dir / "test.csv", max_len=args.max_len),
    }
    label_mapping = load_label_mapping(args.data_dir)
    label_counts = {
        split_name: Counter(int(value) for value in dataset.dataframe["label"].tolist())
        for split_name, dataset in datasets.items()
    }

    train_sampler = None
    if args.use_weighted_sampler:
        train_counts = label_counts["train"]
        sample_weights = []
        for label in datasets["train"].dataframe["label"].tolist():
            count = max(train_counts[int(label)], 1)
            sample_weights.append(1.0 / (float(count) ** args.sampler_power))
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    dataloaders = {}
    for split_name, dataset in datasets.items():
        is_train = split_name == "train"
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(is_train and train_sampler is None),
            sampler=train_sampler if is_train else None,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    return dataloaders, label_mapping, label_counts


def build_model(args: argparse.Namespace, num_classes: int, torch: Any) -> Any:
    if args.model == "cnn":
        from models.cnn_classifier import CNNClassifier

        model = CNNClassifier(
            num_classes=num_classes,
            embedding_dim=args.embedding_dim,
            conv_channels=args.conv_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
        return model

    if args.model == "mamba":
        try:
            from models.mamba_classifier import MambaClassifier
        except ImportError as exc:
            raise ImportError(
                "Unable to import the Mamba model. Install mamba-ssm before using "
                "--model mamba."
            ) from exc

        return MambaClassifier(
            num_classes=num_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            dropout=args.dropout,
            classifier_head=args.classifier_head,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
        )

    raise ValueError(f"Unsupported model type: {args.model}")


def compute_class_weights(
    label_counts: Counter,
    num_classes: int,
    max_class_weight: float,
) -> List[float]:
    total_samples = max(sum(label_counts.values()), 1)
    weights: List[float] = []
    for label_idx in range(num_classes):
        count = max(int(label_counts.get(label_idx, 0)), 1)
        weight = total_samples / (num_classes * count)
        weights.append(min(float(weight), max_class_weight))
    mean_weight = sum(weights) / max(len(weights), 1)
    if mean_weight > 0:
        weights = [weight / mean_weight for weight in weights]
    return weights


def compute_class_balanced_weights(
    label_counts: Counter,
    num_classes: int,
    beta: float,
    max_class_weight: float,
) -> List[float]:
    beta = min(max(beta, 0.0), 0.9999999999)
    weights: List[float] = []
    for label_idx in range(num_classes):
        count = max(int(label_counts.get(label_idx, 0)), 1)
        effective_num = 1.0 - (beta ** count)
        if effective_num <= 0.0:
            weight = max_class_weight
        else:
            weight = (1.0 - beta) / effective_num
        weights.append(min(float(weight), max_class_weight))
    mean_weight = sum(weights) / max(len(weights), 1)
    if mean_weight > 0:
        weights = [weight / mean_weight for weight in weights]
    return weights


class FocalLoss:
    def __init__(
        self,
        torch: Any,
        gamma: float = 2.0,
        class_weights: Any = None,
        reduction: str = "mean",
    ) -> None:
        self.torch = torch
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def __call__(self, logits: Any, targets: Any) -> Any:
        log_probs = self.torch.nn.functional.log_softmax(logits, dim=-1)
        gathered_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        probs = gathered_log_probs.exp()
        focal_factor = (1.0 - probs).clamp(min=0.0) ** self.gamma
        loss = -focal_factor * gathered_log_probs

        if self.class_weights is not None:
            sample_weights = self.class_weights.gather(0, targets)
            loss = loss * sample_weights

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_cross_entropy_loss(
    torch: Any,
    class_weights: Any = None,
    label_smoothing: float = 0.0,
) -> Any:
    kwargs = {}
    if class_weights is not None:
        kwargs["weight"] = class_weights

    if label_smoothing > 0.0:
        try:
            return torch.nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                **kwargs,
            )
        except TypeError:
            print(
                "[WARN] Current PyTorch does not support label_smoothing in "
                "CrossEntropyLoss, falling back to label_smoothing=0.0"
            )

    return torch.nn.CrossEntropyLoss(**kwargs)


def build_criterion(
    args: argparse.Namespace,
    label_counts: Counter,
    num_classes: int,
    device: Any,
    torch: Any,
) -> Any:
    class_weights = None
    if args.loss_type in {"weighted_ce", "focal", "class_balanced_focal"}:
        if args.loss_type == "class_balanced_focal":
            class_weight_values = compute_class_balanced_weights(
                label_counts=label_counts,
                num_classes=num_classes,
                beta=args.cb_beta,
                max_class_weight=args.max_class_weight,
            )
        else:
            class_weight_values = compute_class_weights(
                label_counts=label_counts,
                num_classes=num_classes,
                max_class_weight=args.max_class_weight,
            )
        class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)

    if args.loss_type == "ce":
        return build_cross_entropy_loss(
            torch=torch,
            class_weights=None,
            label_smoothing=args.label_smoothing,
        )
    if args.loss_type == "weighted_ce":
        return build_cross_entropy_loss(
            torch=torch,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )
    if args.loss_type in {"focal", "class_balanced_focal"}:
        if args.label_smoothing > 0.0:
            print("[WARN] label_smoothing is ignored for focal loss variants")
        return FocalLoss(
            torch=torch,
            gamma=args.focal_gamma,
            class_weights=class_weights,
            reduction="mean",
        )

    raise ValueError(f"Unsupported loss type: {args.loss_type}")


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    torch: Any,
    accuracy_score: Any,
    f1_score: Any,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=-1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    accuracy = float(accuracy_score(all_labels, all_predictions))
    macro_f1 = float(f1_score(all_labels, all_predictions, average="macro"))
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def evaluate(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    torch: Any,
    accuracy_score: Any,
    f1_score: Any,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            predictions = logits.argmax(dim=-1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    accuracy = float(accuracy_score(all_labels, all_predictions))
    macro_f1 = float(f1_score(all_labels, all_predictions, average="macro"))
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def predict_dataset(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    torch: Any,
    accuracy_score: Any,
    f1_score: Any,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_seq_lens: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=-1)

            total_loss += float(loss.item()) * labels.size(0)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_seq_lens.extend(attention_mask.sum(dim=-1).detach().cpu().tolist())

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    accuracy = float(accuracy_score(all_labels, all_predictions))
    macro_f1 = float(f1_score(all_labels, all_predictions, average="macro"))
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "predictions": all_predictions,
        "seq_lens": all_seq_lens,
    }


def save_metrics(output_dir: Path, metrics: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in config.items():
        normalized[key] = str(value) if isinstance(value, Path) else value
    return normalized


def build_id_to_family(label_mapping: Dict[str, int]) -> Dict[int, str]:
    return {int(label): family for family, label in label_mapping.items()}


def save_json(output_path: Path, payload: Dict[str, Any]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_confusion_matrix(
    run_dir: Path,
    labels: List[int],
    predictions: List[int],
    label_mapping: Dict[str, int],
    confusion_matrix_fn: Any,
    pd: Any,
) -> None:
    ordered_pairs = sorted(label_mapping.items(), key=lambda item: item[1])
    family_names = [family for family, _ in ordered_pairs]
    label_ids = [label for _, label in ordered_pairs]
    matrix = confusion_matrix_fn(labels, predictions, labels=label_ids)
    matrix_df = pd.DataFrame(matrix, index=family_names, columns=family_names)
    matrix_df.to_csv(run_dir / "confusion_matrix.csv", encoding="utf-8")


def save_per_class_metrics(
    run_dir: Path,
    labels: List[int],
    predictions: List[int],
    label_mapping: Dict[str, int],
    precision_recall_fscore_support_fn: Any,
) -> None:
    ordered_pairs = sorted(label_mapping.items(), key=lambda item: item[1])
    family_names = [family for family, _ in ordered_pairs]
    label_ids = [label for _, label in ordered_pairs]
    precision, recall, f1, support = precision_recall_fscore_support_fn(
        labels,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    payload = {}
    for family, p, r, f, s in zip(family_names, precision, recall, f1, support):
        payload[family] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
    save_json(run_dir / "per_class_metrics.json", payload)


def get_length_bucket(seq_len: int) -> str:
    if seq_len < 100:
        return "short"
    if seq_len < 300:
        return "medium"
    return "long"


def save_length_bucket_metrics(
    run_dir: Path,
    labels: List[int],
    predictions: List[int],
    seq_lens: List[int],
    accuracy_score: Any,
    f1_score: Any,
) -> None:
    bucket_records: Dict[str, Dict[str, List[int]]] = {
        "short": {"labels": [], "predictions": []},
        "medium": {"labels": [], "predictions": []},
        "long": {"labels": [], "predictions": []},
    }

    for label, prediction, seq_len in zip(labels, predictions, seq_lens):
        bucket = get_length_bucket(int(seq_len))
        bucket_records[bucket]["labels"].append(int(label))
        bucket_records[bucket]["predictions"].append(int(prediction))

    payload = {}
    for bucket_name, bucket_data in bucket_records.items():
        bucket_labels = bucket_data["labels"]
        bucket_predictions = bucket_data["predictions"]
        if bucket_labels:
            payload[bucket_name] = {
                "count": int(len(bucket_labels)),
                "accuracy": float(accuracy_score(bucket_labels, bucket_predictions)),
                "macro_f1": float(
                    f1_score(bucket_labels, bucket_predictions, average="macro")
                ),
            }
        else:
            payload[bucket_name] = {
                "count": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
            }

    save_json(run_dir / "length_bucket_metrics.json", payload)


def build_compare_report(output_dir: Path) -> None:
    cnn_metrics_path = output_dir / "cnn_run" / "metrics.json"
    mamba_metrics_path = output_dir / "mamba_run" / "metrics.json"
    if not cnn_metrics_path.exists() or not mamba_metrics_path.exists():
        return

    with cnn_metrics_path.open("r", encoding="utf-8") as handle:
        cnn_metrics = json.load(handle)
    with mamba_metrics_path.open("r", encoding="utf-8") as handle:
        mamba_metrics = json.load(handle)

    report = {
        "cnn": {
            "test_acc": cnn_metrics.get("test_accuracy", 0.0),
            "test_macro_f1": cnn_metrics.get("test_macro_f1", 0.0),
        },
        "mamba": {
            "test_acc": mamba_metrics.get("test_accuracy", 0.0),
            "test_macro_f1": mamba_metrics.get("test_macro_f1", 0.0),
        },
    }
    report["difference"] = {
        "acc_diff": float(report["mamba"]["test_acc"] - report["cnn"]["test_acc"]),
        "f1_diff": float(
            report["mamba"]["test_macro_f1"] - report["cnn"]["test_macro_f1"]
        ),
    }
    save_json(output_dir / "compare_report.json", report)


def run_training(args: argparse.Namespace) -> None:
    ensure_project_root_on_path()
    (
        torch,
        pd,
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    ) = require_core_dependencies()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataloaders, label_mapping, label_counts = create_dataloaders(args, torch)
    num_classes = len(label_mapping)
    split_sizes = {
        "train_size": int(len(dataloaders["train"].dataset)),
        "val_size": int(len(dataloaders["val"].dataset)),
        "test_size": int(len(dataloaders["test"].dataset)),
    }
    device = resolve_device(torch, args.device)

    model = build_model(args, num_classes, torch).to(device)
    criterion = build_criterion(
        args=args,
        label_counts=label_counts["train"],
        num_classes=num_classes,
        device=device,
        torch=torch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = args.output_dir / f"{args.model}_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_dir / "best_model.pt"
    normalized_config = normalize_config(vars(args))
    experiment_config = {
        "model": args.model,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "num_classes": num_classes,
        **split_sizes,
    }
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    best_val_macro_f1 = float("-inf")
    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            torch=torch,
            accuracy_score=accuracy_score,
            f1_score=f1_score,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            torch=torch,
            accuracy_score=accuracy_score,
            f1_score=f1_score,
        )

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f}"
        )
        print(
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_acc"].append(float(train_metrics["accuracy"]))
        history["train_macro_f1"].append(float(train_metrics["macro_f1"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))

        is_better = (
            val_metrics["macro_f1"] > best_val_macro_f1
            or (
                val_metrics["macro_f1"] == best_val_macro_f1
                and val_metrics["accuracy"] > best_val_accuracy
            )
        )
        if is_better:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": normalized_config,
                    "label_mapping": label_mapping,
                },
                best_model_path,
            )
            print("[OK] Saved new best model")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_outputs = predict_dataset(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        torch=torch,
        accuracy_score=accuracy_score,
        f1_score=f1_score,
    )

    save_confusion_matrix(
        run_dir=run_dir,
        labels=test_outputs["labels"],
        predictions=test_outputs["predictions"],
        label_mapping=label_mapping,
        confusion_matrix_fn=confusion_matrix,
        pd=pd,
    )
    save_per_class_metrics(
        run_dir=run_dir,
        labels=test_outputs["labels"],
        predictions=test_outputs["predictions"],
        label_mapping=label_mapping,
        precision_recall_fscore_support_fn=precision_recall_fscore_support,
    )
    save_length_bucket_metrics(
        run_dir=run_dir,
        labels=test_outputs["labels"],
        predictions=test_outputs["predictions"],
        seq_lens=test_outputs["seq_lens"],
        accuracy_score=accuracy_score,
        f1_score=f1_score,
    )

    print(f"[RESULT] test_loss={test_outputs['loss']:.4f}")
    print(f"[RESULT] test_acc={test_outputs['accuracy']:.4f}")
    print(f"[RESULT] test_macro_f1={test_outputs['macro_f1']:.4f}")

    metrics = {
        "model": args.model,
        **experiment_config,
        "config": normalized_config,
        "label_mapping": label_mapping,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_val_macro_f1,
        "test_loss": test_outputs["loss"],
        "test_accuracy": test_outputs["accuracy"],
        "test_macro_f1": test_outputs["macro_f1"],
        "history": history,
    }
    save_metrics(run_dir, metrics)
    build_compare_report(args.output_dir)


def main() -> None:
    args = parse_args()
    try:
        run_training(args)
    except ImportError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
