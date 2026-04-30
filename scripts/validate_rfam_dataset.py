#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REQUIRED_FILES = [
    "train.csv",
    "val.csv",
    "test.csv",
    "label_mapping.json",
    "dataset_summary.json",
]
REQUIRED_COLUMNS = ["sequence_id", "family", "label", "sequence", "seq_len"]
VALID_BASES = set("ACGUN")
SPLIT_NAMES = ["train", "val", "test"]


@dataclass
class ValidationReport:
    errors: int = 0
    warnings: int = 0

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def error(self, message: str) -> None:
        self.errors += 1
        print(f"[ERROR] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate processed Rfam RNA classification dataset files."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("processed"),
        help="Directory containing processed dataset files.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="Maximum number of examples to print for each validation issue.",
    )
    return parser.parse_args()


def check_required_files(data_dir: Path, report: ValidationReport) -> bool:
    all_present = True
    for filename in REQUIRED_FILES:
        file_path = data_dir / filename
        if file_path.exists():
            report.ok(f"{filename} exists")
        else:
            report.error(f"Missing required file: {filename}")
            all_present = False
    return all_present


def load_split(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def load_json(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_required_columns(
    split_name: str, dataframe: pd.DataFrame, report: ValidationReport
) -> bool:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        report.error(f"{split_name}.csv is missing required columns: {missing}")
        return False
    report.ok(f"Required columns present in {split_name}.csv")
    return True


def check_nulls(
    split_name: str, dataframe: pd.DataFrame, report: ValidationReport
) -> None:
    null_counts = dataframe[REQUIRED_COLUMNS].isnull().sum()
    found_issue = False
    for column, count in null_counts.items():
        if int(count) > 0:
            report.error(f"Found {int(count)} null value(s) in {split_name}.csv column '{column}'")
            found_issue = True
    if not found_issue:
        report.ok(f"No null values in required columns for {split_name}.csv")


def _sequence_issue_examples(
    dataframe: pd.DataFrame, row_indices: List[int], max_examples: int
) -> List[str]:
    examples: List[str] = []
    for row_index in row_indices[:max_examples]:
        row = dataframe.loc[row_index]
        sequence = str(row["sequence"])
        digest = hashlib.md5(sequence.encode("utf-8")).hexdigest()[:10]
        snippet = sequence[:30]
        examples.append(
            f"sequence_id={row['sequence_id']}, family={row['family']}, "
            f"seq_len={row['seq_len']}, hash={digest}, prefix={snippet}"
        )
    return examples


def check_sequence_charset(
    split_name: str,
    dataframe: pd.DataFrame,
    report: ValidationReport,
    max_examples: int,
) -> None:
    invalid_indices: List[int] = []

    for row_index, sequence in dataframe["sequence"].astype(str).items():
        invalid_chars = sorted(set(sequence) - VALID_BASES)
        if invalid_chars:
            invalid_indices.append(row_index)

    if invalid_indices:
        report.error(
            f"Found {len(invalid_indices)} sequence(s) with invalid characters in {split_name}.csv"
        )
        for example in _sequence_issue_examples(dataframe, invalid_indices, max_examples):
            print(f"  - {example}")
    else:
        report.ok(f"All sequences contain only A/C/G/U/N in {split_name}.csv")


def check_seq_len_consistency(
    split_name: str,
    dataframe: pd.DataFrame,
    report: ValidationReport,
    max_examples: int,
) -> None:
    actual_lengths = dataframe["sequence"].astype(str).str.len()
    mismatch_mask = actual_lengths != dataframe["seq_len"]
    mismatch_indices = dataframe.index[mismatch_mask].tolist()

    if mismatch_indices:
        report.error(
            f"Found {len(mismatch_indices)} seq_len mismatch row(s) in {split_name}.csv"
        )
        for row_index in mismatch_indices[:max_examples]:
            row = dataframe.loc[row_index]
            print(
                "  - "
                f"sequence_id={row['sequence_id']}, family={row['family']}, "
                f"stored_seq_len={row['seq_len']}, actual_len={len(str(row['sequence']))}"
            )
    else:
        report.ok(f"seq_len matches len(sequence) in {split_name}.csv")


def check_duplicates_within_split(
    split_name: str,
    dataframe: pd.DataFrame,
    report: ValidationReport,
    max_examples: int,
) -> None:
    duplicate_mask = dataframe["sequence"].duplicated(keep=False)
    duplicates = dataframe.loc[duplicate_mask, ["sequence_id", "family", "sequence"]]

    if duplicates.empty:
        report.ok(f"No duplicate sequences within {split_name}.csv")
        return

    unique_duplicate_sequences = duplicates["sequence"].nunique()
    report.error(
        f"Found {unique_duplicate_sequences} duplicated sequence(s) within {split_name}.csv"
    )
    sampled = duplicates.drop_duplicates(subset=["sequence"]).head(max_examples)
    for _, row in sampled.iterrows():
        digest = hashlib.md5(row["sequence"].encode("utf-8")).hexdigest()[:10]
        print(
            "  - "
            f"sequence_id={row['sequence_id']}, family={row['family']}, "
            f"hash={digest}, prefix={row['sequence'][:30]}"
        )


def check_duplicates_across_splits(
    splits: Dict[str, pd.DataFrame],
    report: ValidationReport,
    max_examples: int,
) -> None:
    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]

    for left_name, right_name in split_pairs:
        left_sequences = set(splits[left_name]["sequence"].astype(str))
        right_sequences = set(splits[right_name]["sequence"].astype(str))
        overlap = sorted(left_sequences & right_sequences)
        if overlap:
            report.error(
                f"Found {len(overlap)} overlapping sequence(s) between {left_name} and {right_name}"
            )
            for sequence in overlap[:max_examples]:
                digest = hashlib.md5(sequence.encode("utf-8")).hexdigest()[:10]
                print(f"  - hash={digest}, prefix={sequence[:40]}")
        else:
            report.ok(f"No cross-split leakage between {left_name} and {right_name}")


def check_label_mapping_consistency(
    splits: Dict[str, pd.DataFrame],
    label_mapping: Dict[str, int],
    report: ValidationReport,
) -> None:
    observed_pairs: Dict[str, Dict[str, int]] = {}

    for split_name, dataframe in splits.items():
        family_to_labels = dataframe.groupby("family")["label"].nunique().to_dict()
        for family, unique_count in family_to_labels.items():
            if unique_count > 1:
                report.error(
                    f"Family {family} maps to multiple labels within {split_name}.csv"
                )

        split_mapping = dataframe.groupby("family")["label"].first().to_dict()
        observed_pairs[split_name] = {
            family: int(label) for family, label in split_mapping.items()
        }

    merged_mapping: Dict[str, set] = {}
    for split_mapping in observed_pairs.values():
        for family, label in split_mapping.items():
            merged_mapping.setdefault(family, set()).add(label)

    inconsistent_families = {
        family: labels for family, labels in merged_mapping.items() if len(labels) > 1
    }
    if inconsistent_families:
        for family, labels in inconsistent_families.items():
            report.error(f"Family {family} has inconsistent labels across splits: {sorted(labels)}")
    else:
        report.ok("family -> label mapping is consistent across splits")

    normalized_json_mapping = {str(family): int(label) for family, label in label_mapping.items()}
    for family, labels in merged_mapping.items():
        observed_label = next(iter(labels))
        if family not in normalized_json_mapping:
            report.error(f"Family {family} exists in CSV files but is missing from label_mapping.json")
        elif normalized_json_mapping[family] != observed_label:
            report.error(
                f"Family {family} has label {observed_label} in CSV files but "
                f"{normalized_json_mapping[family]} in label_mapping.json"
            )

    extra_families = sorted(set(normalized_json_mapping) - set(merged_mapping))
    if extra_families:
        report.warn(
            "label_mapping.json contains families not present in CSV files: "
            f"{extra_families}"
        )
    else:
        report.ok("label_mapping.json matches observed family labels")


def print_class_distribution(splits: Dict[str, pd.DataFrame], report: ValidationReport) -> None:
    all_families = sorted(
        set().union(*[set(dataframe["family"].astype(str)) for dataframe in splits.values()])
    )
    total_counts = {family: 0 for family in all_families}

    print("=== Class Distribution ===")
    for split_name in SPLIT_NAMES:
        counts = splits[split_name]["family"].value_counts().to_dict()
        print(f"{split_name}:")
        for family in all_families:
            count = int(counts.get(family, 0))
            total_counts[family] += count
            marker = " [MISSING]" if count == 0 else ""
            print(f"  {family}: {count}{marker}")
            if count == 0:
                report.warn(f"Family {family} is missing from {split_name}.csv")

    print("all:")
    for family in all_families:
        print(f"  {family}: {total_counts[family]}")


def check_summary_consistency(
    splits: Dict[str, pd.DataFrame],
    label_mapping: Dict[str, int],
    summary: dict,
    report: ValidationReport,
) -> None:
    summary_splits = summary.get("splits", {})
    summary_totals = summary.get("totals", {})
    summary_families = set(summary.get("families", []))
    label_families = set(label_mapping.keys())

    if summary_families != label_families:
        report.error(
            "Family set mismatch between dataset_summary.json and label_mapping.json: "
            f"summary_only={sorted(summary_families - label_families)}, "
            f"label_only={sorted(label_families - summary_families)}"
        )
    else:
        report.ok("Family sets match between dataset_summary.json and label_mapping.json")

    actual_total = 0
    for split_name, dataframe in splits.items():
        actual_count = int(len(dataframe))
        actual_total += actual_count
        summary_count = int(summary_splits.get(split_name, {}).get("total", -1))
        if actual_count != summary_count:
            report.error(
                f"{split_name} total mismatch: summary={summary_count}, actual={actual_count}"
            )
        else:
            report.ok(f"{split_name} total matches dataset_summary.json")

        actual_per_family = dataframe["family"].value_counts().to_dict()
        summary_per_family = summary_splits.get(split_name, {}).get("per_family", {})
        families = sorted(set(actual_per_family) | set(summary_per_family))
        mismatches = []
        for family in families:
            actual_family_count = int(actual_per_family.get(family, 0))
            summary_family_count = int(summary_per_family.get(family, 0))
            if actual_family_count != summary_family_count:
                mismatches.append(
                    f"{family}: summary={summary_family_count}, actual={actual_family_count}"
                )
        if mismatches:
            report.error(
                f"{split_name} per-family count mismatch with dataset_summary.json: {mismatches}"
            )
        else:
            report.ok(f"{split_name} per-family counts match dataset_summary.json")

    summary_total = int(summary_totals.get("total_samples", -1))
    if actual_total != summary_total:
        report.error(
            f"Total sample count mismatch: summary={summary_total}, actual={actual_total}"
        )
    else:
        report.ok("Total sample count matches dataset_summary.json")


def print_length_stats(splits: Dict[str, pd.DataFrame]) -> None:
    print("=== Length Statistics ===")
    combined_lengths: List[int] = []
    for split_name in SPLIT_NAMES:
        lengths = splits[split_name]["sequence"].astype(str).str.len()
        combined_lengths.extend(lengths.tolist())
        median_value = float(lengths.median()) if not lengths.empty else 0.0
        mean_value = float(lengths.mean()) if not lengths.empty else 0.0
        min_value = int(lengths.min()) if not lengths.empty else 0
        max_value = int(lengths.max()) if not lengths.empty else 0
        print(
            f"{split_name}: min={min_value}, max={max_value}, "
            f"mean={mean_value:.4f}, median={median_value:.4f}"
        )

    if combined_lengths:
        combined_series = pd.Series(combined_lengths)
        print(
            f"all: min={int(combined_series.min())}, max={int(combined_series.max())}, "
            f"mean={float(combined_series.mean()):.4f}, median={float(combined_series.median()):.4f}"
        )
    else:
        print("all: min=0, max=0, mean=0.0000, median=0.0000")


def main() -> None:
    args = parse_args()
    report = ValidationReport()

    if not check_required_files(args.data_dir, report):
        print(
            f"[FAIL] Dataset validation failed with {report.errors} error(s) and "
            f"{report.warnings} warning(s)"
        )
        sys.exit(1)

    splits: Dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_NAMES:
        split_path = args.data_dir / f"{split_name}.csv"
        dataframe = load_split(split_path)
        if not check_required_columns(split_name, dataframe, report):
            continue
        splits[split_name] = dataframe

    if set(splits) != set(SPLIT_NAMES):
        print(
            f"[FAIL] Dataset validation failed with {report.errors} error(s) and "
            f"{report.warnings} warning(s)"
        )
        sys.exit(1)

    label_mapping = load_json(args.data_dir / "label_mapping.json")
    summary = load_json(args.data_dir / "dataset_summary.json")

    for split_name in SPLIT_NAMES:
        dataframe = splits[split_name]
        check_nulls(split_name, dataframe, report)
        check_sequence_charset(split_name, dataframe, report, args.max_examples)
        check_seq_len_consistency(split_name, dataframe, report, args.max_examples)
        check_duplicates_within_split(split_name, dataframe, report, args.max_examples)

    check_duplicates_across_splits(splits, report, args.max_examples)
    check_label_mapping_consistency(splits, label_mapping, report)
    print_class_distribution(splits, report)
    check_summary_consistency(splits, label_mapping, summary, report)
    print_length_stats(splits)

    if report.errors > 0:
        print(
            f"[FAIL] Dataset validation failed with {report.errors} error(s) and "
            f"{report.warnings} warning(s)"
        )
        sys.exit(1)

    print(
        f"[PASS] Dataset validation passed with {report.errors} error(s) and "
        f"{report.warnings} warning(s)"
    )


if __name__ == "__main__":
    main()
