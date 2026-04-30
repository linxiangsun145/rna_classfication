#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_FAMILIES = ["RF00005", "RF00009", "RF00001", "RF00002", "RF00004"]
MIN_LEN = 10
MAX_LEN = 1000
MAX_SAMPLES_PER_CLASS = 2000
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
VALID_BASES = {"A", "C", "G", "U", "N"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an RNA classification dataset from Rfam FASTA files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing FASTA files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("processed"),
        help="Directory to save processed CSV and JSON files.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=TARGET_FAMILIES,
        help="Target Rfam family IDs to include.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=MIN_LEN,
        help="Minimum allowed sequence length after cleaning.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=MAX_LEN,
        help="Maximum allowed sequence length after cleaning.",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=MAX_SAMPLES_PER_CLASS,
        help="Maximum number of samples kept for each class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used for sampling and splitting.",
    )
    return parser.parse_args()


def parse_fasta(file_path: Path, family: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: str | None = None
    seq_lines: List[str] = []
    generated_count = 0

    def flush_record(current_header: str | None, current_lines: List[str]) -> None:
        nonlocal generated_count
        if current_header is None:
            return
        raw_sequence = "".join(current_lines)
        token = current_header.strip().split()[0] if current_header.strip() else ""
        if token:
            sequence_id = token
        else:
            generated_count += 1
            sequence_id = f"{family}_auto_{generated_count:06d}"
        records.append((sequence_id, raw_sequence))

    with file_path.open("r", encoding="utf-8") as handle:
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
    return records


def clean_sequence(sequence: str) -> str:
    sequence = "".join(sequence.upper().split()).replace("T", "U")
    cleaned = [base for base in sequence if base in VALID_BASES]
    return "".join(cleaned)


def resolve_fasta_path(input_dir: Path, family: str) -> Path:
    candidates = [input_dir / f"{family}.fa", input_dir / f"{family}.fasta"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find FASTA file for family {family} in {input_dir}. "
        f"Expected one of: {', '.join(str(path.name) for path in candidates)}"
    )


def load_family_sequences(
    input_dir: Path,
    families: Iterable[str],
    min_len: int,
    max_len: int,
) -> Tuple[List[dict], Dict[str, dict]]:
    records: List[dict] = []
    stats: Dict[str, dict] = {}
    global_unique_counter = 0

    for family in families:
        fasta_path = resolve_fasta_path(input_dir, family)
        parsed_records = parse_fasta(fasta_path, family)
        raw_count = len(parsed_records)
        cleaned_count = 0
        dropped_empty = 0
        dropped_length = 0

        for original_id, raw_sequence in parsed_records:
            cleaned_sequence = clean_sequence(raw_sequence)
            if not cleaned_sequence:
                dropped_empty += 1
                continue

            seq_len = len(cleaned_sequence)
            if seq_len < min_len or seq_len > max_len:
                dropped_length += 1
                continue

            global_unique_counter += 1
            cleaned_count += 1
            sequence_id = original_id or f"{family}_seq_{global_unique_counter:06d}"
            records.append(
                {
                    "sequence_id": sequence_id,
                    "family": family,
                    "sequence": cleaned_sequence,
                    "seq_len": seq_len,
                }
            )

        stats[family] = {
            "file_path": str(fasta_path),
            "raw_count": raw_count,
            "cleaned_count": cleaned_count,
            "dropped_empty": dropped_empty,
            "dropped_length": dropped_length,
        }

    return records, stats


def deduplicate_records(records: List[dict]) -> Tuple[List[dict], dict]:
    unique_records: List[dict] = []
    seen_sequences: Dict[str, dict] = {}
    duplicate_count = 0
    cross_family_conflicts = 0
    cross_family_examples: List[dict] = []

    for record in records:
        existing = seen_sequences.get(record["sequence"])
        if existing is None:
            seen_sequences[record["sequence"]] = record
            unique_records.append(record)
            continue

        duplicate_count += 1
        if existing["family"] != record["family"]:
            cross_family_conflicts += 1
            cross_family_examples.append(
                {
                    "kept_sequence_id": existing["sequence_id"],
                    "kept_family": existing["family"],
                    "dropped_sequence_id": record["sequence_id"],
                    "dropped_family": record["family"],
                }
            )
            print(
                "[warning] Duplicate sequence across families detected: "
                f"kept {existing['sequence_id']} ({existing['family']}), "
                f"dropped {record['sequence_id']} ({record['family']})."
            )

    dedup_stats = {
        "total_duplicates_removed": duplicate_count,
        "cross_family_conflicts": cross_family_conflicts,
        "cross_family_examples": cross_family_examples[:20],
    }
    return unique_records, dedup_stats


def sample_per_class(
    records: List[dict], max_samples_per_class: int, seed: int
) -> Tuple[List[dict], Dict[str, dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    sampled_records: List[dict] = []
    sampling_stats: Dict[str, dict] = {}
    rng = random.Random(seed)

    for record in records:
        grouped[record["family"]].append(record)

    for family, family_records in grouped.items():
        original_count = len(family_records)
        if max_samples_per_class > 0 and original_count > max_samples_per_class:
            selected_records = rng.sample(family_records, max_samples_per_class)
        else:
            selected_records = list(family_records)

        sampled_records.extend(selected_records)
        sampling_stats[family] = {
            "before_sampling": original_count,
            "after_sampling": len(selected_records),
        }

    return sampled_records, sampling_stats


def _rounded_split_counts(total: int) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0

    train_count = max(1, int(round(total * TRAIN_RATIO)))
    val_count = max(1, int(round(total * VAL_RATIO)))
    test_count = max(1, total - train_count - val_count)

    while train_count + val_count + test_count > total:
        largest = max(
            [("train", train_count), ("val", val_count), ("test", test_count)],
            key=lambda item: item[1],
        )[0]
        if largest == "train" and train_count > 1:
            train_count -= 1
        elif largest == "val" and val_count > 1:
            val_count -= 1
        elif largest == "test" and test_count > 1:
            test_count -= 1
        else:
            break

    while train_count + val_count + test_count < total:
        smallest = min(
            [("train", train_count), ("val", val_count), ("test", test_count)],
            key=lambda item: item[1],
        )[0]
        if smallest == "train":
            train_count += 1
        elif smallest == "val":
            val_count += 1
        else:
            test_count += 1

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        if test_count == 0:
            test_count = 1

    while train_count + val_count + test_count > total:
        if train_count > val_count and train_count > test_count and train_count > 1:
            train_count -= 1
        elif val_count > test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    return train_count, val_count, test_count


def _fallback_split_class_records(
    family_records: List[dict], seed: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    shuffled = list(family_records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count, val_count, test_count = _rounded_split_counts(total)

    train_records = shuffled[:train_count]
    val_records = shuffled[train_count : train_count + val_count]
    test_records = shuffled[train_count + val_count : train_count + val_count + test_count]
    return train_records, val_records, test_records


def split_dataset(
    records: List[dict], seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    if not records:
        raise ValueError("No records available after preprocessing.")

    df = pd.DataFrame(records)
    split_notes: List[str] = []
    label_counts = df["family"].value_counts().to_dict()
    min_class_size = min(label_counts.values())

    try:
        if len(df) >= 3 and min_class_size >= 3:
            train_df, temp_df = train_test_split(
                df,
                test_size=(VAL_RATIO + TEST_RATIO),
                random_state=seed,
                shuffle=True,
                stratify=df["family"],
            )
            temp_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=temp_ratio,
                random_state=seed,
                shuffle=True,
                stratify=temp_df["family"],
            )
            return (
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
                split_notes,
            )
    except ValueError as exc:
        split_notes.append(
            f"Stratified split failed on full dataset, falling back to per-class split: {exc}"
        )

    if min_class_size < 3:
        split_notes.append(
            "At least one class has fewer than 3 samples after preprocessing; "
            "using per-class fallback split to keep the pipeline running."
        )

    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for index, (family, family_df) in enumerate(df.groupby("family", sort=False)):
        family_records = family_df.to_dict("records")
        train_records, val_records, test_records = _fallback_split_class_records(
            family_records, seed + index
        )
        if not val_records or not test_records:
            split_notes.append(
                f"Family {family} has only {len(family_records)} samples; "
                "some splits may be empty for this class."
            )
        train_parts.append(pd.DataFrame(train_records))
        val_parts.append(pd.DataFrame(val_records))
        test_parts.append(pd.DataFrame(test_records))

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pd.DataFrame(columns=df.columns)
    )
    val_df = (
        pd.concat(val_parts, ignore_index=True)
        if val_parts
        else pd.DataFrame(columns=df.columns)
    )
    test_df = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=df.columns)
    )

    return train_df, val_df, test_df, split_notes


def build_label_mapping(families: Iterable[str]) -> Dict[str, int]:
    return {family: idx for idx, family in enumerate(families)}


def attach_labels(dataframe: pd.DataFrame, label_mapping: Dict[str, int]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["label"] = dataframe["family"].map(label_mapping)
    ordered_columns = ["sequence_id", "family", "label", "sequence", "seq_len"]
    return dataframe[ordered_columns].sort_values(
        by=["label", "seq_len", "sequence_id"], ascending=[True, True, True]
    ).reset_index(drop=True)


def family_counts(dataframe: pd.DataFrame, families: Iterable[str]) -> Dict[str, int]:
    counts = dataframe["family"].value_counts().to_dict() if not dataframe.empty else {}
    return {family: int(counts.get(family, 0)) for family in families}


def build_summary(
    families: List[str],
    family_stats: Dict[str, dict],
    dedup_stats: dict,
    sampling_stats: Dict[str, dict],
    final_records: List[dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_notes: List[str],
) -> dict:
    lengths = [record["seq_len"] for record in final_records]

    summary = {
        "families": families,
        "family_stats": {},
        "deduplication": dedup_stats,
        "splits": {
            "train": {
                "total": int(len(train_df)),
                "per_family": family_counts(train_df, families),
            },
            "val": {
                "total": int(len(val_df)),
                "per_family": family_counts(val_df, families),
            },
            "test": {
                "total": int(len(test_df)),
                "per_family": family_counts(test_df, families),
            },
        },
        "totals": {
            "raw_samples": int(sum(family_stats[family]["raw_count"] for family in families)),
            "after_cleaning": int(
                sum(family_stats[family]["cleaned_count"] for family in families)
            ),
            "after_deduplication": int(sum(stat["before_sampling"] for stat in sampling_stats.values())),
            "after_sampling": int(len(final_records)),
            "total_samples": int(len(final_records)),
        },
        "sequence_length": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(round(statistics.mean(lengths), 4)) if lengths else 0.0,
        },
        "notes": split_notes,
    }

    for family in families:
        summary["family_stats"][family] = {
            "file_path": family_stats[family]["file_path"],
            "raw_count": int(family_stats[family]["raw_count"]),
            "cleaned_count": int(family_stats[family]["cleaned_count"]),
            "dropped_empty": int(family_stats[family]["dropped_empty"]),
            "dropped_length": int(family_stats[family]["dropped_length"]),
            "after_deduplication": int(sampling_stats.get(family, {}).get("before_sampling", 0)),
            "after_sampling": int(sampling_stats.get(family, {}).get("after_sampling", 0)),
        }

    return summary


def save_outputs(
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_mapping: Dict[str, int],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(label_mapping, handle, indent=2, ensure_ascii=False)

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def print_summary(summary: dict) -> None:
    print("=== Dataset Build Summary ===")
    for family, stats in summary["family_stats"].items():
        print(
            f"{family}: raw={stats['raw_count']}, "
            f"cleaned={stats['cleaned_count']}, "
            f"dedup={stats['after_deduplication']}, "
            f"sampled={stats['after_sampling']}"
        )
    print(
        "duplicates_removed="
        f"{summary['deduplication']['total_duplicates_removed']}, "
        f"cross_family_conflicts={summary['deduplication']['cross_family_conflicts']}"
    )
    print(
        f"train={summary['splits']['train']['total']}, "
        f"val={summary['splits']['val']['total']}, "
        f"test={summary['splits']['test']['total']}"
    )
    print(
        f"total={summary['totals']['total_samples']}, "
        f"min_len={summary['sequence_length']['min']}, "
        f"max_len={summary['sequence_length']['max']}, "
        f"mean_len={summary['sequence_length']['mean']}"
    )
    for note in summary["notes"]:
        print(f"[note] {note}")


def main() -> None:
    args = parse_args()

    records, family_stats = load_family_sequences(
        input_dir=args.input_dir,
        families=args.families,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    dedup_records, dedup_stats = deduplicate_records(records)
    sampled_records, sampling_stats = sample_per_class(
        dedup_records,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )

    label_mapping = build_label_mapping(args.families)
    train_df, val_df, test_df, split_notes = split_dataset(sampled_records, args.seed)
    train_df = attach_labels(train_df, label_mapping)
    val_df = attach_labels(val_df, label_mapping)
    test_df = attach_labels(test_df, label_mapping)

    summary = build_summary(
        families=args.families,
        family_stats=family_stats,
        dedup_stats=dedup_stats,
        sampling_stats=sampling_stats,
        final_records=sampled_records,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        split_notes=split_notes,
    )
    save_outputs(args.output_dir, train_df, val_df, test_df, label_mapping, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
