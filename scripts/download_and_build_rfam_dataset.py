#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files"
DEFAULT_RAW_DIR = Path("data/rfam_fasta")
DEFAULT_OUTPUT_DIR = Path("processed_rfam_large")
DEFAULT_START_ID = 1
DEFAULT_END_ID = 4360
DEFAULT_MIN_LEN = 10
DEFAULT_MAX_LEN = 1000
DEFAULT_MIN_SAMPLES_PER_FAMILY = 1000
DEFAULT_MAX_FAMILIES = 50
DEFAULT_MAX_SAMPLES_PER_FAMILY = 2000
DEFAULT_SEED = 42
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 2
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
VALID_BASES = {"A", "C", "G", "U", "N"}
CACHE_DIRNAME = ".family_cache"
MAX_CONFLICT_EXAMPLES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Rfam family FASTA files and build a multi-family RNA classification dataset."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to store downloaded Rfam FASTA files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save processed train/val/test CSV files.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL for Rfam FASTA downloads.",
    )
    parser.add_argument("--start_id", type=int, default=DEFAULT_START_ID, help="First RF family numeric ID.")
    parser.add_argument("--end_id", type=int, default=DEFAULT_END_ID, help="Last RF family numeric ID.")
    parser.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN, help="Minimum cleaned sequence length.")
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN, help="Maximum cleaned sequence length.")
    parser.add_argument(
        "--min_samples_per_family",
        type=int,
        default=DEFAULT_MIN_SAMPLES_PER_FAMILY,
        help="Minimum number of sequences required for a family to be eligible.",
    )
    parser.add_argument(
        "--max_families",
        type=int,
        default=DEFAULT_MAX_FAMILIES,
        help="Maximum number of families to include in the final dataset.",
    )
    parser.add_argument(
        "--max_samples_per_family",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_FAMILY,
        help="Maximum number of samples kept per selected family.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Download timeout in seconds.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Download retries per family.")
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download FASTA files and skip dataset construction.",
    )
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Only build from existing local FASTA files and skip downloading.",
    )
    args = parser.parse_args()
    if args.download_only and args.build_only:
        parser.error("--download_only and --build_only cannot both be enabled.")
    return args


def family_id_from_number(number: int) -> str:
    return f"RF{number:05d}"


def family_range(start_id: int, end_id: int) -> List[str]:
    return [family_id_from_number(index) for index in range(start_id, end_id + 1)]


def candidate_file_paths(raw_dir: Path, family_id: str) -> List[Path]:
    return [
        raw_dir / f"{family_id}.fa.gz",
        raw_dir / f"{family_id}.fasta.gz",
        raw_dir / f"{family_id}.fa",
        raw_dir / f"{family_id}.fasta",
    ]


def resolve_local_fasta(raw_dir: Path, family_id: str) -> Path | None:
    for path in candidate_file_paths(raw_dir, family_id):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def open_text_file(file_path: Path):
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")
    return file_path.open("r", encoding="utf-8")


def parse_fasta_records(file_path: Path, family_id: str) -> Iterator[Tuple[str, str]]:
    header: str | None = None
    seq_lines: List[str] = []
    generated_count = 0

    def flush_record(current_header: str | None, current_lines: List[str]) -> Tuple[str, str] | None:
        nonlocal generated_count
        if current_header is None:
            return None
        raw_sequence = "".join(current_lines)
        token = current_header.strip().split()[0] if current_header.strip() else ""
        if token:
            sequence_id = token
        else:
            generated_count += 1
            sequence_id = f"{family_id}_auto_{generated_count:06d}"
        return sequence_id, raw_sequence

    with open_text_file(file_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                record = flush_record(header, seq_lines)
                if record is not None:
                    yield record
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

    final_record = flush_record(header, seq_lines)
    if final_record is not None:
        yield final_record


def clean_sequence(sequence: str) -> str:
    sequence = "".join(sequence.upper().split()).replace("T", "U")
    cleaned = [base for base in sequence if base in VALID_BASES]
    return "".join(cleaned)


def download_family_file(
    family_id: str,
    raw_dir: Path,
    base_url: str,
    timeout: int,
    retries: int,
) -> str:
    existing_path = resolve_local_fasta(raw_dir, family_id)
    if existing_path is not None and existing_path.stat().st_size > 0:
        print(f"[SKIP] Exists {family_id}")
        return "already_exists"

    target_path = raw_dir / f"{family_id}.fa.gz"
    url = f"{base_url.rstrip('/')}/{family_id}.fa.gz"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if getattr(response, "status", 200) == 404:
                    print(f"[MISSING] {family_id} not found")
                    return "missing_or_failed"
                temp_path = target_path.with_suffix(target_path.suffix + ".part")
                with temp_path.open("wb") as handle:
                    handle.write(response.read())
                if temp_path.stat().st_size == 0:
                    temp_path.unlink(missing_ok=True)
                    raise RuntimeError("Downloaded file is empty.")
                temp_path.replace(target_path)
                print(f"[OK] Downloaded {family_id}")
                return "downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"[MISSING] {family_id} not found")
                return "missing_or_failed"
            if attempt == retries:
                print(f"[WARN] Failed {family_id} after retries")
                return "missing_or_failed"
        except Exception:
            if attempt == retries:
                print(f"[WARN] Failed {family_id} after retries")
                return "missing_or_failed"

    print(f"[WARN] Failed {family_id} after retries")
    return "missing_or_failed"


def run_downloads(args: argparse.Namespace, family_ids: Iterable[str]) -> Dict[str, Any]:
    download_stats = {
        "base_url": args.base_url,
        "attempted_downloads": int(len(list(family_ids))) if not isinstance(family_ids, list) else len(family_ids),
        "downloaded": 0,
        "already_exists": 0,
        "missing_or_failed": 0,
    }
    family_ids = list(family_ids)
    download_stats["attempted_downloads"] = len(family_ids)
    for family_id in family_ids:
        status = download_family_file(
            family_id=family_id,
            raw_dir=args.raw_dir,
            base_url=args.base_url,
            timeout=args.timeout,
            retries=args.retries,
        )
        download_stats[status] += 1
    return download_stats


def scan_local_files(raw_dir: Path, family_ids: Iterable[str]) -> List[str]:
    available = []
    for family_id in family_ids:
        if resolve_local_fasta(raw_dir, family_id) is not None:
            available.append(family_id)
    return available


def cache_family_records(cache_dir: Path, family_id: str, records: List[dict]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{family_id}.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return cache_path


def load_cached_family_records(cache_path: Path) -> Iterator[dict]:
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def process_family_file(
    file_path: Path,
    family_id: str,
    min_len: int,
    max_len: int,
) -> Tuple[Dict[str, Any], List[dict]]:
    raw_count = 0
    cleaned_count = 0
    dropped_empty = 0
    dropped_length = 0
    family_duplicate_count = 0
    unique_records: List[dict] = []
    seen_sequences: set[str] = set()

    for sequence_id, raw_sequence in parse_fasta_records(file_path, family_id):
        raw_count += 1
        cleaned_sequence = clean_sequence(raw_sequence)
        if not cleaned_sequence:
            dropped_empty += 1
            continue

        seq_len = len(cleaned_sequence)
        if seq_len < min_len or seq_len > max_len:
            dropped_length += 1
            continue

        cleaned_count += 1
        if cleaned_sequence in seen_sequences:
            family_duplicate_count += 1
            continue

        seen_sequences.add(cleaned_sequence)
        unique_records.append(
            {
                "sequence_id": sequence_id,
                "family": family_id,
                "sequence": cleaned_sequence,
                "seq_len": seq_len,
            }
        )

    family_stats = {
        "file_path": str(file_path),
        "raw_count": raw_count,
        "cleaned_count": cleaned_count,
        "dropped_empty": dropped_empty,
        "dropped_length": dropped_length,
        "after_family_dedup": len(unique_records),
        "after_global_dedup": 0,
        "selected": False,
        "after_sampling": 0,
    }
    return family_stats, unique_records


def rank_families_for_selection(
    family_stats: Dict[str, Dict[str, Any]],
    min_samples_per_family: int,
) -> List[str]:
    eligible = [
        family_id
        for family_id, stats in family_stats.items()
        if stats["after_family_dedup"] >= min_samples_per_family
    ]
    return sorted(
        eligible,
        key=lambda family_id: (-family_stats[family_id]["after_family_dedup"], family_id),
    )


def sample_family_records(records: List[dict], max_samples_per_family: int, seed: int) -> List[dict]:
    if max_samples_per_family <= 0 or len(records) <= max_samples_per_family:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, max_samples_per_family)


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
        if train_count >= val_count and train_count >= test_count and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    while train_count + val_count + test_count < total:
        if train_count <= val_count and train_count <= test_count:
            train_count += 1
        elif val_count <= test_count:
            val_count += 1
        else:
            test_count += 1

    return train_count, val_count, test_count


def fallback_split_family_records(
    family_records: List[dict], seed: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    shuffled = list(family_records)
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, test_count = _rounded_split_counts(len(shuffled))
    train_records = shuffled[:train_count]
    val_records = shuffled[train_count : train_count + val_count]
    test_records = shuffled[train_count + val_count : train_count + val_count + test_count]
    return train_records, val_records, test_records


def split_dataset(records: List[dict], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    if not records:
        raise ValueError("No records available after preprocessing.")

    df = pd.DataFrame(records)
    split_notes: List[str] = []
    family_counts = df["family"].value_counts().to_dict()
    min_family_size = min(family_counts.values())

    try:
        if len(df) >= 3 and min_family_size >= 3:
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
            f"Stratified split failed on full dataset, falling back to per-family split: {exc}"
        )

    if min_family_size < 3:
        split_notes.append(
            "At least one family has fewer than 3 samples after preprocessing; using per-family fallback split."
        )

    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for index, (family_id, family_df) in enumerate(df.groupby("family", sort=False)):
        train_records, val_records, test_records = fallback_split_family_records(
            family_df.to_dict("records"),
            seed + index,
        )
        if not val_records or not test_records:
            split_notes.append(
                f"Family {family_id} has limited samples ({len(family_df)}); some splits may be small."
            )
        train_parts.append(pd.DataFrame(train_records))
        val_parts.append(pd.DataFrame(val_records))
        test_parts.append(pd.DataFrame(test_records))

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, val_df, test_df, split_notes


def build_label_mapping(selected_families: List[str]) -> Dict[str, int]:
    ordered_families = sorted(selected_families)
    return {family_id: label for label, family_id in enumerate(ordered_families)}


def attach_labels(dataframe: pd.DataFrame, label_mapping: Dict[str, int]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["label"] = dataframe["family"].map(label_mapping)
    ordered_columns = ["sequence_id", "family", "label", "sequence", "seq_len"]
    return dataframe[ordered_columns].sort_values(
        by=["label", "seq_len", "sequence_id"], ascending=[True, True, True]
    ).reset_index(drop=True)


def family_counts(dataframe: pd.DataFrame, selected_families: List[str]) -> Dict[str, int]:
    counts = dataframe["family"].value_counts().to_dict() if not dataframe.empty else {}
    return {family_id: int(counts.get(family_id, 0)) for family_id in selected_families}


def build_summary(
    args: argparse.Namespace,
    download_stats: Dict[str, Any],
    selected_families: List[str],
    family_stats: Dict[str, Dict[str, Any]],
    deduplication_stats: Dict[str, Any],
    sampled_records: List[dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_notes: List[str],
) -> Dict[str, Any]:
    lengths = [record["seq_len"] for record in sampled_records]
    summary = {
        "families": selected_families,
        "download": download_stats,
        "selection": {
            "min_samples_per_family": args.min_samples_per_family,
            "max_families": args.max_families,
            "max_samples_per_family": args.max_samples_per_family,
            "selected_families": selected_families,
            "family_selection_order": "after_family_dedup_desc_then_family_id",
            "label_assignment_order": "family_id_ascending",
        },
        "totals": {
            "raw_samples": int(
                sum(stats["raw_count"] for stats in family_stats.values())
            ),
            "after_cleaning": int(
                sum(stats["cleaned_count"] for stats in family_stats.values())
            ),
            "after_family_deduplication": int(
                sum(stats["after_family_dedup"] for stats in family_stats.values())
            ),
            "after_global_deduplication": int(
                sum(family_stats[family_id]["after_global_dedup"] for family_id in selected_families)
            ),
            "after_sampling": int(len(sampled_records)),
            "total_samples": int(len(sampled_records)),
        },
        "family_stats": family_stats,
        "splits": {
            "train": {
                "total": int(len(train_df)),
                "per_family": family_counts(train_df, selected_families),
            },
            "val": {
                "total": int(len(val_df)),
                "per_family": family_counts(val_df, selected_families),
            },
            "test": {
                "total": int(len(test_df)),
                "per_family": family_counts(test_df, selected_families),
            },
        },
        "sequence_length": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(statistics.mean(lengths)) if lengths else 0.0,
            "median": float(statistics.median(lengths)) if lengths else 0.0,
        },
        "deduplication": deduplication_stats,
        "notes": split_notes,
    }
    return summary


def save_outputs(
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_mapping: Dict[str, int],
    summary: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(label_mapping, handle, indent=2, ensure_ascii=False)
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def print_build_summary(
    selected_families: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    print(f"[OK] Selected {len(selected_families)} families")
    print(
        f"[INFO] Dedup: family_removed={summary['deduplication']['family_duplicates_removed']}, "
        f"global_removed={summary['deduplication']['global_duplicates_removed']}, "
        f"cross_family_conflicts={summary['deduplication']['cross_family_conflicts']}"
    )
    print(
        f"[INFO] Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(f"[RESULT] Output saved to {summary['selection']['selected_families'] and summary['selection'] and ''}{''}")


def build_dataset(args: argparse.Namespace, download_stats: Dict[str, Any]) -> None:
    family_ids = family_range(args.start_id, args.end_id)
    available_families = scan_local_files(args.raw_dir, family_ids)
    if not available_families:
        raise FileNotFoundError(f"No FASTA files were found in {args.raw_dir}.")

    cache_dir = args.output_dir / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    family_stats: Dict[str, Dict[str, Any]] = {}
    family_duplicates_removed = 0

    print("[INFO] Building dataset from local FASTA files...")
    for family_id in available_families:
        file_path = resolve_local_fasta(args.raw_dir, family_id)
        if file_path is None:
            continue
        stats, unique_records = process_family_file(
            file_path=file_path,
            family_id=family_id,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        family_duplicates_removed += stats["cleaned_count"] - stats["after_family_dedup"]
        family_stats[family_id] = stats
        if unique_records:
            cache_family_records(cache_dir, family_id, unique_records)

    ranked_families = rank_families_for_selection(
        family_stats=family_stats,
        min_samples_per_family=args.min_samples_per_family,
    )

    seen_sequences: Dict[str, Dict[str, Any]] = {}
    selected_families: List[str] = []
    sampled_records: List[dict] = []
    global_duplicates_removed = 0
    cross_family_conflicts = 0
    cross_family_examples: List[Dict[str, Any]] = []

    for selection_index, family_id in enumerate(ranked_families):
        if len(selected_families) >= args.max_families:
            break

        cache_path = cache_dir / f"{family_id}.jsonl"
        if not cache_path.exists():
            continue

        family_global_unique: List[dict] = []
        for record in load_cached_family_records(cache_path):
            existing = seen_sequences.get(record["sequence"])
            if existing is not None:
                global_duplicates_removed += 1
                if existing["family"] != record["family"]:
                    cross_family_conflicts += 1
                    if len(cross_family_examples) < MAX_CONFLICT_EXAMPLES:
                        cross_family_examples.append(
                            {
                                "kept_family": existing["family"],
                                "kept_sequence_id": existing["sequence_id"],
                                "dropped_family": record["family"],
                                "dropped_sequence_id": record["sequence_id"],
                            }
                        )
                continue

            seen_sequences[record["sequence"]] = record
            family_global_unique.append(record)

        family_stats[family_id]["after_global_dedup"] = len(family_global_unique)
        if len(family_global_unique) < 3:
            continue

        selected_families.append(family_id)
        family_stats[family_id]["selected"] = True
        sampled_family_records = sample_family_records(
            records=family_global_unique,
            max_samples_per_family=args.max_samples_per_family,
            seed=args.seed + selection_index,
        )
        family_stats[family_id]["after_sampling"] = len(sampled_family_records)
        sampled_records.extend(sampled_family_records)

    if not selected_families:
        raise ValueError("No families satisfied the selection criteria.")

    label_mapping = build_label_mapping(selected_families)
    train_df, val_df, test_df, split_notes = split_dataset(sampled_records, args.seed)
    train_df = attach_labels(train_df, label_mapping)
    val_df = attach_labels(val_df, label_mapping)
    test_df = attach_labels(test_df, label_mapping)

    summary = build_summary(
        args=args,
        download_stats=download_stats,
        selected_families=selected_families,
        family_stats=family_stats,
        deduplication_stats={
            "family_duplicates_removed": int(family_duplicates_removed),
            "global_duplicates_removed": int(global_duplicates_removed),
            "cross_family_conflicts": int(cross_family_conflicts),
            "cross_family_examples": cross_family_examples,
        },
        sampled_records=sampled_records,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        split_notes=split_notes,
    )
    save_outputs(args.output_dir, train_df, val_df, test_df, label_mapping, summary)

    print(f"[OK] Selected {len(selected_families)} families")
    print(
        f"[OK] Wrote train.csv / val.csv / test.csv"
    )
    print(f"[RESULT] Output saved to {args.output_dir}")


def main() -> None:
    args = parse_args()
    family_ids = family_range(args.start_id, args.end_id)
    print(
        f"[INFO] Download range: {family_ids[0]}-{family_ids[-1]}"
    )

    download_stats = {
        "base_url": args.base_url,
        "attempted_downloads": len(family_ids),
        "downloaded": 0,
        "already_exists": 0,
        "missing_or_failed": 0,
    }

    if not args.build_only:
        download_stats = run_downloads(args, family_ids)

    if args.download_only:
        print("[RESULT] Download completed")
        return

    build_dataset(args, download_stats)


if __name__ == "__main__":
    main()
