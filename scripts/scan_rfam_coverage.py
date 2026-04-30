#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_RAW_DIR = Path("data/rfam_fasta")
DEFAULT_MAX_LEN = 512
DEFAULT_THRESHOLDS = "300,200,100,50,20,10,1"
DEFAULT_OUTPUT_JSON = Path("reports/rfam_coverage_scan.json")
DEFAULT_OUTPUT_CSV = Path("reports/rfam_coverage_by_family.csv")
VALID_BASES = {"A", "C", "G", "U", "N"}
TOP_K = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan local Rfam FASTA files and summarize family coverage under different sample thresholds."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing local Rfam FASTA files.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Maximum cleaned sequence length to keep.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated min_samples_per_family thresholds.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to save aggregated coverage JSON.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save per-family coverage CSV.",
    )
    return parser.parse_args()


def open_text_file(file_path: Path):
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")
    return file_path.open("r", encoding="utf-8")


def parse_fasta_records(file_path: Path, family_id: str) -> Iterator[tuple[str, str]]:
    header: str | None = None
    seq_lines: list[str] = []
    generated_count = 0

    def flush_record(current_header: str | None, current_lines: list[str]) -> tuple[str, str] | None:
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


def family_id_from_path(file_path: Path) -> str:
    name = file_path.name
    for suffix in (".fa.gz", ".fasta.gz", ".fa", ".fasta"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return file_path.stem


def find_fasta_files(raw_dir: Path) -> list[Path]:
    patterns = ("*.fa", "*.fasta", "*.fa.gz", "*.fasta.gz")
    seen: set[Path] = set()
    matched: list[Path] = []
    for pattern in patterns:
        for path in raw_dir.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                matched.append(path)
    return sorted(matched, key=lambda path: path.name)


def parse_thresholds(value: str) -> list[int]:
    thresholds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(int(item))
    return sorted(set(thresholds), reverse=True)


def scan_family(file_path: Path, max_len: int) -> dict:
    family_id = family_id_from_path(file_path)
    unique_sequences: set[str] = set()
    warnings: list[str] = []

    try:
        for _, raw_sequence in parse_fasta_records(file_path, family_id):
            cleaned = clean_sequence(raw_sequence)
            if not cleaned:
                continue
            if len(cleaned) > max_len:
                continue
            unique_sequences.add(cleaned)
    except Exception as exc:
        warnings.append(f"[WARN] Failed to parse {file_path.name}: {exc}")

    return {
        "family_id": family_id,
        "valid_sequence_count": len(unique_sequences),
        "source_file": str(file_path),
        "warnings": warnings,
    }


def estimate_capped_sequences(counts: Iterable[int], threshold: int, cap: int) -> int:
    total = 0
    for count in counts:
        if count >= threshold:
            total += min(count, cap)
    return total


def write_family_csv(output_csv: Path, family_rows: list[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family_id", "valid_sequence_count", "source_file"],
        )
        writer.writeheader()
        for row in family_rows:
            writer.writerow(
                {
                    "family_id": row["family_id"],
                    "valid_sequence_count": row["valid_sequence_count"],
                    "source_file": row["source_file"],
                }
            )


def build_report(
    raw_dir: Path,
    max_len: int,
    thresholds: list[int],
    family_rows: list[dict],
) -> dict:
    counts = [row["valid_sequence_count"] for row in family_rows]
    valid_rows = [row for row in family_rows if row["valid_sequence_count"] > 0]
    threshold_summary = []

    for threshold in thresholds:
        covered_rows = [row for row in valid_rows if row["valid_sequence_count"] >= threshold]
        covered_counts = [row["valid_sequence_count"] for row in covered_rows]
        threshold_summary.append(
            {
                "min_samples_per_family": threshold,
                "covered_families": len(covered_rows),
                "estimated_sequences_if_capped_500": estimate_capped_sequences(covered_counts, threshold, 500),
                "estimated_sequences_if_capped_800": estimate_capped_sequences(covered_counts, threshold, 800),
                "estimated_sequences_if_capped_1000": estimate_capped_sequences(covered_counts, threshold, 1000),
            }
        )

    sorted_rows = sorted(family_rows, key=lambda row: row["valid_sequence_count"], reverse=True)
    positive_rows = [row for row in sorted_rows if row["valid_sequence_count"] > 0]

    return {
        "raw_dir": str(raw_dir),
        "max_len": max_len,
        "total_files_found": len(family_rows),
        "total_families_with_valid_sequences": len(valid_rows),
        "total_valid_sequences": sum(counts),
        "threshold_summary": threshold_summary,
        "top_families": [
            {
                "family_id": row["family_id"],
                "valid_sequence_count": row["valid_sequence_count"],
                "source_file": row["source_file"],
            }
            for row in sorted_rows[:TOP_K]
        ],
        "bottom_families": [
            {
                "family_id": row["family_id"],
                "valid_sequence_count": row["valid_sequence_count"],
                "source_file": row["source_file"],
            }
            for row in sorted(positive_rows, key=lambda row: row["valid_sequence_count"])[:TOP_K]
        ],
    }


def write_json(output_json: Path, payload: dict) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    fasta_files = find_fasta_files(args.raw_dir)
    family_rows: list[dict] = []
    warning_messages: list[str] = []

    for file_path in fasta_files:
        row = scan_family(file_path, args.max_len)
        family_rows.append(
            {
                "family_id": row["family_id"],
                "valid_sequence_count": row["valid_sequence_count"],
                "source_file": row["source_file"],
            }
        )
        warning_messages.extend(row["warnings"])

    family_rows.sort(key=lambda row: row["valid_sequence_count"], reverse=True)
    report = build_report(args.raw_dir, args.max_len, thresholds, family_rows)

    write_json(args.output_json, report)
    write_family_csv(args.output_csv, family_rows)

    for warning in warning_messages:
        print(warning)

    print("[OK] Scanned Rfam FASTA directory")
    print(f"raw_dir = {args.raw_dir}")
    print(f"max_len = {args.max_len}")
    print(f"files_found = {report['total_files_found']}")
    print(f"families_with_valid_sequences = {report['total_families_with_valid_sequences']}")
    print(f"total_valid_sequences = {report['total_valid_sequences']}")
    print("")
    print("Threshold summary:")
    for item in report["threshold_summary"]:
        print(
            f"min_samples >= {item['min_samples_per_family']}: "
            f"covered_families = {item['covered_families']}, "
            f"capped500 = {item['estimated_sequences_if_capped_500']}, "
            f"capped800 = {item['estimated_sequences_if_capped_800']}, "
            f"capped1000 = {item['estimated_sequences_if_capped_1000']}"
        )
    print("")
    print(f"[OK] Saved JSON to {args.output_json}")
    print(f"[OK] Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
