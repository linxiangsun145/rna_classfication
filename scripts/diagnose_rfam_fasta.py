#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


DEFAULT_RAW_DIR = Path("data/rfam_fasta")
DEFAULT_START_ID = 1
DEFAULT_END_ID = 4360
DEFAULT_MIN_LEN = 1
DEFAULT_MAX_LEN = 512
DEFAULT_ALLOWED_CHARS = "ACGUTN"
DEFAULT_OUTPUT_CSV = Path("reports/rfam_fasta_diagnosis.csv")
DEFAULT_OUTPUT_JSON = Path("reports/rfam_fasta_diagnosis_summary.json")
SUPPORTED_SUFFIXES = (".fa", ".fasta", ".fa.gz", ".fasta.gz")
MAX_INVALID_CHAR_EXAMPLES = 5
TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why local Rfam family FASTA files do or do not leave valid sequences."
    )
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing local Rfam FASTA files.")
    parser.add_argument("--start_id", type=int, default=DEFAULT_START_ID, help="First RF family numeric ID.")
    parser.add_argument("--end_id", type=int, default=DEFAULT_END_ID, help="Last RF family numeric ID.")
    parser.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN, help="Minimum cleaned sequence length.")
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN, help="Maximum cleaned sequence length.")
    parser.add_argument("--allowed_chars", type=str, default=DEFAULT_ALLOWED_CHARS, help="Allowed RNA characters before filtering.")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Detailed per-family diagnosis CSV.")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Aggregated diagnosis JSON.")
    parser.add_argument("--check_global_dedup", action="store_true", help="Run cross-family global deduplication diagnosis.")
    return parser.parse_args()


def family_id_from_number(number: int) -> str:
    return f"RF{number:05d}"


def family_range(start_id: int, end_id: int) -> list[str]:
    return [family_id_from_number(index) for index in range(start_id, end_id + 1)]


def candidate_file_paths(raw_dir: Path, family_id: str) -> list[Path]:
    return [raw_dir / f"{family_id}{suffix}" for suffix in SUPPORTED_SUFFIXES]


def resolve_local_fasta(raw_dir: Path, family_id: str) -> Path | None:
    for path in candidate_file_paths(raw_dir, family_id):
        if path.exists():
            return path
    return None


def open_text_file(file_path: Path):
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8", errors="strict")
    return file_path.open("r", encoding="utf-8", errors="strict")


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.upper().split())


def parse_fasta_records(file_path: Path, family_id: str) -> Iterator[tuple[str, str]]:
    header: str | None = None
    seq_lines: list[str] = []
    generated_count = 0
    saw_any_line = False
    saw_header = False

    def flush_record(current_header: str | None, current_lines: list[str]) -> tuple[str, str] | None:
        nonlocal generated_count
        if current_header is None:
            return None
        token = current_header.strip().split()[0] if current_header.strip() else ""
        if token:
            sequence_id = token
        else:
            generated_count += 1
            sequence_id = f"{family_id}_auto_{generated_count:06d}"
        return sequence_id, "".join(current_lines)

    with open_text_file(file_path) as handle:
        for raw_line in handle:
            saw_any_line = True
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                saw_header = True
                record = flush_record(header, seq_lines)
                if record is not None:
                    yield record
                header = line[1:]
                seq_lines = []
            else:
                if header is None:
                    raise ValueError("Sequence content encountered before FASTA header.")
                seq_lines.append(line)

    if not saw_any_line:
        return
    if not saw_header:
        raise ValueError("No FASTA header lines found.")

    final_record = flush_record(header, seq_lines)
    if final_record is not None:
        yield final_record


def clean_and_measure(
    raw_sequence: str,
    allowed_chars: set[str],
) -> tuple[str, set[str]]:
    normalized = normalize_sequence(raw_sequence)
    invalid_chars = {char for char in normalized if char not in allowed_chars}
    cleaned = "".join(char for char in normalized if char in allowed_chars)
    return cleaned, invalid_chars


def base_row(family_id: str, file_path: Path | None) -> dict[str, object]:
    return {
        "family": family_id,
        "file_path": str(file_path) if file_path is not None else "",
        "file_exists": file_path is not None,
        "primary_status": "",
        "secondary_flags": "",
        "total_records": 0,
        "valid_before_dedup": 0,
        "valid_after_dedup": 0,
        "invalid_char_count": 0,
        "too_short_count": 0,
        "too_long_count": 0,
        "duplicate_within_family_count": 0,
        "dedup_removed_count": 0,
        "example_invalid_chars": "",
        "example_sequence_id": "",
        "example_length": "",
        "_candidate_sequences": [],
        "_status_flags": set(),
    }


def set_example(row: dict[str, object], sequence_id: str, seq_len: int | str) -> None:
    if not row["example_sequence_id"]:
        row["example_sequence_id"] = sequence_id
        row["example_length"] = seq_len


def diagnose_family(
    family_id: str,
    raw_dir: Path,
    min_len: int,
    max_len: int,
    allowed_chars: set[str],
) -> dict[str, object]:
    file_path = resolve_local_fasta(raw_dir, family_id)
    row = base_row(family_id, file_path)

    if file_path is None:
        row["primary_status"] = "file_missing"
        return row

    try:
        if file_path.stat().st_size == 0:
            row["primary_status"] = "empty_file"
            return row
    except OSError:
        row["primary_status"] = "parse_error"
        return row

    try:
        records = list(parse_fasta_records(file_path, family_id))
    except gzip.BadGzipFile:
        row["primary_status"] = "gzip_error"
        return row
    except EOFError:
        row["primary_status"] = "gzip_error"
        return row
    except OSError as exc:
        if file_path.suffix == ".gz":
            row["primary_status"] = "gzip_error"
        else:
            row["primary_status"] = "parse_error"
        row["secondary_flags"] = str(exc)
        return row
    except Exception as exc:
        row["primary_status"] = "parse_error"
        row["secondary_flags"] = str(exc)
        return row

    if not records:
        try:
            content = open_text_file(file_path).read()
        except Exception:
            content = None
        row["primary_status"] = "empty_file" if content is not None and content.strip() == "" else "no_records"
        return row

    row["total_records"] = len(records)
    unique_sequences: set[str] = set()
    candidate_sequences: list[str] = []
    invalid_examples: set[str] = set()

    for sequence_id, raw_sequence in records:
        cleaned, invalid_chars = clean_and_measure(raw_sequence, allowed_chars)
        if invalid_chars and len(invalid_examples) < MAX_INVALID_CHAR_EXAMPLES:
            invalid_examples.update(sorted(invalid_chars))

        if not cleaned:
            row["invalid_char_count"] += 1
            if invalid_chars:
                set_example(row, sequence_id, 0)
            continue

        seq_len = len(cleaned)
        if seq_len < min_len:
            row["too_short_count"] += 1
            set_example(row, sequence_id, seq_len)
            continue
        if seq_len > max_len:
            row["too_long_count"] += 1
            set_example(row, sequence_id, seq_len)
            continue

        if cleaned in unique_sequences:
            row["duplicate_within_family_count"] += 1
            set_example(row, sequence_id, seq_len)
            continue

        unique_sequences.add(cleaned)
        candidate_sequences.append(cleaned)
        set_example(row, sequence_id, seq_len)

    row["valid_before_dedup"] = len(candidate_sequences)
    row["valid_after_dedup"] = len(candidate_sequences)
    row["_candidate_sequences"] = candidate_sequences
    row["example_invalid_chars"] = ";".join(sorted(invalid_examples)[:MAX_INVALID_CHAR_EXAMPLES])

    has_filtered = any(
        int(row[key]) > 0 for key in ("invalid_char_count", "too_short_count", "too_long_count")
    )
    if has_filtered and row["valid_before_dedup"] > 0:
        row["_status_flags"].add("mixed_filtered")
    if int(row["too_short_count"]) > 0:
        row["_status_flags"].add("too_short")
    if int(row["too_long_count"]) > 0:
        row["_status_flags"].add("too_long")
    if int(row["invalid_char_count"]) > 0:
        row["_status_flags"].add("invalid_char")

    if row["valid_before_dedup"] > 0:
        row["primary_status"] = "valid_after_filter"
    elif int(row["invalid_char_count"]) == int(row["total_records"]) and int(row["total_records"]) > 0:
        row["primary_status"] = "invalid_char_only"
    elif int(row["too_short_count"]) == int(row["total_records"]) and int(row["total_records"]) > 0:
        row["primary_status"] = "too_short_only"
    elif int(row["too_long_count"]) == int(row["total_records"]) and int(row["total_records"]) > 0:
        row["primary_status"] = "too_long_only"
    else:
        row["primary_status"] = "no_records"

    return row


def apply_global_dedup(rows: list[dict[str, object]]) -> None:
    first_owner: dict[str, str] = {}
    for row in rows:
        family_id = str(row["family"])
        candidate_sequences = list(row["_candidate_sequences"])
        kept_count = 0
        dedup_removed = 0
        for sequence in candidate_sequences:
            owner = first_owner.get(sequence)
            if owner is None:
                first_owner[sequence] = family_id
                kept_count += 1
            else:
                dedup_removed += 1
        row["valid_after_dedup"] = kept_count
        row["dedup_removed_count"] = dedup_removed
        if int(row["valid_before_dedup"]) > 0 and kept_count == 0:
            row["primary_status"] = "removed_by_dedup"
        elif dedup_removed > 0:
            row["_status_flags"].add("global_dedup")


def finalize_secondary_flags(rows: list[dict[str, object]]) -> None:
    for row in rows:
        flags = set(row["_status_flags"])
        primary = str(row["primary_status"])
        primary_related = {
            "invalid_char_only": "invalid_char",
            "too_short_only": "too_short",
            "too_long_only": "too_long",
            "removed_by_dedup": "global_dedup",
        }
        flags.discard(primary_related.get(primary, ""))
        row["secondary_flags"] = ";".join(sorted(flags))
        row.pop("_candidate_sequences", None)
        row.pop("_status_flags", None)


def build_summary(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    status_counts = Counter(str(row["primary_status"]) for row in rows)
    valid_rows = [row for row in rows if str(row["primary_status"]) == "valid_after_filter"]
    no_valid_rows = [row for row in rows if str(row["primary_status"]) != "valid_after_filter"]

    def top_rows_by(key: str) -> list[dict[str, object]]:
        ranked = sorted(rows, key=lambda row: int(row[key]), reverse=True)
        return [
            {
                "family": row["family"],
                key: int(row[key]),
                "primary_status": row["primary_status"],
            }
            for row in ranked[:TOP_K]
            if int(row[key]) > 0
        ]

    return {
        "total_families_scanned": len(rows),
        "status_counts": dict(status_counts),
        "families_with_valid_sequences": len(valid_rows),
        "families_without_valid_sequences": len(no_valid_rows),
        "global_dedup_checked": bool(args.check_global_dedup),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "allowed_chars": args.allowed_chars,
        "top_too_long_families": top_rows_by("too_long_count"),
        "top_invalid_char_families": top_rows_by("invalid_char_count"),
        "top_dedup_removed_families": top_rows_by("dedup_removed_count"),
    }


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "file_path",
        "file_exists",
        "primary_status",
        "secondary_flags",
        "total_records",
        "valid_before_dedup",
        "valid_after_dedup",
        "invalid_char_count",
        "too_short_count",
        "too_long_count",
        "duplicate_within_family_count",
        "dedup_removed_count",
        "example_invalid_chars",
        "example_sequence_id",
        "example_length",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise SystemExit(f"[ERROR] raw_dir does not exist: {args.raw_dir}")

    allowed_chars = set(args.allowed_chars.upper())
    family_ids = family_range(args.start_id, args.end_id)
    rows: list[dict[str, object]] = []

    for family_id in family_ids:
        row = diagnose_family(
            family_id=family_id,
            raw_dir=args.raw_dir,
            min_len=args.min_len,
            max_len=args.max_len,
            allowed_chars=allowed_chars,
        )
        rows.append(row)

    if args.check_global_dedup:
        apply_global_dedup(rows)

    finalize_secondary_flags(rows)
    summary = build_summary(rows, args)

    write_csv(args.output_csv, rows)
    write_json(args.output_json, summary)

    print("[OK] Diagnosis complete")
    print(f"[SUMMARY] total_families_scanned = {summary['total_families_scanned']}")
    for status, count in sorted(summary["status_counts"].items()):
        print(f"[SUMMARY] {status} = {count}")
    print(f"[SUMMARY] families_with_valid_sequences = {summary['families_with_valid_sequences']}")
    print(f"[SUMMARY] families_without_valid_sequences = {summary['families_without_valid_sequences']}")
    print(f"[OUTPUT] CSV  = {args.output_csv}")
    print(f"[OUTPUT] JSON = {args.output_json}")


if __name__ == "__main__":
    main()
