#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator


DEFAULT_RAW_DIR = Path("data/rfam_fasta")
DEFAULT_DIAGNOSIS_CSV = Path("reports/rfam_fasta_diagnosis.csv")
DEFAULT_START_ID = 1
DEFAULT_END_ID = 4360
DEFAULT_MAX_LENS = "512,1024,2048,4096"
DEFAULT_CAPS = "100,300,500"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_ALLOWED_CHARS = "ACGUTN"
DEFAULT_MIN_LEN = 1
SUPPORTED_SUFFIXES = (".fa", ".fasta", ".fa.gz", ".fasta.gz")
COUNT_THRESHOLDS = (1, 5, 10, 20, 50, 100, 300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate repair strategies for near-full Rfam family coverage without training a model."
    )
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing local Rfam FASTA files.")
    parser.add_argument("--diagnosis_csv", type=Path, default=DEFAULT_DIAGNOSIS_CSV, help="Diagnosis CSV from diagnose_rfam_fasta.py.")
    parser.add_argument("--start_id", type=int, default=DEFAULT_START_ID, help="First RF family numeric ID.")
    parser.add_argument("--end_id", type=int, default=DEFAULT_END_ID, help="Last RF family numeric ID.")
    parser.add_argument("--max_lens", type=str, default=DEFAULT_MAX_LENS, help="Comma-separated max_len values to evaluate.")
    parser.add_argument("--caps", type=str, default=DEFAULT_CAPS, help="Comma-separated per-family caps to estimate.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for repair evaluation reports.")
    parser.add_argument("--allowed_chars", type=str, default=DEFAULT_ALLOWED_CHARS, help="Allowed RNA characters before filtering.")
    parser.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN, help="Minimum cleaned sequence length.")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    sequence = "".join(sequence.upper().split()).replace("T", "U")
    return sequence


def parse_fasta_records(file_path: Path, family_id: str) -> Iterator[tuple[str, str]]:
    header: str | None = None
    seq_lines: list[str] = []
    generated_count = 0
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

    if not saw_header:
        return
    final_record = flush_record(header, seq_lines)
    if final_record is not None:
        yield final_record


def clean_sequence(sequence: str, allowed_chars: set[str]) -> str:
    normalized = normalize_sequence(sequence)
    cleaned = [char for char in normalized if char in allowed_chars]
    return "".join(cleaned)


def sequence_hash(sequence: str) -> str:
    return hashlib.blake2b(sequence.encode("utf-8"), digest_size=16).hexdigest()


def load_diagnosis_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_bucket_counts(counts: dict[str, int]) -> dict[str, int]:
    result = {}
    for threshold in COUNT_THRESHOLDS:
        result[f"families_with_at_least_{threshold}"] = sum(1 for count in counts.values() if count >= threshold)
    return result


def compute_capped_totals(counts: dict[str, int], caps: list[int]) -> dict[str, int]:
    totals = {}
    for cap in caps:
        totals[f"capped_{cap}_total"] = sum(min(count, cap) for count in counts.values())
    return totals


def create_sqlite_index(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode=OFF")
    cursor.execute("PRAGMA synchronous=OFF")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute(
        "CREATE TABLE seen (seq_hash TEXT PRIMARY KEY, first_family TEXT NOT NULL, seq_len INTEGER NOT NULL)"
    )
    connection.commit()
    return connection


def scan_max_len(
    family_ids: list[str],
    raw_dir: Path,
    allowed_chars: set[str],
    min_len: int,
    max_len: int,
    caps: list[int],
    scratch_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    preserving_counts: dict[str, int] = {family_id: 0 for family_id in family_ids}
    strict_counts: dict[str, int] = {family_id: 0 for family_id in family_ids}
    dedup_removed_counts: dict[str, int] = defaultdict(int)
    conflict_groups: dict[str, dict[str, object]] = {}
    warnings: list[str] = []

    scratch_dir.mkdir(parents=True, exist_ok=True)
    db_path = scratch_dir / f"rfam_dedup_{max_len}.sqlite"
    if db_path.exists():
        db_path.unlink()
    connection = create_sqlite_index(db_path)
    cursor = connection.cursor()

    try:
        for family_id in family_ids:
            file_path = resolve_local_fasta(raw_dir, family_id)
            if file_path is None:
                continue

            seen_in_family: set[str] = set()
            try:
                for _, raw_sequence in parse_fasta_records(file_path, family_id):
                    cleaned = clean_sequence(raw_sequence, allowed_chars)
                    if not cleaned:
                        continue
                    seq_len = len(cleaned)
                    if seq_len < min_len or seq_len > max_len:
                        continue

                    seq_hash = sequence_hash(cleaned)
                    if seq_hash in seen_in_family:
                        continue
                    seen_in_family.add(seq_hash)
                    preserving_counts[family_id] += 1

                    cursor.execute(
                        "INSERT OR IGNORE INTO seen (seq_hash, first_family, seq_len) VALUES (?, ?, ?)",
                        (seq_hash, family_id, seq_len),
                    )
                    if cursor.rowcount == 1:
                        strict_counts[family_id] += 1
                        continue

                    owner_row = cursor.execute(
                        "SELECT first_family, seq_len FROM seen WHERE seq_hash = ?",
                        (seq_hash,),
                    ).fetchone()
                    if owner_row is None:
                        continue
                    owner_family, owner_len = owner_row
                    if owner_family == family_id:
                        continue

                    dedup_removed_counts[family_id] += 1
                    conflict = conflict_groups.setdefault(
                        seq_hash,
                        {
                            "max_len": max_len,
                            "sequence_hash": seq_hash,
                            "sequence_length": int(owner_len),
                            "first_family": owner_family,
                            "conflict_families": set(),
                        },
                    )
                    conflict["conflict_families"].add(family_id)
            except gzip.BadGzipFile as exc:
                warnings.append(f"[WARN] gzip_error at {family_id} ({file_path.name}): {exc}")
            except EOFError as exc:
                warnings.append(f"[WARN] gzip_error at {family_id} ({file_path.name}): {exc}")
            except Exception as exc:
                warnings.append(f"[WARN] parse_error at {family_id} ({file_path.name}): {exc}")

        connection.commit()
    finally:
        connection.close()
        if db_path.exists():
            db_path.unlink()

    strict_covered = sum(1 for count in strict_counts.values() if count > 0)
    preserving_covered = sum(1 for count in preserving_counts.values() if count > 0)
    lost_by_dedup = sum(
        1
        for family_id in family_ids
        if preserving_counts[family_id] > 0 and strict_counts[family_id] == 0
    )

    per_strategy = {
        "strict_global_dedup": {
            "covered_families": strict_covered,
            "total_sequences": sum(strict_counts.values()),
            "families_lost_by_dedup": lost_by_dedup,
            "cross_family_duplicate_groups": len(conflict_groups),
            "counts": strict_counts,
            **compute_bucket_counts(strict_counts),
            **compute_capped_totals(strict_counts, caps),
        },
        "family_preserving_dedup": {
            "covered_families": preserving_covered,
            "total_sequences": sum(preserving_counts.values()),
            "families_lost_by_dedup": 0,
            "cross_family_duplicate_groups": len(conflict_groups),
            "counts": preserving_counts,
            **compute_bucket_counts(preserving_counts),
            **compute_capped_totals(preserving_counts, caps),
        },
        "warnings": warnings,
        "dedup_removed_counts": dict(dedup_removed_counts),
    }

    conflict_rows: list[dict[str, object]] = []
    for conflict in conflict_groups.values():
        families = sorted(conflict["conflict_families"])
        conflict_rows.append(
            {
                "max_len": max_len,
                "sequence_hash": conflict["sequence_hash"],
                "sequence_length": conflict["sequence_length"],
                "first_family": conflict["first_family"],
                "conflict_families": ";".join(families),
                "conflict_count": len(families),
            }
        )

    return per_strategy, conflict_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise SystemExit(f"[ERROR] raw_dir does not exist: {args.raw_dir}")
    if not args.diagnosis_csv.exists():
        raise SystemExit(f"[ERROR] diagnosis_csv does not exist: {args.diagnosis_csv}")

    max_lens = parse_int_list(args.max_lens)
    caps = parse_int_list(args.caps)
    allowed_chars = set(args.allowed_chars.upper().replace("T", "U"))
    family_ids = family_range(args.start_id, args.end_id)
    scratch_dir = args.output_dir / ".repair_eval_tmp"
    diagnosis_rows = load_diagnosis_rows(args.diagnosis_csv)

    diagnosis_by_family = {row["family"]: row for row in diagnosis_rows}
    too_long_families = [row["family"] for row in diagnosis_rows if row["primary_status"] == "too_long_only"]
    removed_by_dedup_families = [row["family"] for row in diagnosis_rows if row["primary_status"] == "removed_by_dedup"]
    missing_rows = [row for row in diagnosis_rows if row["primary_status"] == "file_missing"]

    maxlen_rows: list[dict[str, object]] = []
    conflict_rows_all: list[dict[str, object]] = []
    per_maxlen_results: dict[int, dict[str, object]] = {}
    warnings_all: list[str] = []

    for max_len in max_lens:
        result, conflict_rows = scan_max_len(
            family_ids=family_ids,
            raw_dir=args.raw_dir,
            allowed_chars=allowed_chars,
            min_len=args.min_len,
            max_len=max_len,
            caps=caps,
            scratch_dir=scratch_dir,
        )
        per_maxlen_results[max_len] = result
        conflict_rows_all.extend(conflict_rows)
        warnings_all.extend(result["warnings"])

    baseline_recovered: dict[str, set[str]] = {}
    for strategy in ("strict_global_dedup", "family_preserving_dedup"):
        baseline_counts = per_maxlen_results[max_lens[0]][strategy]["counts"]
        baseline_recovered[strategy] = {family for family, count in baseline_counts.items() if count > 0}

    for max_len in max_lens:
        for strategy in ("strict_global_dedup", "family_preserving_dedup"):
            strategy_result = per_maxlen_results[max_len][strategy]
            current_families = {family for family, count in strategy_result["counts"].items() if count > 0}
            recovered_vs_512 = len(current_families - baseline_recovered[strategy])
            row = {
                "max_len": max_len,
                "dedup_strategy": strategy,
                "covered_families": strategy_result["covered_families"],
                "total_sequences": strategy_result["total_sequences"],
                "families_recovered_vs_512": recovered_vs_512,
            }
            for threshold in COUNT_THRESHOLDS:
                row[f"families_with_at_least_{threshold}"] = strategy_result[f"families_with_at_least_{threshold}"]
            for cap in caps:
                row[f"capped_{cap}_total"] = strategy_result[f"capped_{cap}_total"]
            maxlen_rows.append(row)

    too_long_rows: list[dict[str, object]] = []
    recovery_counts = {max_len: 0 for max_len in max_lens[1:]}
    preserving_counts_by_maxlen = {
        max_len: per_maxlen_results[max_len]["family_preserving_dedup"]["counts"] for max_len in max_lens
    }
    for family_id in too_long_families:
        values = {max_len: preserving_counts_by_maxlen[max_len].get(family_id, 0) for max_len in max_lens}
        min_required = next((max_len for max_len in max_lens if values[max_len] > 0), "")
        for max_len in max_lens[1:]:
            if values[512] == 0 and values[max_len] > 0:
                recovery_counts[max_len] += 1
        too_long_rows.append(
            {
                "family": family_id,
                "valid_at_512": values.get(512, 0),
                "valid_at_1024": values.get(1024, 0),
                "valid_at_2048": values.get(2048, 0),
                "valid_at_4096": values.get(4096, 0),
                "min_required_max_len": min_required,
                "example_length": diagnosis_by_family.get(family_id, {}).get("example_length", ""),
            }
        )

    missing_family_rows = []
    for row in missing_rows:
        family_id = row["family"]
        missing_family_rows.append(
            {
                "family": family_id,
                "expected_paths_checked": ";".join(str(path) for path in candidate_file_paths(args.raw_dir, family_id)),
                "status": "file_missing",
            }
        )

    summary = {
        "raw_dir": str(args.raw_dir),
        "diagnosis_csv": str(args.diagnosis_csv),
        "min_len": args.min_len,
        "allowed_chars": args.allowed_chars,
        "max_lens": max_lens,
        "caps": caps,
        "status_counts_from_diagnosis": dict(Counter(row["primary_status"] for row in diagnosis_rows)),
        "per_max_len": {},
        "too_long_only_families": too_long_families,
        "removed_by_dedup_families": removed_by_dedup_families,
        "missing_family_count": len(missing_rows),
        "too_long_recovery_counts": recovery_counts,
        "warnings": warnings_all,
    }

    for max_len in max_lens:
        strict_result = per_maxlen_results[max_len]["strict_global_dedup"]
        preserving_result = per_maxlen_results[max_len]["family_preserving_dedup"]
        strict_families = {family for family, count in strict_result["counts"].items() if count > 0}
        preserving_families = {family for family, count in preserving_result["counts"].items() if count > 0}
        summary["per_max_len"][str(max_len)] = {
            "valid_families_before_global_dedup": preserving_result["covered_families"],
            "valid_families_after_global_dedup": strict_result["covered_families"],
            "families_recovered_vs_512_strict": len(strict_families - baseline_recovered["strict_global_dedup"]),
            "families_recovered_vs_512_preserving": len(
                preserving_families - baseline_recovered["family_preserving_dedup"]
            ),
            "strict_global_dedup": {
                key: value
                for key, value in strict_result.items()
                if key != "counts"
            },
            "family_preserving_dedup": {
                key: value
                for key, value in preserving_result.items()
                if key != "counts"
            },
        }

    output_dir = args.output_dir
    summary_json_path = output_dir / "rfam_coverage_repair_summary.json"
    by_maxlen_csv_path = output_dir / "rfam_coverage_repair_by_maxlen.csv"
    too_long_csv_path = output_dir / "rfam_too_long_recovery.csv"
    missing_csv_path = output_dir / "rfam_missing_families.csv"
    conflicts_csv_path = output_dir / "rfam_dedup_conflicts.csv"

    write_json(summary_json_path, summary)
    write_csv(
        by_maxlen_csv_path,
        [
            "max_len",
            "dedup_strategy",
            "covered_families",
            "total_sequences",
            "families_recovered_vs_512",
            *[f"families_with_at_least_{threshold}" for threshold in COUNT_THRESHOLDS],
            *[f"capped_{cap}_total" for cap in caps],
        ],
        maxlen_rows,
    )
    write_csv(
        too_long_csv_path,
        ["family", "valid_at_512", "valid_at_1024", "valid_at_2048", "valid_at_4096", "min_required_max_len", "example_length"],
        too_long_rows,
    )
    write_csv(
        missing_csv_path,
        ["family", "expected_paths_checked", "status"],
        missing_family_rows,
    )
    write_csv(
        conflicts_csv_path,
        ["max_len", "sequence_hash", "sequence_length", "first_family", "conflict_families", "conflict_count"],
        conflict_rows_all,
    )

    print("[OK] Coverage repair evaluation complete")
    baseline = summary["per_max_len"][str(max_lens[0])]
    print(f"[BASELINE] max_len={max_lens[0]} covered_families={baseline['valid_families_after_global_dedup']}")
    for max_len in max_lens[1:]:
        item = summary["per_max_len"][str(max_len)]
        print(
            f"[MAX_LEN {max_len}] covered_families={item['valid_families_after_global_dedup']} "
            f"recovered={item['families_recovered_vs_512_strict']}"
        )
    baseline_strict = summary["per_max_len"][str(max_lens[0])]["strict_global_dedup"]
    baseline_preserving = summary["per_max_len"][str(max_lens[0])]["family_preserving_dedup"]
    print(f"[DEDUP] strict_global_dedup lost_families={baseline_strict['families_lost_by_dedup']}")
    print(f"[DEDUP] family_preserving_dedup lost_families={baseline_preserving['families_lost_by_dedup']}")
    print(f"[MISSING] file_missing={len(missing_rows)}")
    print(f"[OUTPUT] {summary_json_path}")


if __name__ == "__main__":
    main()
