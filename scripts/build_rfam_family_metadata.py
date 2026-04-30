#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


KNOWN_FAMILY_METADATA: dict[str, tuple[str, str]] = {
    "RF00001": ("5S rRNA", "5S ribosomal RNA"),
    "RF00002": ("5.8S rRNA", "5.8S ribosomal RNA"),
    "RF00003": ("U1", "U1 spliceosomal RNA"),
    "RF00004": ("U2", "U2 spliceosomal RNA"),
    "RF00005": ("tRNA", "Transfer RNA family"),
    "RF00006": ("Vault RNA", "Vault RNA"),
    "RF00007": ("RNase P", "RNase P RNA"),
    "RF00008": ("Hammerhead", "Hammerhead ribozyme"),
    "RF00009": ("IRES", "Internal ribosome entry site family"),
    "RF00010": ("Telomerase", "Telomerase RNA"),
    "RF00011": ("SRP", "Signal recognition particle RNA"),
    "RF00012": ("U3", "U3 small nucleolar RNA"),
    "RF00013": ("U4", "U4 spliceosomal RNA"),
    "RF00014": ("U5", "U5 spliceosomal RNA"),
    "RF00015": ("U6", "U6 spliceosomal RNA"),
    "RF00016": ("6S", "Bacterial 6S RNA"),
    "RF00017": ("Metazoan SRP", "Metazoan signal recognition particle RNA"),
    "RF00018": ("TmRNA", "Transfer-messenger RNA"),
    "RF00174": ("SAM", "SAM riboswitch family"),
    "RF00177": ("CsrB", "CsrB and related bacterial small RNAs"),
}

SUPPORTED_SUFFIXES = (".fa", ".fasta", ".fa.gz", ".fasta.gz")
HEADER_SCAN_LIMIT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight Rfam family metadata table for demo display."
    )
    parser.add_argument("--raw_dir", type=Path, required=True, help="Directory containing Rfam FASTA files.")
    parser.add_argument(
        "--label_mapping",
        type=Path,
        required=True,
        help="Path to label_mapping.json for selecting target families.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output metadata CSV path.")
    return parser.parse_args()


def load_family_ids(label_mapping_path: Path) -> list[str]:
    with label_mapping_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "family_to_label" in payload:
        families = list(payload["family_to_label"].keys())
    else:
        families = list(payload.keys())
    return sorted(str(family) for family in families)


def find_family_file(raw_dir: Path, family_id: str) -> Path | None:
    for suffix in SUPPORTED_SUFFIXES:
        candidate = raw_dir / f"{family_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def parse_annotation_from_header_line(header: str, family_id: str) -> tuple[str, str] | None:
    text = header.strip().lstrip(">").strip()
    if not text:
        return None

    tokens = text.split()
    if len(tokens) <= 1:
        return None

    remainder = " ".join(tokens[1:]).strip(" -|;:,")
    if not remainder:
        return None
    if remainder.startswith(family_id):
        return None
    if len(remainder) < 3:
        return None

    short_name = remainder.split(";")[0].split("|")[0].strip()
    if len(short_name) > 80:
        short_name = short_name[:80].rstrip()
    description = remainder
    if len(description) > 240:
        description = description[:240].rstrip()
    return short_name, description


def parse_from_fasta_headers(path: Path, family_id: str) -> tuple[str, str] | None:
    try:
        with open_text_file(path) as handle:
            scanned = 0
            for line in handle:
                if not line.startswith(">"):
                    continue
                scanned += 1
                parsed = parse_annotation_from_header_line(line, family_id)
                if parsed is not None:
                    return parsed
                if scanned >= HEADER_SCAN_LIMIT:
                    break
    except Exception:
        return None
    return None


def build_metadata_rows(raw_dir: Path, family_ids: list[str]) -> tuple[list[dict[str, str]], int, int, int]:
    rows: list[dict[str, str]] = []
    parsed_from_header = 0
    parsed_from_builtin = 0
    fallback_count = 0

    for family_id in family_ids:
        source_file = find_family_file(raw_dir, family_id)
        parsed = parse_from_fasta_headers(source_file, family_id) if source_file is not None else None

        if parsed is not None:
            short_name, description = parsed
            source = "header"
            parsed_from_header += 1
        elif family_id in KNOWN_FAMILY_METADATA:
            short_name, description = KNOWN_FAMILY_METADATA[family_id]
            source = "builtin"
            parsed_from_builtin += 1
        else:
            short_name = family_id
            description = family_id
            source = "fallback"
            fallback_count += 1

        rows.append(
            {
                "family": family_id,
                "short_name": short_name,
                "description": description,
                "source": source,
            }
        )

    return rows, parsed_from_header, parsed_from_builtin, fallback_count


def write_metadata(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family", "short_name", "description", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    family_ids = load_family_ids(args.label_mapping)
    rows, parsed_from_header, parsed_from_builtin, fallback_count = build_metadata_rows(
        args.raw_dir,
        family_ids,
    )
    write_metadata(rows, args.output)

    print(f"[OK] total families = {len(rows)}")
    print(f"[OK] parsed from header count = {parsed_from_header}")
    print(f"[OK] parsed from builtin count = {parsed_from_builtin}")
    print(f"[OK] fallback count = {fallback_count}")
    print(f"[OK] output path = {args.output}")


if __name__ == "__main__":
    main()
