from __future__ import annotations

import csv
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from flask import Flask, render_template, request, send_from_directory
    from werkzeug.utils import secure_filename
except ImportError as exc:
    raise SystemExit(
        "Flask is not installed. Install the project requirements before running app.py."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PREDICTION_DIR = BASE_DIR / "predictions" / "web"
EXAMPLES_DIR = BASE_DIR / "examples"
RFAM_METADATA_PATH = BASE_DIR / "data" / "rfam_family_metadata.csv"
PREDICT_SCRIPT = BASE_DIR / "scripts" / "predict_rna_family.py"
WSL_PROJECT_DIR = "/mnt/d/vibe_coding/rna_classfication"
WSL_VENV_PYTHON = f"{WSL_PROJECT_DIR}/.wsl_mamba_env/bin/python"
ALLOWED_EXTENSIONS = {".fasta", ".fa", ".csv"}
ALLOWED_EXAMPLE_FILES = {"test_sequences.fasta", "test_sequences.csv"}
DISPLAY_COLUMNS = [
    "sequence_id",
    "seq_len",
    "predicted_family",
    "predicted_family_annotation",
    "confidence",
    "risk_flag",
    "top1_family",
    "top1_prob",
    "top2_family",
    "top2_prob",
    "top3_family",
    "top3_prob",
]
MODEL_CONFIGS = {
    "stable_1669": {
        "key": "stable_1669",
        "display_name": "Stable model (1669 families)",
        "model_path": BASE_DIR / "runs_rfam_1669" / "mamba_run" / "best_model.pt",
        "label_mapping": BASE_DIR / "processed_rfam_1669" / "label_mapping.json",
        "max_len": 512,
        "description": "Recommended for demonstration. Higher reliability.",
        "family_count": 1669,
        "is_default": True,
    },
    "high_coverage_2238": {
        "key": "high_coverage_2238",
        "display_name": "High-coverage model (2238 families, experimental)",
        "model_path": BASE_DIR
        / "runs_rfam_v2_1_baseline_ce"
        / "mamba_run"
        / "best_model.pt",
        "label_mapping": BASE_DIR
        / "processed_rfam_near_full_v2_1_min20_1024"
        / "label_mapping.json",
        "max_len": 1024,
        "description": (
            "Covers more Rfam families but may be less stable for low-support families."
        ),
        "family_count": 2238,
        "is_default": False,
    },
}

app = Flask(__name__)


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_default_model_key() -> str:
    for key, config in MODEL_CONFIGS.items():
        if config.get("is_default"):
            return key
    return "stable_1669"


def get_model_config(model_key: str | None) -> dict[str, Any]:
    if model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[str(model_key)]
    return MODEL_CONFIGS[get_default_model_key()]


def get_model_options() -> list[dict[str, Any]]:
    return list(MODEL_CONFIGS.values())


def build_job_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}"


def format_display_value(column: str, value: str) -> str:
    if column in {"confidence", "top1_prob", "top2_prob", "top3_prob"}:
        if value == "":
            return ""
        try:
            return f"{float(value):.4f}"
        except ValueError:
            return value
    return value


def load_rfam_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    metadata: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                family = str(row.get("family", "")).strip()
                if not family:
                    continue
                metadata[family] = {
                    "short_name": str(row.get("short_name", "")).strip() or family,
                    "description": str(row.get("description", "")).strip() or family,
                    "source": str(row.get("source", "")).strip() or "fallback",
                }
    except Exception:
        return {}
    return metadata


def format_family_annotation(family: str, metadata_map: dict[str, dict[str, str]]) -> str:
    family = str(family).strip()
    if not family:
        return ""
    meta = metadata_map.get(family)
    if meta is None:
        return family
    description = meta.get("description", "").strip()
    if not description or description == family:
        return family
    return f"{family} - {description}"


def classify_confidence(value: str) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def normalize_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_prediction_summary(rows: list[dict[str, str]]) -> dict[str, str | int]:
    summary = {
        "total": len(rows),
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "truncated": 0,
        "most_frequent_family": "N/A",
    }
    if not rows:
        return summary

    family_counter: Counter[str] = Counter()
    for row in rows:
        confidence_level = classify_confidence(row.get("confidence", ""))
        if confidence_level == "high":
            summary["high_confidence"] += 1
        elif confidence_level == "medium":
            summary["medium_confidence"] += 1
        elif confidence_level == "low":
            summary["low_confidence"] += 1

        if normalize_bool(row.get("was_truncated", "")):
            summary["truncated"] += 1

        predicted_family = str(row.get("predicted_family", "")).strip()
        if predicted_family:
            family_counter[predicted_family] += 1

    if family_counter:
        summary["most_frequent_family"] = family_counter.most_common(1)[0][0]
    return summary


def build_display_rows(
    rows: list[dict[str, str]],
    metadata_map: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    display_rows: list[dict[str, str]] = []
    for row in rows:
        display_row = {
            column: format_display_value(column, row.get(column, "")) for column in DISPLAY_COLUMNS
        }
        predicted_family = str(row.get("predicted_family", "")).strip()
        display_row["predicted_family_annotation"] = format_family_annotation(
            predicted_family,
            metadata_map,
        )
        confidence_level = classify_confidence(row.get("confidence", ""))
        display_row["confidence_level"] = (
            confidence_level.title() if confidence_level != "unknown" else "Unknown"
        )
        display_row["confidence_badge_class"] = f"badge-{confidence_level}"

        risk_flag = str(row.get("risk_flag", "")).strip() or "unknown"
        display_row["risk_badge_class"] = "badge-ok" if risk_flag == "ok" else "badge-risk"
        display_rows.append(display_row)
    return display_rows


def read_prediction_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {column: raw_row.get(column, "") for column in DISPLAY_COLUMNS}
            row["was_truncated"] = raw_row.get("was_truncated", "")
            rows.append(row)
    return rows


def run_prediction(
    input_path: Path,
    output_path: Path,
    model_config: dict[str, Any],
) -> tuple[bool, str]:
    command = build_prediction_command(input_path, output_path, model_config)

    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Prediction failed."
        return False, message
    return True, completed.stdout.strip()


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def should_use_wsl_runtime() -> bool:
    return os.name == "nt" and shutil.which("wsl") is not None and (BASE_DIR / ".wsl_mamba_env").exists()


def build_wsl_prediction_command(
    input_path: Path,
    output_path: Path,
    model_config: dict[str, Any],
) -> list[str]:
    predict_args = [
        WSL_VENV_PYTHON,
        f"{WSL_PROJECT_DIR}/scripts/predict_rna_family.py",
        "--input",
        windows_path_to_wsl(input_path),
        "--output",
        windows_path_to_wsl(output_path),
        "--model_path",
        windows_path_to_wsl(Path(model_config["model_path"])),
        "--label_mapping",
        windows_path_to_wsl(Path(model_config["label_mapping"])),
        "--model_type",
        "mamba",
        "--max_len",
        str(model_config["max_len"]),
        "--batch_size",
        "64",
        "--top_k",
        "3",
        "--device",
        "cuda",
    ]
    quoted_args = " ".join(shlex.quote(arg) for arg in predict_args)
    shell_command = " && ".join(
        [
            f"cd {shlex.quote(WSL_PROJECT_DIR)}",
            "export CUDA_HOME=/usr/local/cuda-12.8",
            "export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin:$PATH",
            "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
            quoted_args,
        ]
    )
    return ["wsl", "-d", "Ubuntu", "bash", "-lc", shell_command]


def build_local_prediction_command(
    input_path: Path,
    output_path: Path,
    model_config: dict[str, Any],
) -> list[str]:
    return [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model_path",
        str(model_config["model_path"]),
        "--label_mapping",
        str(model_config["label_mapping"]),
        "--model_type",
        "mamba",
        "--max_len",
        str(model_config["max_len"]),
        "--batch_size",
        "64",
        "--top_k",
        "3",
        "--device",
        "cuda",
    ]


def build_prediction_command(
    input_path: Path,
    output_path: Path,
    model_config: dict[str, Any],
) -> list[str]:
    if should_use_wsl_runtime():
        return build_wsl_prediction_command(input_path, output_path, model_config)
    return build_local_prediction_command(input_path, output_path, model_config)


def validate_runtime_paths(model_config: dict[str, Any]) -> str | None:
    missing = []
    if not PREDICT_SCRIPT.exists():
        missing.append(str(PREDICT_SCRIPT))
    if not Path(model_config["model_path"]).exists():
        missing.append(str(model_config["model_path"]))
    if not Path(model_config["label_mapping"]).exists():
        missing.append(str(model_config["label_mapping"]))
    if missing:
        return "Missing required runtime files: " + ", ".join(missing)
    return None


def build_selected_model_context(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_model_name": model_config["display_name"],
        "selected_model_key": model_config["key"],
        "selected_model_description": model_config["description"],
        "selected_model_max_len": model_config["max_len"],
        "selected_model_family_count": model_config["family_count"],
        "selected_model_is_experimental": model_config["key"] == "high_coverage_2238",
    }


def render_index(error_message: str | None, selected_model_key: str, status_code: int = 200):
    return (
        render_template(
            "index.html",
            error_message=error_message,
            model_options=get_model_options(),
            selected_model_key=selected_model_key,
        ),
        status_code,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        error_message=None,
        model_options=get_model_options(),
        selected_model_key=get_default_model_key(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    ensure_directories()

    selected_model = get_model_config(request.form.get("model_key"))
    runtime_error = validate_runtime_paths(selected_model)
    if runtime_error is not None:
        return render_index(runtime_error, selected_model["key"], status_code=500)

    if "file" not in request.files:
        return render_index("No file was uploaded.", selected_model["key"], status_code=400)

    uploaded_file = request.files["file"]
    if uploaded_file.filename is None or uploaded_file.filename.strip() == "":
        return render_index(
            "Please choose a FASTA or CSV file.",
            selected_model["key"],
            status_code=400,
        )

    original_name = uploaded_file.filename
    if not allowed_file(original_name):
        return render_index(
            "Unsupported file type. Please upload .fasta, .fa, or .csv.",
            selected_model["key"],
            status_code=400,
        )

    file_bytes = uploaded_file.read()
    if not file_bytes:
        return render_index(
            "The uploaded file is empty.",
            selected_model["key"],
            status_code=400,
        )

    job_id = build_job_id()
    safe_name = secure_filename(original_name)
    upload_name = f"{job_id}_{safe_name}"
    upload_path = UPLOAD_DIR / upload_name
    upload_path.write_bytes(file_bytes)

    output_name = f"prediction_{job_id}_{selected_model['key']}.csv"
    output_path = PREDICTION_DIR / output_name

    ok, message = run_prediction(upload_path, output_path, selected_model)
    if not ok:
        return render_index(
            (
                "Prediction failed. If you are not running inside the WSL Mamba environment, "
                "mamba-ssm may be unavailable.\n\n"
                f"{message}"
            ),
            selected_model["key"],
            status_code=500,
        )

    if not output_path.exists():
        return render_index(
            "Prediction finished but no output CSV was created.",
            selected_model["key"],
            status_code=500,
        )

    rows = read_prediction_rows(output_path)
    if not rows:
        return render_index(
            "Prediction output is empty.",
            selected_model["key"],
            status_code=500,
        )

    metadata_map = load_rfam_metadata(RFAM_METADATA_PATH)
    summary = build_prediction_summary(rows)
    display_rows = build_display_rows(rows, metadata_map)

    return render_template(
        "result.html",
        original_name=original_name,
        output_name=output_name,
        row_count=len(display_rows),
        summary=summary,
        rows=display_rows,
        columns=DISPLAY_COLUMNS,
        message=message,
        **build_selected_model_context(selected_model),
    )


@app.route("/download/<path:filename>", methods=["GET"])
def download(filename: str):
    safe_filename = Path(filename).name
    return send_from_directory(PREDICTION_DIR, safe_filename, as_attachment=True)


@app.route("/examples/<path:filename>", methods=["GET"])
def download_example(filename: str):
    safe_filename = Path(filename).name
    if safe_filename not in ALLOWED_EXAMPLE_FILES:
        return "Example file not found.", 404

    example_path = EXAMPLES_DIR / safe_filename
    if not example_path.exists():
        return f"Missing example file: {safe_filename}", 404

    return send_from_directory(EXAMPLES_DIR, safe_filename, as_attachment=True)


if __name__ == "__main__":
    ensure_directories()
    app.run(host="0.0.0.0", port=5000, debug=False)
