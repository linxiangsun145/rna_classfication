#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


MODEL_NAMES = ["cnn", "mamba"]
RESULT_FILES = [
    "metrics.json",
    "confusion_matrix.csv",
    "per_class_metrics.json",
    "length_bucket_metrics.json",
]
LENGTH_BUCKETS = ["short", "medium", "long"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CNN vs Mamba experiment results from the runs directory."
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing cnn_run and mamba_run results.",
    )
    return parser.parse_args()


def load_json(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(file_path: Path, payload: Dict[str, Any]) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_model_results(runs_dir: Path, model_name: str) -> Dict[str, Any]:
    run_dir = runs_dir / f"{model_name}_run"
    payload: Dict[str, Any] = {
        "model": model_name,
        "run_dir": str(run_dir),
        "available": True,
        "missing_files": [],
    }

    for filename in RESULT_FILES:
        path = run_dir / filename
        if not path.exists():
            payload["missing_files"].append(filename)

    if payload["missing_files"]:
        payload["available"] = False
        return payload

    payload["metrics"] = load_json(run_dir / "metrics.json")
    payload["per_class_metrics"] = load_json(run_dir / "per_class_metrics.json")
    payload["length_bucket_metrics"] = load_json(run_dir / "length_bucket_metrics.json")
    payload["confusion_matrix"] = pd.read_csv(run_dir / "confusion_matrix.csv", index_col=0)
    return payload


def load_all_results(runs_dir: Path) -> Dict[str, Dict[str, Any]]:
    results = {}
    for model_name in MODEL_NAMES:
        model_result = load_model_results(runs_dir, model_name)
        if model_result["available"]:
            print(f"[OK] Loaded {model_name}_run metrics")
        else:
            missing = ", ".join(model_result["missing_files"])
            print(f"[WARN] Missing {model_name}_run artifacts: {missing}")
        results[model_name] = model_result
    return results


def build_overall_comparison(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {"cnn": None, "mamba": None, "difference": None}
    for model_name in MODEL_NAMES:
        if results[model_name]["available"]:
            metrics = results[model_name]["metrics"]
            comparison[model_name] = {
                "test_accuracy": float(metrics["test_accuracy"]),
                "test_macro_f1": float(metrics["test_macro_f1"]),
            }

    if comparison["cnn"] and comparison["mamba"]:
        acc_diff = comparison["mamba"]["test_accuracy"] - comparison["cnn"]["test_accuracy"]
        f1_diff = comparison["mamba"]["test_macro_f1"] - comparison["cnn"]["test_macro_f1"]
        if f1_diff > 0:
            better_model = "mamba"
        elif f1_diff < 0:
            better_model = "cnn"
        else:
            better_model = "tie"
        comparison["difference"] = {
            "acc_diff": float(acc_diff),
            "f1_diff": float(f1_diff),
            "better_model": better_model,
        }
    return comparison


def build_per_class_analysis(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {
        "classes": {},
        "mamba_top3_improvements": [],
        "mamba_top3_declines": [],
    }
    if not (results["cnn"]["available"] and results["mamba"]["available"]):
        return analysis

    cnn_metrics = results["cnn"]["per_class_metrics"]
    mamba_metrics = results["mamba"]["per_class_metrics"]
    diffs: List[Dict[str, Any]] = []

    families = sorted(set(cnn_metrics) | set(mamba_metrics))
    for family in families:
        cnn_family = cnn_metrics.get(family, {})
        mamba_family = mamba_metrics.get(family, {})
        cnn_f1 = float(cnn_family.get("f1", 0.0))
        mamba_f1 = float(mamba_family.get("f1", 0.0))
        diff = mamba_f1 - cnn_f1
        analysis["classes"][family] = {
            "cnn_precision": float(cnn_family.get("precision", 0.0)),
            "cnn_recall": float(cnn_family.get("recall", 0.0)),
            "cnn_f1": cnn_f1,
            "mamba_precision": float(mamba_family.get("precision", 0.0)),
            "mamba_recall": float(mamba_family.get("recall", 0.0)),
            "mamba_f1": mamba_f1,
            "diff": float(diff),
        }
        diffs.append({"family": family, "f1_diff": float(diff)})

    sorted_desc = sorted(diffs, key=lambda item: item["f1_diff"], reverse=True)
    sorted_asc = sorted(diffs, key=lambda item: item["f1_diff"])
    analysis["mamba_top3_improvements"] = sorted_desc[:3]
    analysis["mamba_top3_declines"] = sorted_asc[:3]
    return analysis


def _top_confusions(confusion_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for actual_label in confusion_df.index:
        for predicted_label in confusion_df.columns:
            count = int(confusion_df.loc[actual_label, predicted_label])
            if actual_label == predicted_label or count == 0:
                continue
            rows.append(
                {
                    "actual": str(actual_label),
                    "predicted": str(predicted_label),
                    "count": count,
                }
            )
    return sorted(rows, key=lambda item: item["count"], reverse=True)[:top_k]


def build_confusion_analysis(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {
        "cnn_top_confusions": [],
        "mamba_top_confusions": [],
        "difference_note": None,
    }
    if results["cnn"]["available"]:
        analysis["cnn_top_confusions"] = _top_confusions(results["cnn"]["confusion_matrix"])
    if results["mamba"]["available"]:
        analysis["mamba_top_confusions"] = _top_confusions(results["mamba"]["confusion_matrix"])

    if analysis["cnn_top_confusions"] and analysis["mamba_top_confusions"]:
        cnn_pairs = {
            (item["actual"], item["predicted"]): item["count"]
            for item in analysis["cnn_top_confusions"]
        }
        mamba_pairs = {
            (item["actual"], item["predicted"]): item["count"]
            for item in analysis["mamba_top_confusions"]
        }
        shared_pairs = sorted(set(cnn_pairs) & set(mamba_pairs))
        if shared_pairs:
            pair_summaries = []
            for pair in shared_pairs:
                pair_summaries.append(
                    {
                        "actual": pair[0],
                        "predicted": pair[1],
                        "cnn_count": int(cnn_pairs[pair]),
                        "mamba_count": int(mamba_pairs[pair]),
                    }
                )
            analysis["difference_note"] = {
                "shared_confusions": pair_summaries,
                "cnn_only_pairs": [
                    {"actual": a, "predicted": p, "count": int(c)}
                    for (a, p), c in cnn_pairs.items()
                    if (a, p) not in mamba_pairs
                ],
                "mamba_only_pairs": [
                    {"actual": a, "predicted": p, "count": int(c)}
                    for (a, p), c in mamba_pairs.items()
                    if (a, p) not in cnn_pairs
                ],
            }
    return analysis


def build_length_analysis(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {bucket: {"cnn": None, "mamba": None, "difference": None} for bucket in LENGTH_BUCKETS}
    for bucket in LENGTH_BUCKETS:
        cnn_bucket = None
        mamba_bucket = None
        if results["cnn"]["available"]:
            cnn_bucket = results["cnn"]["length_bucket_metrics"].get(bucket)
            analysis[bucket]["cnn"] = cnn_bucket
        if results["mamba"]["available"]:
            mamba_bucket = results["mamba"]["length_bucket_metrics"].get(bucket)
            analysis[bucket]["mamba"] = mamba_bucket
        if cnn_bucket and mamba_bucket:
            analysis[bucket]["difference"] = {
                "accuracy_diff": float(mamba_bucket["accuracy"] - cnn_bucket["accuracy"]),
                "macro_f1_diff": float(mamba_bucket["macro_f1"] - cnn_bucket["macro_f1"]),
            }

    long_bucket = analysis["long"]
    if long_bucket["difference"] is not None:
        long_bucket["mamba_advantage"] = bool(long_bucket["difference"]["macro_f1_diff"] > 0)
    else:
        long_bucket["mamba_advantage"] = None
    return analysis


def _summarize_dynamics(model_name: str, metrics: Dict[str, Any]) -> str:
    history = metrics.get("history", {})
    val_loss = history.get("val_loss", [])
    val_f1 = history.get("val_macro_f1", [])
    if len(val_loss) <= 1 or len(val_f1) <= 1:
        return (
            f"{model_name} only has {len(val_loss)} epoch(s) recorded, so convergence and "
            "stability cannot be judged reliably."
        )

    loss_moves = [val_loss[idx] - val_loss[idx - 1] for idx in range(1, len(val_loss))]
    f1_moves = [val_f1[idx] - val_f1[idx - 1] for idx in range(1, len(val_f1))]
    loss_increases = sum(1 for delta in loss_moves if delta > 0)
    f1_drops = sum(1 for delta in f1_moves if delta < 0)

    if loss_increases == 0 and f1_drops == 0:
        return f"{model_name} shows smooth validation behavior with no obvious oscillation."
    if loss_increases <= 1 and f1_drops <= 1:
        return f"{model_name} is mostly stable, with only minor validation fluctuation."
    return f"{model_name} shows noticeable validation oscillation and may need more careful tuning."


def build_training_dynamics(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    analysis = {"cnn": None, "mamba": None}
    for model_name in MODEL_NAMES:
        if results[model_name]["available"]:
            analysis[model_name] = _summarize_dynamics(
                model_name, results[model_name]["metrics"]
            )
    return analysis


def build_conclusion(
    overall: Dict[str, Any],
    per_class: Dict[str, Any],
    length_analysis: Dict[str, Any],
    confusion_analysis: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    conclusion = {
        "which_model_better": "insufficient_data",
        "where_mamba_wins": "Unavailable because one side of the comparison is missing.",
        "where_cnn_wins": "Unavailable because one side of the comparison is missing.",
        "key_observation": "Only partial experiment artifacts are available.",
    }

    if overall.get("difference") is None:
        available_models = [name for name in MODEL_NAMES if results[name]["available"]]
        if available_models == ["cnn"]:
            conclusion["which_model_better"] = "cnn_only_available"
            conclusion["key_observation"] = (
                "Only CNN results are present in the current runs directory, so no direct "
                "CNN vs Mamba conclusion can be drawn yet."
            )
        elif available_models == ["mamba"]:
            conclusion["which_model_better"] = "mamba_only_available"
            conclusion["key_observation"] = (
                "Only Mamba results are present in the current runs directory, so no direct "
                "CNN vs Mamba conclusion can be drawn yet."
            )
        return conclusion

    better_model = overall["difference"]["better_model"]
    conclusion["which_model_better"] = better_model

    improvements = per_class.get("mamba_top3_improvements", [])
    declines = per_class.get("mamba_top3_declines", [])
    if improvements:
        conclusion["where_mamba_wins"] = ", ".join(
            f"{item['family']} (f1_diff={item['f1_diff']:.4f})" for item in improvements
        )
    if declines:
        conclusion["where_cnn_wins"] = ", ".join(
            f"{item['family']} (f1_diff={item['f1_diff']:.4f})" for item in declines
        )

    long_advantage = length_analysis["long"].get("mamba_advantage")
    if long_advantage is True:
        length_note = "Mamba shows an advantage on long sequences."
    elif long_advantage is False:
        length_note = "Mamba does not show an advantage on long sequences."
    else:
        length_note = "Long-sequence comparison is unavailable."

    top_confusions = confusion_analysis.get("cnn_top_confusions", [])
    confusion_note = (
        f"The strongest observed CNN confusion is {top_confusions[0]['actual']} -> "
        f"{top_confusions[0]['predicted']} ({top_confusions[0]['count']})."
        if top_confusions
        else "Confusion analysis is limited by missing artifacts."
    )
    conclusion["key_observation"] = f"{length_note} {confusion_note}"
    return conclusion


def build_text_report(report: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("CNN vs Mamba Experiment Analysis")
    lines.append("")

    overall = report["overall_comparison"]
    if overall["cnn"] is not None:
        lines.append(
            f"CNN test accuracy = {overall['cnn']['test_accuracy']:.4f}, "
            f"macro-F1 = {overall['cnn']['test_macro_f1']:.4f}."
        )
    if overall["mamba"] is not None:
        lines.append(
            f"Mamba test accuracy = {overall['mamba']['test_accuracy']:.4f}, "
            f"macro-F1 = {overall['mamba']['test_macro_f1']:.4f}."
        )
    if overall["difference"] is not None:
        lines.append(
            f"Overall winner: {overall['difference']['better_model']}, "
            f"acc diff = {overall['difference']['acc_diff']:.4f}, "
            f"F1 diff = {overall['difference']['f1_diff']:.4f}."
        )
    else:
        lines.append("Direct overall comparison is incomplete because one model result set is missing.")

    lines.append("")
    lines.append("Per-class observations:")
    per_class = report["per_class_analysis"]
    if per_class["mamba_top3_improvements"]:
        lines.append(
            "Mamba top improvements: "
            + "; ".join(
                f"{item['family']} ({item['f1_diff']:.4f})"
                for item in per_class["mamba_top3_improvements"]
            )
        )
        lines.append(
            "Mamba largest declines: "
            + "; ".join(
                f"{item['family']} ({item['f1_diff']:.4f})"
                for item in per_class["mamba_top3_declines"]
            )
        )
    else:
        lines.append("Per-class comparison is not available yet.")

    lines.append("")
    lines.append("Length-bucket observations:")
    for bucket in LENGTH_BUCKETS:
        bucket_data = report["length_analysis"][bucket]
        cnn_bucket = bucket_data.get("cnn")
        mamba_bucket = bucket_data.get("mamba")
        if cnn_bucket and mamba_bucket:
            lines.append(
                f"{bucket}: CNN acc={cnn_bucket['accuracy']:.4f}, F1={cnn_bucket['macro_f1']:.4f}; "
                f"Mamba acc={mamba_bucket['accuracy']:.4f}, F1={mamba_bucket['macro_f1']:.4f}."
            )
        elif cnn_bucket or mamba_bucket:
            available_name = "CNN" if cnn_bucket else "Mamba"
            available_bucket = cnn_bucket if cnn_bucket else mamba_bucket
            lines.append(
                f"{bucket}: only {available_name} is available "
                f"(acc={available_bucket['accuracy']:.4f}, F1={available_bucket['macro_f1']:.4f})."
            )
        else:
            lines.append(f"{bucket}: no available results.")

    lines.append("")
    lines.append("Training dynamics:")
    for model_name in MODEL_NAMES:
        model_note = report["training_dynamics"].get(model_name)
        if model_note:
            lines.append(f"{model_name}: {model_note}")

    lines.append("")
    lines.append("Conclusion:")
    conclusion = report["conclusion"]
    lines.append(f"which_model_better: {conclusion['which_model_better']}")
    lines.append(f"where_mamba_wins: {conclusion['where_mamba_wins']}")
    lines.append(f"where_cnn_wins: {conclusion['where_cnn_wins']}")
    lines.append(f"key_observation: {conclusion['key_observation']}")

    missing_models = [name for name in MODEL_NAMES if not results[name]["available"]]
    if missing_models:
        lines.append("")
        lines.append(
            "Missing artifacts: "
            + "; ".join(
                f"{name}_run missing {', '.join(results[name]['missing_files'])}"
                for name in missing_models
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir
    results = load_all_results(runs_dir)

    report = {
        "overall_comparison": build_overall_comparison(results),
        "per_class_analysis": build_per_class_analysis(results),
        "confusion_analysis": build_confusion_analysis(results),
        "length_analysis": build_length_analysis(results),
        "training_dynamics": build_training_dynamics(results),
    }
    report["conclusion"] = build_conclusion(
        overall=report["overall_comparison"],
        per_class=report["per_class_analysis"],
        length_analysis=report["length_analysis"],
        confusion_analysis=report["confusion_analysis"],
        results=results,
    )

    analysis_json_path = runs_dir / "analysis_report.json"
    analysis_txt_path = runs_dir / "analysis_report.txt"
    save_json(analysis_json_path, report)
    analysis_txt_path.write_text(build_text_report(report, results), encoding="utf-8")

    print("[OK] Analysis complete")
    print(f"[RESULT] Report saved to {analysis_json_path}")
    print(f"[RESULT] Report saved to {analysis_txt_path}")


if __name__ == "__main__":
    main()
