from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev


DEFAULT_SUMMARY_FILES = (
    "fixed_budget_frozen.summary.json",
    "sequential_policy_keyword_frozen.summary.json",
    "sequential_policy_frozen.summary.json",
)

DEFAULT_METRICS = (
    "accuracy",
    "selected_evidence_cost",
    "selected_evidence_count",
    "selected_temporal_iou",
)


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_method_metrics(
    run_roots: list[str | Path],
    summary_files: tuple[str, ...] = DEFAULT_SUMMARY_FILES,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, dict]:
    aggregated: dict[str, dict] = {}

    for summary_name in summary_files:
        values_by_metric: dict[str, list[float]] = {metric: [] for metric in metrics}
        contributing_runs: list[str] = []

        for run_root in run_roots:
            summary_path = Path(run_root) / "outputs" / summary_name
            if not summary_path.is_file():
                continue

            payload = load_summary(summary_path)
            summary_metrics = payload.get("metrics", {})
            contributing_runs.append(str(run_root))
            for metric in metrics:
                value = summary_metrics.get(metric)
                if value is None:
                    continue
                values_by_metric[metric].append(float(value))

        if not contributing_runs:
            continue

        aggregated[summary_name] = {
            "num_runs": len(contributing_runs),
            "run_roots": contributing_runs,
            "metrics": {
                metric: summarize_metric(values)
                for metric, values in values_by_metric.items()
                if values
            },
        }

    return aggregated


def summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "num_runs": 0,
        }

    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "num_runs": len(values),
    }


def format_mean_std(value_mean: float, value_std: float, decimals: int = 3) -> str:
    if math.isnan(value_mean):
        return "n/a"
    return f"{value_mean:.{decimals}f} +/- {value_std:.{decimals}f}"


def format_markdown_table(
    aggregated: dict[str, dict],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    *,
    decimals: int = 3,
) -> str:
    headers = ["Method", "Runs"] + [metric.replace("_", " ").title() for metric in metrics]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for method_name, payload in aggregated.items():
        metric_payload = payload.get("metrics", {})
        row = [method_name, str(payload.get("num_runs", 0))]
        for metric in metrics:
            stats = metric_payload.get(metric)
            if stats is None:
                row.append("n/a")
            else:
                row.append(format_mean_std(stats["mean"], stats["std"], decimals=decimals))
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)
