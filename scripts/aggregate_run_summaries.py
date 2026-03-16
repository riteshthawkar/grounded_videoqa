import argparse
import json
from pathlib import Path

from adaptive_evidence_vqa.eval.aggregate import (
    DEFAULT_METRICS,
    DEFAULT_SUMMARY_FILES,
    collect_method_metrics,
    format_markdown_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment summary JSON files across multiple run directories."
    )
    parser.add_argument(
        "--run-roots",
        nargs="+",
        required=True,
        help="One or more experiment run directories containing an outputs/ subdirectory.",
    )
    parser.add_argument(
        "--summary-files",
        nargs="+",
        default=list(DEFAULT_SUMMARY_FILES),
        help="Summary filenames to aggregate from each run's outputs/ directory.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric names to aggregate from each summary file.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the aggregated JSON payload.",
    )
    parser.add_argument(
        "--output-markdown",
        help="Optional path to write the Markdown table.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimals to show in the Markdown table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregated = collect_method_metrics(
        run_roots=args.run_roots,
        summary_files=tuple(args.summary_files),
        metrics=tuple(args.metrics),
    )

    markdown = format_markdown_table(
        aggregated,
        metrics=tuple(args.metrics),
        decimals=args.decimals,
    )

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    if args.output_markdown:
        output_markdown_path = Path(args.output_markdown)
        output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        output_markdown_path.write_text(markdown + "\n", encoding="utf-8")

    print(json.dumps(aggregated, indent=2))
    print()
    print(markdown)


if __name__ == "__main__":
    main()
