from adaptive_evidence_vqa.eval.aggregate import collect_method_metrics, format_markdown_table


def test_collect_method_metrics_and_markdown_table(tmp_path) -> None:
    run_a = tmp_path / "run_a" / "outputs"
    run_b = tmp_path / "run_b" / "outputs"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    (run_a / "fixed_budget_frozen.summary.json").write_text(
        """
        {
          "metrics": {
            "accuracy": 0.36,
            "selected_evidence_cost": 7.5,
            "selected_evidence_count": 6.0,
            "selected_temporal_iou": 0.27
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (run_b / "fixed_budget_frozen.summary.json").write_text(
        """
        {
          "metrics": {
            "accuracy": 0.40,
            "selected_evidence_cost": 7.0,
            "selected_evidence_count": 5.5,
            "selected_temporal_iou": 0.30
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    aggregated = collect_method_metrics(
        [tmp_path / "run_a", tmp_path / "run_b"],
        summary_files=("fixed_budget_frozen.summary.json",),
        metrics=("accuracy", "selected_evidence_cost"),
    )

    assert aggregated["fixed_budget_frozen.summary.json"]["num_runs"] == 2
    assert aggregated["fixed_budget_frozen.summary.json"]["metrics"]["accuracy"]["mean"] == 0.38
    assert aggregated["fixed_budget_frozen.summary.json"]["metrics"]["selected_evidence_cost"]["mean"] == 7.25

    markdown = format_markdown_table(
        aggregated,
        metrics=("accuracy", "selected_evidence_cost"),
        decimals=3,
    )
    assert "0.380 +/- 0.028" in markdown
    assert "7.250 +/- 0.354" in markdown
