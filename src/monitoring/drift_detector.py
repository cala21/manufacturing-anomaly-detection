"""Data and model drift detection using Evidently."""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from pathlib import Path
import json

def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str = "monitoring/reports",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path = f"{output_dir}/drift_report.html"
    report.save_html(report_path)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    summary = {
        "drift_detected": drift_detected,
        "drifted_columns": [
            col
            for col, stats in result["metrics"][0]["result"]["drift_by_columns"].items()
            if stats["drift_detected"]
        ],
        "report_path": report_path,
    }

    with open(f"{output_dir}/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
