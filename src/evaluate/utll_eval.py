import datetime
import json
import os
import pandas as pd

def save_report(report: dict, out_dir: str ="benchmark_report", out_prefix: str="report") -> None:
    """
    Save the evaluation report to a JSON file and a CSV file.
    Args:
        report: A dictionary containing the evaluation report data.
        out_dir: string, directory where the report will be saved.
        out_prefix: string, prefix for the report filename.

    Returns: None

    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON with proper indentation
    filename = f"{out_prefix}_{ts}.json"
    with open(filename, "w", encoding='utf-8') as jf:
        json.dump(report, jf, indent=4, ensure_ascii=False, sort_keys=True)

    # Create reports directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Also save in reports directory
    json_path = f"reports/benchmark/{out_prefix}_{ts}.json"
    with open(json_path, "w", encoding='utf-8') as jf:
        json.dump(report, jf, indent=4, ensure_ascii=False, sort_keys=True)

    # Save CSV version
    csv_path = f"reports/benchmark/{out_prefix}_{ts}.csv"
    pd.DataFrame([report]).to_csv(csv_path, index=False)

    print(f"Report saved to:")
    print(f"  JSON: {filename}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

