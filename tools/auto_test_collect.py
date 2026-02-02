import argparse
import csv
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESHOUT_DIR = PROJECT_ROOT / "meshout"

RESULTS_RE = re.compile(r"results_(\d+)\.json$")
UV_DIR_RE = re.compile(r"scan(\d+)_uv_debug$")
PSNR_RE = re.compile(
    r"Average PSNR \(([^)]+)\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf)\s*over\s*(\d+)\s*views\.",
    re.IGNORECASE,
)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_valid_number(value):
    if not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


def _parse_scan_id_from_results(path: Path):
    match = RESULTS_RE.match(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_scan_id_from_uv_dir(path: Path):
    match = UV_DIR_RE.match(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_psnr_log(psnr_path: Path):
    if not psnr_path.is_file():
        return None
    try:
        lines = psnr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        match = PSNR_RE.search(line)
        if match:
            split = match.group(1)
            psnr_value = _to_float(match.group(2))
            try:
                views = int(match.group(3))
            except ValueError:
                views = None
            return {
                "uv_psnr": psnr_value,
                "uv_psnr_split": split,
                "uv_psnr_views": views,
            }
    return None


def collect_results(meshout_dir: Path):
    records = {}

    for result_path in sorted(meshout_dir.glob("results_*.json")):
        scan_id = _parse_scan_id_from_results(result_path)
        if scan_id is None:
            continue
        try:
            with result_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        record = records.setdefault(
            scan_id,
            {
                "scan": f"scan{scan_id}",
                "scan_id": scan_id,
                "mean_d2s": None,
                "mean_s2d": None,
                "overall": None,
                "uv_psnr": None,
                "uv_psnr_split": None,
                "uv_psnr_views": None,
            },
        )
        record["mean_d2s"] = _to_float(data.get("mean_d2s"))
        record["mean_s2d"] = _to_float(data.get("mean_s2d"))
        record["overall"] = _to_float(data.get("overall"))

    for uv_dir in sorted(meshout_dir.glob("scan*_uv_debug")):
        if not uv_dir.is_dir():
            continue
        scan_id = _parse_scan_id_from_uv_dir(uv_dir)
        if scan_id is None:
            continue
        psnr_info = _parse_psnr_log(uv_dir / "psnr.txt")
        if not psnr_info:
            continue
        record = records.setdefault(
            scan_id,
            {
                "scan": f"scan{scan_id}",
                "scan_id": scan_id,
                "mean_d2s": None,
                "mean_s2d": None,
                "overall": None,
                "uv_psnr": None,
                "uv_psnr_split": None,
                "uv_psnr_views": None,
            },
        )
        record.update(psnr_info)

    sorted_records = [records[k] for k in sorted(records.keys())]
    return sorted_records


def compute_averages(records):
    metrics = ["mean_d2s", "mean_s2d", "overall", "uv_psnr"]
    averages = {}
    for metric in metrics:
        values = [rec.get(metric) for rec in records if _is_valid_number(rec.get(metric))]
        if values:
            averages[metric] = sum(values) / len(values)
        else:
            averages[metric] = None
    return averages


def _format_metric(value):
    if not _is_valid_number(value):
        return ""
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Collect DTU mesh CD results and UV PSNR logs into a single CSV summary."
    )
    parser.add_argument(
        "--meshout_dir",
        type=str,
        default=str(DEFAULT_MESHOUT_DIR),
        help="Path to meshout directory (default: neuralangelo/meshout).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output summary CSV path (default: <meshout_dir>/auto_test_summary.csv).",
    )
    args = parser.parse_args()

    meshout_dir = Path(args.meshout_dir).expanduser().resolve()
    if not meshout_dir.exists():
        raise FileNotFoundError(f"meshout_dir not found: {meshout_dir}")

    records = collect_results(meshout_dir)
    averages = compute_averages(records)

    output_path = Path(args.output).expanduser() if args.output else meshout_dir / "auto_test_summary.csv"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scan",
        "scan_id",
        "mean_d2s",
        "mean_s2d",
        "overall",
        "uv_psnr",
        "uv_psnr_split",
        "uv_psnr_views",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "scan": record.get("scan", ""),
                    "scan_id": record.get("scan_id", ""),
                    "mean_d2s": _format_metric(record.get("mean_d2s")),
                    "mean_s2d": _format_metric(record.get("mean_s2d")),
                    "overall": _format_metric(record.get("overall")),
                    "uv_psnr": _format_metric(record.get("uv_psnr")),
                    "uv_psnr_split": record.get("uv_psnr_split", "") or "",
                    "uv_psnr_views": record.get("uv_psnr_views", "") or "",
                }
            )
        writer.writerow(
            {
                "scan": "AVERAGE",
                "scan_id": "",
                "mean_d2s": _format_metric(averages.get("mean_d2s")),
                "mean_s2d": _format_metric(averages.get("mean_s2d")),
                "overall": _format_metric(averages.get("overall")),
                "uv_psnr": _format_metric(averages.get("uv_psnr")),
                "uv_psnr_split": "",
                "uv_psnr_views": "",
            }
        )

    print(f"Found {len(records)} scans. Summary written to: {output_path}")


if __name__ == "__main__":
    main()
