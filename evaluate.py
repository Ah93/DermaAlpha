"""
DermaAlpha Evaluation Script
Measures Top-1 Accuracy, Top-3 Accuracy, and Severity MAE
against a ground-truth CSV derived from HAM10000 / ISIC 2019.

Ground-truth CSV format (ground_truth.csv):
  image_path,true_label,true_severity
  /path/to/img1.jpg,Melanoma,8.0
  /path/to/img2.jpg,Basal Cell Carcinoma,6.5
  ...

true_severity scale: 0–10 (0 = clear, 10 = severe)
  HAM10000 proxy: mel→8, bcc→7, akiec→6, bkl→3, df→2, vasc→2, nv→2

Usage:
  python evaluate.py --csv ground_truth.csv --url http://localhost:5000
"""

import argparse
import csv
import json
import sys
import base64
import statistics
import requests
from pathlib import Path

# ── Label normalisation map ──────────────────────────────────────────────────
# Maps ground-truth label variants → canonical form used by DermaAlpha
LABEL_ALIASES = {
    "melanoma": "Melanoma",
    "mel": "Melanoma",
    "basal cell carcinoma": "Basal Cell Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "actinic keratosis": "Actinic Keratosis",
    "akiec": "Actinic Keratosis",
    "benign keratosis": "Benign Keratosis",
    "bkl": "Benign Keratosis",
    "seborrheic keratosis": "Benign Keratosis",
    "dermatofibroma": "Dermatofibroma",
    "df": "Dermatofibroma",
    "vascular lesion": "Vascular Lesion",
    "vasc": "Vascular Lesion",
    "melanocytic nevi": "Melanocytic Nevi",
    "nv": "Melanocytic Nevi",
}

def normalise(label: str) -> str:
    return LABEL_ALIASES.get(label.strip().lower(), label.strip())


def load_ground_truth(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_path": row["image_path"].strip(),
                "true_label": normalise(row["true_label"]),
                "true_severity": float(row["true_severity"]),
            })
    return rows


def analyse_image(image_path: str, base_url: str) -> dict | None:
    """POST image to /api/analyze, return parsed JSON or None on error."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    payload = {
        "image": f"data:{mime};base64,{b64}",
        "patient_id": "eval_patient",
        "is_first_encounter": True,
    }

    try:
        resp = requests.post(f"{base_url}/api/analyze", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] {image_path}: {e}", file=sys.stderr)
        return None


def extract_top3(result: dict) -> list[str]:
    """Return up to 3 normalised diagnosis labels from the API result."""
    # DermaAlpha returns 'diagnosis' (primary) and optionally 'differentials' list
    labels = []
    primary = result.get("diagnosis", "")
    if primary:
        labels.append(normalise(primary))

    for d in result.get("differentials", []):
        name = d if isinstance(d, str) else d.get("name", "")
        if name:
            labels.append(normalise(name))

    return labels[:3]


def run_evaluation(csv_path: str, base_url: str, limit: int | None):
    rows = load_ground_truth(csv_path)
    if limit:
        rows = rows[:limit]

    top1_hits, top3_hits = 0, 0
    severity_errors = []
    skipped = 0
    total = len(rows)

    print(f"Evaluating {total} samples against {base_url} …\n")

    for i, row in enumerate(rows, 1):
        print(f"[{i}/{total}] {Path(row['image_path']).name}", end=" … ")

        result = analyse_image(row["image_path"], base_url)
        if result is None:
            skipped += 1
            print("SKIP")
            continue

        top3 = extract_top3(result)
        true_label = row["true_label"]
        hit1 = len(top3) > 0 and top3[0] == true_label
        hit3 = true_label in top3

        pred_severity = result.get("severity_score", None)
        if pred_severity is not None:
            severity_errors.append(abs(float(pred_severity) - row["true_severity"]))

        top1_hits += int(hit1)
        top3_hits += int(hit3)
        print(f"{'✓' if hit1 else '✗'} pred={top3[0] if top3 else '?'} true={true_label}")

    evaluated = total - skipped
    if evaluated == 0:
        print("\nNo samples evaluated — check CSV paths and server.")
        return

    top1_acc = top1_hits / evaluated * 100
    top3_acc = top3_hits / evaluated * 100
    mae = statistics.mean(severity_errors) if severity_errors else float("nan")

    print("\n" + "=" * 50)
    print("  DermaAlpha Evaluation Results")
    print("=" * 50)
    print(f"  Samples evaluated : {evaluated}  (skipped: {skipped})")
    print(f"  Top-1 Accuracy    : {top1_acc:.1f}%  ({top1_hits}/{evaluated})")
    print(f"  Top-3 Accuracy    : {top3_acc:.1f}%  ({top3_hits}/{evaluated})")
    print(f"  Severity MAE      : {mae:.2f}  (scale 0–10)")
    print("=" * 50)

    # Save results to JSON for paper table
    output = {
        "samples_evaluated": evaluated,
        "samples_skipped": skipped,
        "top1_accuracy": round(top1_acc, 2),
        "top3_accuracy": round(top3_acc, 2),
        "severity_mae": round(mae, 3) if severity_errors else None,
    }
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermaAlpha accuracy evaluation")
    parser.add_argument("--csv", required=True, help="Path to ground_truth.csv")
    parser.add_argument("--url", default="http://localhost:5000", help="DermaAlpha base URL")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples")
    args = parser.parse_args()

    run_evaluation(args.csv, args.url, args.limit)
