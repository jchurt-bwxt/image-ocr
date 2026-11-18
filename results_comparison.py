import json
import csv
import re
from pathlib import Path
from collections import defaultdict


# ======================================================
# Text Normalization
# ======================================================

def raw_text(text: str) -> str:
    """Minimal processing for raw CER."""
    if text is None:
        return ""

    return text.replace("\r\n", "\n").strip()


def normalize_text(text: str) -> str:
    """
    OCR-friendly normalization.
    Removes whitespace and converts to upper case.
    """

    if text is None:
        return ""

    text = text.upper()
    text = re.sub(r"\s+", "", text)

    return text


# ======================================================
# Levenshtein Distance
# ======================================================

def levenshtein(a: str, b: str) -> int:

    rows = len(a) + 1
    cols = len(b) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):

            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost   # substitution
            )

    return dp[-1][-1]


def cer_and_distance(truth: str, prediction: str):

    if len(truth) == 0:
        return 0.0, 0

    distance = levenshtein(truth, prediction)

    cer = distance / len(truth)

    return cer, distance


# ======================================================
# Load Labels
# ======================================================

def load_labels(label_file):

    labels = {}

    with open(label_file, encoding="utf-8") as f:

        reader = csv.DictReader(
            f,
            delimiter=";"
        )

        for row in reader:

            image_name = row["File Name"].strip()
            category = row["Category"].strip()
            truth_label = row["Label (Human)"].strip()

            labels[image_name] = {
                "category": category,
                "truth": truth_label,
            }

    return labels


# ======================================================
# Main Evaluation
# ======================================================

def evaluate(
    results_json="ocr_all_results.json",
    labels_file="ocr_labels.txt"
):

    # --------------------------------------------------
    # Load files
    # --------------------------------------------------

    labels = load_labels(labels_file)

    with open(results_json, encoding="utf-8") as f:
        results = json.load(f)

    detail_rows = []

    model_summary_data = defaultdict(list)

    # --------------------------------------------------
    # Process images
    # --------------------------------------------------

    for image in results["images"]:

        image_stem = Path(image["image"]).stem

        if image_stem not in labels:
            print(
                f"Warning: No label found for {image_stem}"
            )
            continue

        label_info = labels[image_stem]

        category = label_info["category"]
        truth_text = label_info["truth"]

        for model in image["models"]:

            ocr_text = model.get("text", "")

            # Raw
            truth_raw = raw_text(truth_text)
            pred_raw = raw_text(ocr_text)

            raw_cer, raw_distance = cer_and_distance(
                truth_raw,
                pred_raw
            )

            # Normalized
            truth_norm = normalize_text(truth_text)
            pred_norm = normalize_text(ocr_text)

            norm_cer, norm_distance = cer_and_distance(
                truth_norm,
                pred_norm
            )

            # Exact match statistics
            exact_match_raw = int(raw_distance == 0)
            exact_match_norm = int(norm_distance == 0)

            row = {
                "image": image_stem,
                "category": category,

                "profile": model.get("profile", ""),
                "label": model.get("label", ""),

                "truth_text": truth_text,
                "ocr_text": ocr_text,

                "truth_length_raw": len(truth_raw),
                "ocr_length_raw": len(pred_raw),

                "raw_edit_distance": raw_distance,
                "raw_cer": round(raw_cer, 6),

                "normalized_edit_distance": norm_distance,
                "normalized_cer": round(norm_cer, 6),

                "exact_match_raw": exact_match_raw,
                "exact_match_normalized": exact_match_norm,

                "analysis_seconds":
                    model.get("processing_seconds", 0),

                "total_seconds":
                    model.get("total_seconds", 0),

                "status":
                    model.get("status", "")
            }

            detail_rows.append(row)

            model_summary_data[
                model.get("profile", "unknown")
            ].append(row)

    if not detail_rows:
        raise RuntimeError(
            "No matching OCR results and labels found."
        )

    # --------------------------------------------------
    # Write Detailed CSV
    # --------------------------------------------------

    detail_file = "ocr_evaluation_detail.csv"

    with open(
        detail_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=detail_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    # --------------------------------------------------
    # Build Summary CSV
    # --------------------------------------------------

    summary_rows = []

    for profile, rows in model_summary_data.items():

        count = len(rows)

        avg_raw_cer = (
            sum(r["raw_cer"] for r in rows)
            / count
        )

        avg_norm_cer = (
            sum(r["normalized_cer"] for r in rows)
            / count
        )

        avg_analysis_time = (
            sum(r["analysis_seconds"] for r in rows)
            / count
        )

        avg_total_time = (
            sum(r["total_seconds"] for r in rows)
            / count
        )

        raw_accuracy = (
            100 *
            sum(r["exact_match_raw"] for r in rows)
            / count
        )

        normalized_accuracy = (
            100 *
            sum(r["exact_match_normalized"] for r in rows)
            / count
        )

        summary_rows.append({
            "profile": profile,
            "samples": count,

            "average_raw_cer":
                round(avg_raw_cer, 6),

            "average_normalized_cer":
                round(avg_norm_cer, 6),

            "raw_exact_match_percent":
                round(raw_accuracy, 2),

            "normalized_exact_match_percent":
                round(normalized_accuracy, 2),

            "average_analysis_seconds":
                round(avg_analysis_time, 3),

            "average_total_seconds":
                round(avg_total_time, 3),
        })

    summary_file = "ocr_evaluation_summary.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------
    # Print Summary
    # --------------------------------------------------

    print("\n=== OCR MODEL SUMMARY ===\n")

    for row in summary_rows:

        print(
            f"{row['profile']:12s}"
            f"  Samples={row['samples']:4d}"
            f"  Raw CER={row['average_raw_cer']:.4f}"
            f"  Norm CER={row['average_normalized_cer']:.4f}"
            f"  Raw Acc={row['raw_exact_match_percent']:.1f}%"
            f"  Norm Acc={row['normalized_exact_match_percent']:.1f}%"
            f"  Time={row['average_analysis_seconds']:.2f}s"
        )

    print("\nCreated:")
    print(f"  {detail_file}")
    print(f"  {summary_file}")


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":

    evaluate(
        results_json="ocr_all_results.json",
        labels_file="ocr_labels.txt"
    )