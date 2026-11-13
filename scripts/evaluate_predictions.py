
"""Evaluate four-class ECG predictions.

Input CSV must have:
    true_label,pred_label

Run:
    python scripts/evaluate_predictions.py \
        --input results/predictions.csv \
        --output results/evaluation.json
"""

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

LABELS = ["Normal", "AFib", "Other", "Noisy"]

ALIASES = {
    "n": "Normal",
    "normal": "Normal",
    "a": "AFib",
    "af": "AFib",
    "afib": "AFib",
    "o": "Other",
    "other": "Other",
    "~": "Noisy",
    "noisy": "Noisy",
}


def normalize_label(value):
    key = str(value).strip().lower()

    if key not in ALIASES:
        raise ValueError(f"Unknown ECG class: {value!r}")

    return ALIASES[key]


def evaluate_rows(rows):
    if not rows:
        raise ValueError("No prediction rows supplied.")

    truth = [
        normalize_label(row["true_label"])
        for row in rows
    ]

    predictions = [
        normalize_label(row["pred_label"])
        for row in rows
    ]

    per_class_f1 = f1_score(
        truth,
        predictions,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    per_class_recall = recall_score(
        truth,
        predictions,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        truth,
        predictions,
        labels=LABELS,
    )

    return {
        "samples": len(rows),
        "accuracy": float(
            accuracy_score(truth, predictions)
        ),
        "macro_f1": float(
            per_class_f1.mean()
        ),
        "paper_style_f1": float(
            per_class_f1[:3].mean()
        ),
        "afib_recall": float(
            per_class_recall[1]
        ),
        "afib_f1": float(
            per_class_f1[1]
        ),
        "class_order": LABELS,
        "per_class_f1": {
            label: float(score)
            for label, score in zip(
                LABELS, per_class_f1
            )
        },
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_file(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"true_label", "pred_label"}
        columns = set(reader.fieldnames or [])

        if not required.issubset(columns):
            raise ValueError(
                "CSV must contain true_label and pred_label"
            )

        return evaluate_rows(list(reader))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ECG model predictions"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing held-out predictions",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output",
    )

    args = parser.parse_args()

    results = evaluate_file(args.input)

    print(json.dumps(results, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
