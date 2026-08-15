"""
Field-vs-controlled evaluation — Milestone 1 §"controlled vs field gap".

Evaluates the SAME trained model separately on a controlled/lab-image test
set and an independent field-image test set, and reports the accuracy gap
directly, rather than a single blended number that would hide it. This is
the script that answers Milestone-1 question #10.
"""
import argparse
import json

from src.evaluation.metrics import compute_metrics


def compare_controlled_vs_field(controlled_results: dict, field_results: dict) -> dict:
    gap = {
        "controlled_accuracy": controlled_results["accuracy"],
        "field_accuracy": field_results["accuracy"],
        "accuracy_gap": controlled_results["accuracy"] - field_results["accuracy"],
        "controlled_macro_f1": controlled_results["macro_f1"],
        "field_macro_f1": field_results["macro_f1"],
        "macro_f1_gap": controlled_results["macro_f1"] - field_results["macro_f1"],
    }
    return gap


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(
        "Expects two JSON files, each produced by dumping compute_metrics() output "
        "for a model's predictions on a controlled-image set and a field-image set."
    ))
    ap.add_argument("--controlled_metrics_json", required=True)
    ap.add_argument("--field_metrics_json", required=True)
    ap.add_argument("--out_json", default="reports/controlled_vs_field_gap.json")
    args = ap.parse_args()

    with open(args.controlled_metrics_json) as f:
        controlled = json.load(f)
    with open(args.field_metrics_json) as f:
        field = json.load(f)

    gap = compare_controlled_vs_field(controlled, field)
    with open(args.out_json, "w") as f:
        json.dump(gap, f, indent=2)
    print(json.dumps(gap, indent=2))
