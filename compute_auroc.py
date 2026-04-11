#!/usr/bin/env python3
"""
Compute AUROC between a suspected-seen dataset and a known-unseen control.

Usage:
    python compute_auroc.py \
        --seen   results/contamination/Qwen3.5-2B/mmlu_all.csv \
        --unseen results/contamination/Qwen3.5-2B/controlled_dataset_english.csv

Both CSVs must have already been scored by run_contamination.py
(i.e. they must contain the minkpp_* and loss columns).

AUROC interpretation:
    0.5  = no signal, model treats both datasets identically
    0.6  = weak signal
    0.7  = moderate signal, likely contamination
    0.8+ = strong signal, high confidence of contamination
    1.0  = perfect separation
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

METHODS = (
    ["loss", "zlib"] +
    [f"mink_{r:.1f}" for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]] +
    [f"minkpp_{r:.1f}" for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen", required=True,
                        help="CSV scored by run_contamination.py — suspected seen dataset (label=1)")
    parser.add_argument("--unseen", required=True,
                        help="CSV scored by run_contamination.py — known unseen control (label=0)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save results CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    seen_df   = pd.read_csv(args.seen).assign(label=1)
    unseen_df = pd.read_csv(args.unseen).assign(label=0)

    # Check scoring columns exist
    available = [m for m in METHODS if m in seen_df.columns and m in unseen_df.columns]
    if not available:
        raise ValueError(
            "No scoring columns found. Make sure both CSVs were produced by run_contamination.py."
        )

    combined = pd.concat([seen_df, unseen_df], ignore_index=True)
    labels   = combined["label"].values

    seen_name   = args.seen.split("/")[-1].replace(".csv", "")
    unseen_name = args.unseen.split("/")[-1].replace(".csv", "")
    model_name  = args.seen.split("/")[-2] if "/" in args.seen else "unknown_model"

    print(f"")
    print(f"================================================================")
    print(f"  Model:        {model_name}")
    print(f"  Seen:         {seen_name}  ({seen_df.shape[0]} samples, label=1)")
    print(f"  Unseen:       {unseen_name}  ({unseen_df.shape[0]} samples, label=0)")
    print(f"")
    print(f"  METHOD          AUROC    INTERPRETATION")
    print(f"  ------------------------------------------------------")

    results = []
    for method in available:
        scores = combined[method].values
        if np.isnan(scores).all():
            continue
        # Drop NaN pairs
        mask   = ~np.isnan(scores)
        auroc  = roc_auc_score(labels[mask], scores[mask])

        if auroc < 0.55:
            interp = "no signal"
        elif auroc < 0.65:
            interp = "weak signal"
        elif auroc < 0.75:
            interp = "moderate — possible contamination"
        elif auroc < 0.85:
            interp = "strong — likely contamination"
        else:
            interp = "very strong — high confidence contamination"

        marker = "  <- MAIN RESULT" if method == "minkpp_0.2" else ""
        print(f"  {method:<16}  {auroc:.4f}   {interp}{marker}")
        results.append({"method": method, "auroc": round(auroc, 4), "interpretation": interp})

    print(f"  ------------------------------------------------------")
    print(f"  Seen dataset:   {seen_name}")
    print(f"  Control:        {unseen_name}")
    print(f"================================================================")
    print(f"")
    print(f"  HOW TO READ: AUROC > 0.5 means the model scores '{seen_name}'")
    print(f"  higher than '{unseen_name}', i.e. the model is more familiar")
    print(f"  with '{seen_name}' — evidence that it was seen during training.")
    print(f"================================================================")

    if args.output:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
