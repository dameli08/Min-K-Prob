#!/usr/bin/env python3
"""
Per-sample contamination scoring using the original Min-K%++ method
(Zhang et al., ICLR 2025) applied to any CSV dataset.

No ground-truth labels required.

Per-row columns added to the CSV (identical to original run.py):
    loss              - raw log-likelihood (higher = more familiar)
    zlib              - loss / zlib-compressed length
    mink_0.1 ... 1.0  - Min-K% at each ratio (10 columns)
    minkpp_0.1 ... 1.0 - Min-K%++ at each ratio (10 columns)  <- main method

Summary CSV (summary.csv in the model folder):
    One row per dataset, reporting the MEAN of each method across all samples.
    Higher mean = model is more familiar with that dataset = stronger contamination signal.
    Compare these numbers across models and datasets -- that is the intended use.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import zlib as zlib_module

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to input CSV file")
    parser.add_argument("--text_column", type=str, default="question",
                        help="Column in the CSV with text to score (default: question)")
    parser.add_argument("--output_dir", type=str, default="results/contamination")
    parser.add_argument("--half", action="store_true",
                        help="Load in bfloat16 (recommended for >=7B models)")
    parser.add_argument("--int8", action="store_true",
                        help="Load in 8-bit (requires bitsandbytes)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Truncate texts to this many tokens (default: 512)")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Random sample size (default: 100). Use -1 for all rows.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output if it already exists")
    return parser.parse_args()


def load_model(name, half=False, int8=False):
    kwargs = {}
    if int8:
        kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif half:
        kwargs = dict(torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        name, return_dict=True, device_map="auto",
        trust_remote_code=True, **kwargs
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return model, tokenizer


def score_text(text, model, tokenizer, max_length):
    """
    Identical scoring logic to the original run.py.
    Returns a dict of all method scores for this text.
    """
    nan_result = {"loss": float("nan"), "zlib": float("nan")}
    for r in RATIOS:
        nan_result[f"mink_{r:.1f}"]   = float("nan")
        nan_result[f"minkpp_{r:.1f}"] = float("nan")

    ids = tokenizer.encode(text, truncation=True, max_length=max_length)
    if len(ids) < 2:
        return nan_result

    input_ids = torch.tensor(ids).unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    ll = -loss.item()   # log-likelihood -- higher = more familiar

    result = {}
    result["loss"]  = ll
    result["zlib"]  = ll / len(zlib_module.compress(text.encode("utf-8")))

    # Per-token scores
    shifted_ids = input_ids[0][1:].unsqueeze(-1)        # (T-1, 1)
    probs       = F.softmax(logits[0, :-1], dim=-1)     # (T-1, V)
    log_probs   = F.log_softmax(logits[0, :-1], dim=-1) # (T-1, V)

    token_log_probs = log_probs.gather(dim=-1, index=shifted_ids).squeeze(-1)  # (T-1,)

    # Min-K%: average of the bottom-k% log-probs
    sorted_log_probs = np.sort(token_log_probs.cpu().float().numpy())  # ascending
    for r in RATIOS:
        k = max(1, int(len(sorted_log_probs) * r))
        result[f"mink_{r:.1f}"] = float(np.mean(sorted_log_probs[:k]))

    # Min-K%++: same but z-score normalised per position
    mu    = (probs * log_probs).sum(-1)                                     # (T-1,)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)    # (T-1,)
    sigma = sigma.clamp(min=1e-8)
    mink_plus = (token_log_probs - mu) / sigma.sqrt()                       # (T-1,)

    sorted_minkpp = np.sort(mink_plus.cpu().float().numpy())  # ascending
    for r in RATIOS:
        k = max(1, int(len(sorted_minkpp) * r))
        result[f"minkpp_{r:.1f}"] = float(np.mean(sorted_minkpp[:k]))

    return result


def main():
    args = parse_args()

    model_name   = os.path.basename(args.model.rstrip("/"))
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    out_dir      = os.path.join(args.output_dir, model_name)
    out_path     = os.path.join(out_dir, f"{dataset_name}.csv")

    if os.path.exists(out_path) and not args.force:
        print(f"[SKIP] Already exists: {out_path}  (use --force to overwrite)")
        return

    print(f"[model]      {args.model}")
    print(f"[dataset]    {args.dataset_path}")
    print(f"[output]     {out_path}")

    model, tokenizer = load_model(args.model, half=args.half, int8=args.int8)

    df = pd.read_csv(args.dataset_path)
    total_rows = len(df)
    if args.sample_size != -1 and args.sample_size < total_rows:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"[sample]     {args.sample_size} / {total_rows} rows (seed=42)")
    else:
        print(f"[sample]     all {total_rows} rows")

    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    # Score every row
    all_scores = []
    for text in tqdm(df[args.text_column].astype(str).tolist(),
                     desc=f"{dataset_name} | {model_name}"):
        all_scores.append(score_text(text, model, tokenizer, args.max_length))

    scores_df = pd.DataFrame(all_scores)
    result_df = pd.concat([df, scores_df], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    # Summary: mean of every method across all samples
    method_cols = ["loss", "zlib"] + \
                  [f"mink_{r:.1f}" for r in RATIOS] + \
                  [f"minkpp_{r:.1f}" for r in RATIOS]
    means = {col: round(float(scores_df[col].mean()), 6) for col in method_cols}

    summary_row = pd.DataFrame([{"dataset": dataset_name, **means}])
    summary_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[existing["dataset"] != dataset_name]
        summary_row = pd.concat([existing, summary_row], ignore_index=True)
    summary_row.to_csv(summary_path, index=False)

    print(f"")
    print(f"========================================================")
    print(f"  Model:    {model_name}")
    print(f"  Dataset:  {dataset_name}  ({len(df)} samples)")
    print(f"")
    print(f"  METHOD          MEAN SCORE   (higher = more familiar to model)")
    print(f"  -------------------------------------------------------")
    print(f"  loss            {means['loss']:>10.4f}")
    print(f"  zlib            {means['zlib']:>10.4f}")
    print(f"  mink_0.2        {means['mink_0.2']:>10.4f}")
    print(f"  minkpp_0.2      {means['minkpp_0.2']:>10.4f}  <- main result (best ratio)")
    print(f"  minkpp_0.1      {means['minkpp_0.1']:>10.4f}")
    print(f"  minkpp_0.3      {means['minkpp_0.3']:>10.4f}")
    print(f"  -------------------------------------------------------")
    print(f"  Per-row CSV:    {out_path}")
    print(f"  Summary CSV:    {summary_path}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
