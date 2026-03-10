#!/usr/bin/env python3
"""
snr.py (fixed)

Reads a value.jsonl where each item has:
  - question/prompt
  - steps: list[str]
  - gt_value: list[float|None]  (fraction correct per step)

Builds a token sequence:
  [chat-templated user prompt] + steps joined by "\n\n" with <extra_0> inserted
at evaluation points (evaluate_n_steps stride), and labels placed *exactly at
those <extra_0> positions.

Extracts PRM predictions at the <extra_0> token positions and computes:

- Pooled/global SNR over all valid (gt, pred) pairs:
    Var(gt) / Var(gt - pred)

- Mean-per-example variances (for diagnostics), but SNR is reported pooled
  to avoid inf poisoning (err_var==0 for some examples).

Key fixes vs your previous version:
  1) Align gt/pred by separator indices (not by filtering labels != -100).
  2) Drop invalid gt labels (None or < 0).
  3) Report pooled SNR and finite mean SNR.
  4) Avoid trailing empty steps (drops "" steps).
"""

import argparse
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from rely.train import RegressionPRMModel


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_name_or_path: Optional[str],
    device: str,
    value_model_type: str,
):
    resolved = tokenizer_name_or_path or checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if value_model_type == "classification":
        model = AutoModel.from_pretrained(
            checkpoint_path,
            num_labels=2,
            dtype=torch.bfloat16,
            device_map=device if device != "cuda" else "auto",
            trust_remote_code=True,
        ).to(dtype=torch.bfloat16)
    else:
        model = RegressionPRMModel.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map=device if device != "cuda" else "auto",
            trust_remote_code=True,
        ).to(dtype=torch.bfloat16)

    model.eval()
    return model, tokenizer


def _clean_steps(steps: List[Any]) -> List[str]:
    """Drop trailing empty steps and ensure all are strings."""
    out = [("" if s is None else str(s)) for s in (steps or [])]
    # Drop trailing empties
    while out and out[-1].strip() == "":
        out.pop()
    return out


def tokenize_example(
    example: dict,
    tokenizer,
    step_separator: str,
    max_length: int,
    evaluate_n_steps: int,
) -> Tuple[List[int], List[float], int]:
    """
    Tokenize and return (input_ids, labels, sep_token_id).
    labels are -100 everywhere except at the inserted sep tokens, where label is gt or -1 for missing.
    """
    eval_every = max(1, int(evaluate_n_steps))

    sep_ids = tokenizer.encode(step_separator, add_special_tokens=False)
    sep_token_id = sep_ids[0] if sep_ids else tokenizer.eos_token_id

    question = example.get("prompt") or example.get("question") or ""
    messages = [{"role": "user", "content": question}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        prompt_text = question + "\n\nAssistant:"

    input_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
    labels: List[float] = [-100.0] * len(input_ids)

    steps = _clean_steps(example.get("steps") or example.get("completions") or example.get("prefixes") or [])
    gt_values = example.get("gt_value") or example.get("labels") or []
    if gt_values is None:
        gt_values = []

    # Ensure list length matches steps (pad with None)
    if len(gt_values) < len(steps):
        gt_values = list(gt_values) + [None] * (len(steps) - len(gt_values))
    else:
        gt_values = list(gt_values)[: len(steps)]

    steps_since_sep = 0
    for idx, (step, gt_val) in enumerate(zip(steps, gt_values), start=1):
        step_text = str(step).strip()
        if idx > 1:
            step_text = "\n\n" + step_text

        step_ids = tokenizer(step_text, add_special_tokens=False, truncation=False)["input_ids"]
        input_ids.extend(step_ids)
        labels.extend([-100.0] * len(step_ids))

        steps_since_sep += 1
        is_last = idx == len(steps)
        if steps_since_sep == eval_every or is_last:
            input_ids.append(sep_token_id)
            # Use -1.0 as placeholder for missing gt; we will filter these out later.
            if gt_val is None:
                labels.append(-1.0)
            else:
                try:
                    labels.append(float(gt_val))
                except Exception:
                    labels.append(-1.0)
            steps_since_sep = 0

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return input_ids, labels, sep_token_id


def extract_step_predictions(model, tokenizer, input_ids: List[int], sep_token_id: int, value_model_type: str) -> List[float]:
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device, dtype=torch.long)
    attn = (ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        if value_model_type == "classification":
            # This assumes your trust_remote_code model has model.model + model.score
            base_out = model.model(input_ids=ids, attention_mask=attn, use_cache=False)
            cls_logits = model.score(base_out.last_hidden_state)  # (batch, seq, 2)
            probs = torch.softmax(cls_logits, dim=-1)
            logits = probs[0, :, 1]  # prob of class 1
        else:
            out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits[0]  # (seq_len,)

    sep_mask = (ids[0] == sep_token_id)
    sep_indices = sep_mask.nonzero(as_tuple=True)[0]
    if len(sep_indices) == 0:
        return []
    return logits[sep_indices].float().cpu().tolist()


def aligned_gt_pred(
    input_ids: List[int],
    labels: List[float],
    preds: List[float],
    sep_token_id: int,
) -> Tuple[List[float], List[float]]:
    """
    Align gt and preds by *the separator positions* and filter invalid gt.
    Invalid gt: -100 (non-label), -1 placeholder, None, or negative.
    """
    sep_positions = [i for i, tid in enumerate(input_ids) if tid == sep_token_id]
    L = min(len(sep_positions), len(preds))
    gt, pr = [], []
    for k in range(L):
        pos = sep_positions[k]
        lbl = labels[pos]
        if lbl == -100.0:
            continue
        try:
            lbl_f = float(lbl)
        except Exception:
            continue
        if lbl_f < 0.0:
            continue
        gt.append(lbl_f)
        pr.append(float(preds[k]))
    return gt, pr


def compute_example_stats(gt: List[float], pr: List[float]) -> Dict[str, Any]:
    if not gt or not pr:
        return {}
    gt_arr = np.asarray(gt, dtype=np.float64)
    pr_arr = np.asarray(pr, dtype=np.float64)
    err = gt_arr - pr_arr
    gt_var = float(np.var(gt_arr, ddof=0))
    err_var = float(np.var(err, ddof=0))
    # keep snr finite (still meaningful as diagnostic)
    snr = gt_var / (err_var + 1e-12)
    return {
        "gt_var": gt_var,
        "err_var": err_var,
        "snr": float(snr),
        "gt_mean": float(gt_arr.mean()),
        "pred_mean": float(pr_arr.mean()),
        "n_steps": int(len(gt_arr)),
        "bias": float(err.mean()),  # mean(gt - pred)
        "mse": float(np.mean(err ** 2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="Path to value.jsonl")
    ap.add_argument("--pum_model_ckpt", type=str, required=True, help="Checkpoint for PRM/Value model")
    ap.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Optional tokenizer path (defaults to model ckpt)")
    ap.add_argument("--step_separator", type=str, default="<extra_0>")
    ap.add_argument("--max_length", type=int, default=12000)
    ap.add_argument("--evaluate_n_steps", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--value_model_type", type=str, default="classification", choices=["regression", "classification"])
    args = ap.parse_args()

    dataset = load_jsonl(args.data_path)
    model, tokenizer = load_model_and_tokenizer(
        args.pum_model_ckpt, args.tokenizer_name_or_path, args.device, args.value_model_type
    )

    example_stats = []
    all_gt: List[float] = []
    all_pr: List[float] = []

    for ex in tqdm(dataset, desc="Scoring"):
        input_ids, labels, sep_token_id = tokenize_example(
            ex, tokenizer, args.step_separator, args.max_length, args.evaluate_n_steps
        )
        preds = extract_step_predictions(model, tokenizer, input_ids, sep_token_id, args.value_model_type)
        gt_vals, pr_vals = aligned_gt_pred(input_ids, labels, preds, sep_token_id)

        stats = compute_example_stats(gt_vals, pr_vals)
        if stats:
            example_stats.append(stats)
            all_gt.extend(gt_vals)
            all_pr.extend(pr_vals)

    if not example_stats or not all_gt:
        print("No data to report.")
        return

    # Mean-per-example diagnostic variances (still useful)
    mean_gt_var = float(np.mean([s["gt_var"] for s in example_stats]))
    mean_err_var = float(np.mean([s["err_var"] for s in example_stats]))

    # Pooled/global SNR (stable, does not blow up due to a few degenerate examples)
    gt_arr = np.asarray(all_gt, dtype=np.float64)
    pr_arr = np.asarray(all_pr, dtype=np.float64)
    err_arr = gt_arr - pr_arr

    pooled_gt_var = float(np.var(gt_arr, ddof=0))
    pooled_err_var = float(np.var(err_arr, ddof=0))
    pooled_snr = float(pooled_gt_var / (pooled_err_var + 1e-12))

    # Finite mean SNR (optional)
    snrs = np.asarray([s["snr"] for s in example_stats], dtype=np.float64)
    mean_snr_finite = float(np.mean(snrs[np.isfinite(snrs)])) if np.any(np.isfinite(snrs)) else float("nan")

    mean_gt = float(gt_arr.mean())
    mean_pred = float(pr_arr.mean())
    mean_bias = float(err_arr.mean())
    mse = float(np.mean(err_arr ** 2))

    print(
        json.dumps(
            {
                "mean_cot_gt_variance": mean_gt_var,
                "mean_cot_error_variance": mean_err_var,
                "pooled_gt_variance": pooled_gt_var,
                "pooled_error_variance": pooled_err_var,
                "pooled_snr": pooled_snr,
                "mean_snr_finite": mean_snr_finite,
                "mean_gt": mean_gt,
                "mean_pred": mean_pred,
                "mean_bias_gt_minus_pred": mean_bias,
                "mse": mse,
                "n_examples": len(example_stats),
                "n_steps_total": int(len(all_gt)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
