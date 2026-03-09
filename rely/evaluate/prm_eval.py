import argparse
import sys

# Workaround: The 'datasets' library tries to import 'jax' if available.
# In some environments (like the cluster), 'jax' metadata might be broken,
# causing a crash in 'importlib.metadata.version'. 
# Since we don't need JAX here, we force it to appear missing.
sys.modules["jax"] = None
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from rely.train import RegressionPRMModel


# -----------------------------
# Utils
# -----------------------------
def calculate_expected_calibration_error(predictions: np.ndarray, labels_binary: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE for binary classification.
    predictions: probability in [0,1]
    labels_binary: {0,1}
    """
    preds = np.clip(predictions.astype(np.float64), 0.0, 1.0)
    y = labels_binary.astype(np.int64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(preds)
    if n == 0:
        return float("nan")

    for b0, b1 in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (preds > b0) & (preds <= b1)
        m = int(in_bin.sum())
        if m == 0:
            continue
        prop = m / n
        acc = float(y[in_bin].mean())
        conf = float(preds[in_bin].mean())
        ece += abs(conf - acc) * prop

    return float(ece)


def evaluate_first_positive_detection(predictions: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> int:
    """
    Determine whether the first positive (>thr) label step is identified by the first positive prediction.
    Returns:
      1 correct
      0 incorrect
     -1 if no positive labels and no positive predictions
    """
    positive_labels = labels > thr
    if not positive_labels.any():
        return -1 if not (predictions > thr).any() else 0

    first_positive_idx = int(np.argmax(positive_labels))
    positive_preds = predictions > thr
    if not positive_preds.any():
        return 0

    first_positive_pred_idx = int(np.argmax(positive_preds))
    return 1 if first_positive_idx == first_positive_pred_idx else 0


def print_dataset_quick_stats(dataset, num_preview: int = 3):
    print("\n" + "=" * 60)
    print("DATASET DEBUG")
    print("=" * 60)
    print(f"Dataset size: {len(dataset)}")
    try:
        print(f"Columns: {dataset.column_names}")
    except Exception:
        pass

    for i in range(min(num_preview, len(dataset))):
        ex = dataset[i]
        keys = list(ex.keys())
        print(f"\nExample {i} keys: {keys}")
        for k in ["prompt", "question", "completions", "labels"]:
            if k in ex:
                try:
                    v = ex[k]
                    if isinstance(v, (list, tuple)):
                        print(f"  {k}: list len={len(v)}")
                    else:
                        print(f"  {k}: type={type(v).__name__}")
                except Exception:
                    print(f"  {k}: present but could not inspect")
    print("=" * 60 + "\n")


# -----------------------------
# Model loading
# -----------------------------
def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    If a HF cache folder is passed (models--.../snapshots/<hash>), pick latest snapshot.
    Otherwise return as is.
    """
    cp_path = Path(checkpoint_path)
    if cp_path.exists() and cp_path.is_dir():
        snapshots_dir = cp_path / "snapshots"
        if snapshots_dir.is_dir():
            snapshot_dirs = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshot_dirs:
                return str(snapshot_dirs[0])
        return str(cp_path)
    return checkpoint_path


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_name_or_path: Optional[str],
    prm_type: str,
    device: str,
):
    resolved_model_path = resolve_checkpoint_path(checkpoint_path)
    resolved_tok_path = resolve_checkpoint_path(tokenizer_name_or_path) if tokenizer_name_or_path else resolved_model_path

    print(f"Loading tokenizer from: {resolved_tok_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_tok_path, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f"Fast tokenizer failed ({e}); retrying use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(resolved_tok_path, trust_remote_code=True, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from: {resolved_model_path} (prm_type={prm_type})")
    if prm_type == "regression":
        model = RegressionPRMModel.from_pretrained(
            resolved_model_path,
            dtype=torch.bfloat16,
            device_map=device if device != "cuda" else "auto",
            trust_remote_code=True,
        ).to(dtype=torch.bfloat16)
    else:
        # classification-style PRM (custom trust_remote_code model)
        model = AutoModel.from_pretrained(
            resolved_model_path,
            dtype=torch.bfloat16,
            device_map=device if device != "cuda" else "auto",
            trust_remote_code=True,
        ).to(dtype=torch.bfloat16)

    model.eval()

    print("\n" + "=" * 60)
    print("TOKENIZER / MODEL DEBUG")
    print("=" * 60)
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Resolved tok path:   {resolved_tok_path}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Model class:     {model.__class__.__name__}")
    print(f"pad_token: {repr(tokenizer.pad_token)} id={tokenizer.pad_token_id}")
    print(f"eos_token: {repr(tokenizer.eos_token)} id={tokenizer.eos_token_id}")
    print(f"bos_token: {repr(tokenizer.bos_token)} id={tokenizer.bos_token_id}")
    print(f"padding_side: {tokenizer.padding_side}")
    try:
        print(f"all_special_tokens: {tokenizer.all_special_tokens}")
    except Exception:
        pass
    print("=" * 60 + "\n")

    return model, tokenizer


# -----------------------------
# Tokenization (robust mapping)
# -----------------------------
def _clean_steps(steps: List[Any]) -> List[str]:
    out = [("" if s is None else str(s)) for s in (steps or [])]
    while out and out[-1].strip() == "":
        out.pop()
    return out


def tokenize_example(
    example: Dict[str, Any],
    tokenizer,
    step_separator: str,
    max_length: int,
    evaluate_n_steps: int,
    prompt_field: str,
    steps_field: str,
    labels_field: str,
    value_baseline: str = "none",
    require_single_token_separator: bool = True,
) -> Dict[str, Any]:
    """
    Build:
      [chat prompt] + step1 + step2 + ... and insert sep token at evaluation points.

    Returns dict:
      input_ids: List[int]
      labels:    List[float]  (-100 except at sep positions)
      eval_steps: List[int]   (1-based original step index where sep was inserted)
      sep_token_id: int
      was_truncated: bool
    """
    eval_every = max(1, int(evaluate_n_steps))

    sep_ids = tokenizer.encode(step_separator, add_special_tokens=False)
    if len(sep_ids) != 1:
        msg = f"step_separator {repr(step_separator)} tokenizes to {sep_ids} (len={len(sep_ids)})."
        if require_single_token_separator:
            raise ValueError(msg + " Require single-token separator; pass --allow_multitoken_separator to override.")
        else:
            print("[Warning] " + msg + " Using first token id; evaluation may be unreliable.")
    sep_token_id = sep_ids[0] if sep_ids else tokenizer.eos_token_id

    # Prompt
    prompt = example.get(prompt_field) or example.get("prompt") or example.get("question") or ""
    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        prompt_text = prompt + "\n\nAssistant:"

    input_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
    labels: List[float] = [-100.0] * len(input_ids)

    steps_raw = example.get(steps_field)
    if steps_raw is None and steps_field != "completions":
        steps_raw = example.get("completions")
    if steps_raw is None and steps_field != "steps":
        steps_raw = example.get("steps")
    if steps_raw is None:
        steps_raw = []
    steps = _clean_steps(steps_raw)

    labels_list = example.get(labels_field)
    if labels_list is None and labels_field != "labels":
        labels_list = example.get("labels")
    if labels_list is None:
        labels_list = []

    # zip truncates if mismatch; we want this explicit
    if isinstance(labels_list, (list, tuple)) and isinstance(steps, list):
        if len(labels_list) != len(steps):
            # keep going, but warn once per example upstream
            pass
    else:
        labels_list = list(labels_list) if labels_list is not None else []

    eval_steps: List[int] = []

    steps_since_sep = 0
    for step_idx_1based, (step, lbl) in enumerate(zip(steps, labels_list), start=1):
        step_text = str(step).strip()
        if step_idx_1based > 1:
            step_text = "\n\n" + step_text

        step_ids = tokenizer(step_text, add_special_tokens=False, truncation=False)["input_ids"]
        input_ids.extend(step_ids)
        labels.extend([-100.0] * len(step_ids))

        steps_since_sep += 1
        is_last = (step_idx_1based == len(steps))  # note: zip truncation could make this false; ok

        if steps_since_sep == eval_every or is_last:
            input_ids.append(sep_token_id)
            try:
                labels.append(float(lbl))
            except Exception:
                labels.append(-1.0)  # invalid label placeholder
            eval_steps.append(step_idx_1based)
            steps_since_sep = 0

    # Apply value baseline transformation if needed
    if value_baseline in ["normalized", "cot_mean"]:
        # Standardize the labels (z-score normalization)
        import numpy as np
        # get all labels that are not -100 (-1.0 placeholders are technically valid floats but we want the 'real' ones)
        # Note: 'labels' list currently has floats. 
        # But wait, 'labels' has -100.0 mixed in. 
        # We need to extract the specific values at separator positions to normalize them, 
        # then put them back.
        
        # 1. Identify indices of separator tokens
        sep_indices = [i for i, x in enumerate(labels) if x != -100.0]
        separator_values = [labels[i] for i in sep_indices]
        if len(separator_values) > 0:
           # print(f"DEBUG: sep_values before: {separator_values[:5]}...")
           pass
        
        if len(separator_values) > 0:
            mean_val = float(np.mean(separator_values))
            
            if value_baseline == "normalized":
                std_val = float(np.std(separator_values))
                if std_val < 1e-8:
                   std_val = 1.0
                new_values = [(lbl - mean_val) / std_val for lbl in separator_values]
            else: # cot_mean
                 new_values = [lbl - mean_val for lbl in separator_values]
                 
            # 2. Update list
            for idx, new_lbl in zip(sep_indices, new_values):
                labels[idx] = new_lbl
            
            # print(f"DEBUG: sep_values after:  {new_values[:5]}...")

    elif value_baseline == "advantage":
        # s_t = s_t - s_{t-1}, assuming s_{-1} = 0.0
        sep_indices = [i for i, x in enumerate(labels) if x != -100.0]
        separator_values = [labels[i] for i in sep_indices]
        
        new_values = []
        prev_val = 0.0
        for lbl in separator_values:
            new_values.append(lbl - prev_val)
            prev_val = lbl
            
        for idx, new_lbl in zip(sep_indices, new_values):
                labels[idx] = new_lbl

    was_truncated = False
    if len(input_ids) > max_length:
        was_truncated = True
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        # If truncation cut off some label positions, eval_steps must be truncated later using valid_count.

    return {
        "input_ids": input_ids,
        "labels": labels,
        "eval_steps": eval_steps,
        "sep_token_id": sep_token_id,
        "was_truncated": was_truncated,
    }


# -----------------------------
# Forward pass helpers
# -----------------------------
def forward_token_scores(
    model,
    tokenizer,
    input_ids: List[int],
    prm_type: str,
    regression_output: str,
) -> np.ndarray:
    """
    Returns per-token scores as numpy float32 array of shape (seq_len,).
    - For classification: returns P(class=1) per token
    - For regression: returns either raw or sigmoid(logit) depending on regression_output
    """
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device, dtype=torch.long)
    attn = (ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        if prm_type == "classification":
            # Prefer SBS-server-style if present
            if hasattr(model, "model") and hasattr(model, "score"):
                base_out = model.model(input_ids=ids, attention_mask=attn, use_cache=False)
                cls_logits = model.score(base_out.last_hidden_state)  # (B, T, 2)
                probs = torch.softmax(cls_logits, dim=-1)
                scores = probs[0, :, 1]
            else:
                out = model(input_ids=ids, attention_mask=attn)
                logits = getattr(out, "logits", None)
                if logits is None:
                    raise RuntimeError("Classification model forward returned no .logits and has no model.score() path.")
                if logits.ndim == 3 and logits.shape[-1] == 2:
                    probs = torch.softmax(logits, dim=-1)
                    scores = probs[0, :, 1]
                else:
                    raise RuntimeError(
                        f"Classification logits shape unsupported: {tuple(logits.shape)}. "
                        "Expected (B,T,2) per-token logits."
                    )
        else:
            out = model(input_ids=ids, attention_mask=attn)
            logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError("Regression model forward returned no .logits")
            scores = logits[0]  # (T,)
            if regression_output == "logit":
                scores = torch.sigmoid(scores)
            # regression_output == "prob": leave as-is

    return scores.to(torch.float32).detach().cpu().numpy()


def extract_valid_pairs(tokenized: Dict[str, Any], token_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Align by label positions (labels != -100). This is robust to truncation.
    Filters invalid labels (<0) as well.
    Returns: (preds, labels, eval_steps_aligned)
    """
    labels = np.asarray(tokenized["labels"], dtype=np.float32)
    valid_pos = np.where(labels != -100.0)[0]

    if valid_pos.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []

    # truncate by available length (should match, but be safe)
    max_pos = int(valid_pos.max())
    if max_pos >= token_scores.shape[0]:
        valid_pos = valid_pos[valid_pos < token_scores.shape[0]]

    preds = token_scores[valid_pos].astype(np.float32)
    y = labels[valid_pos].astype(np.float32)

    # for regression with normalized values, negative labels are valid.
    # only drop if strictly -100 or sentinel (which we already filtered via valid_pos)
    # confusion: valid_pos filters -100.0. 
    # y comes from labels[valid_pos].
    # so we should just keep everything unless we have another sentinel like -1.0
    
    # Check for our -1.0 sentinel from tokenize_example (exception case)
    keep = y != -1.0 
    preds = preds[keep]
    y = y[keep]
    valid_count = int(len(y))

    eval_steps = tokenized.get("eval_steps", [])
    # eval_steps length equals "number of insertions before truncation", but truncation may cut some off.
    if len(eval_steps) >= valid_count:
        eval_steps_aligned = eval_steps[:valid_count]
    else:
        # If this happens, we lost mapping info due to zip mismatch or earlier issues.
        # Fall back to evaluation-point index.
        eval_steps_aligned = list(range(1, valid_count + 1))

    return preds, y, eval_steps_aligned


# -----------------------------
# Plotting
# -----------------------------
def plot_step_distributions(step_labels: Dict[int, List[float]], step_preds: Dict[int, List[float]], output_path: str):
    if not step_labels and not step_preds:
        print("No step-wise data to plot.")
        return

    steps = sorted(
        s for s in set(step_labels.keys()) | set(step_preds.keys())
        if (step_labels.get(s) and len(step_labels[s]) > 0) or (step_preds.get(s) and len(step_preds[s]) > 0)
    )
    if not steps:
        print("No step-wise data to plot.")
        return

    def _mean_ci(vals: List[float]) -> Tuple[float, float]:
        mean = float(np.mean(vals))
        if len(vals) > 1:
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            ci = 1.96 * se
        else:
            ci = 0.0
        return mean, ci

    gt_means, gt_errs, pr_means, pr_errs = [], [], [], []
    for s in steps:
        m, ci = _mean_ci(step_labels.get(s, [])) if step_labels.get(s) else (np.nan, 0.0)
        gt_means.append(m)
        gt_errs.append(ci)
        m, ci = _mean_ci(step_preds.get(s, [])) if step_preds.get(s) else (np.nan, 0.0)
        pr_means.append(m)
        pr_errs.append(ci)

    x = np.arange(len(steps))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, max(4, len(steps) * 0.35)))
    ax.bar(x - width / 2, gt_means, width=width, yerr=gt_errs, capsize=3, label="Ground truth", alpha=0.8)
    ax.bar(x + width / 2, pr_means, width=width, yerr=pr_errs, capsize=3, label="Predictions", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.set_xlabel("Evaluated step index (original step number where <extra_0> inserted)")
    ax.set_ylabel("Mean score")
    ax.set_title("Per-step mean scores (±95% CI)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved step-wise plot to: {output_path}")


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate_model(
    model,
    tokenizer,
    dataset,
    device: str,
    max_length: int,
    step_separator: str,
    evaluate_n_steps: int,
    prompt_field: str,
    steps_field: str,
    labels_field: str,
    value_baseline: str,
    prm_type: str,
    regression_output: str,
    max_examples: Optional[int],
    num_print_examples: int,
    allow_multitoken_separator: bool,
):
    if max_examples is not None:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    print(f"Evaluating on {len(dataset)} examples...")

    # Separator debug
    sep_ids = tokenizer.encode(step_separator, add_special_tokens=False)
    print("\n" + "=" * 60)
    print("STEP SEPARATOR DEBUG")
    print("=" * 60)
    print(f"step_separator string: {repr(step_separator)}")
    print(f"token ids: {sep_ids} (num_tokens={len(sep_ids)})")
    if len(sep_ids) != 1:
        print("[Warning] multi-token separator. If you trained with a single special token, fix this.")
    print("=" * 60 + "\n")

    all_preds: List[float] = []
    all_labels: List[float] = []
    first_positive_results: List[int] = []

    chain_preds_list: List[np.ndarray] = []
    chain_labels_list: List[np.ndarray] = []

    step_labels: Dict[int, List[float]] = defaultdict(list)
    step_preds: Dict[int, List[float]] = defaultdict(list)

    # Coverage/debug
    valid_counts_per_chain: List[int] = []
    raw_steps_per_chain: List[int] = []
    truncated_like_flags = 0
    zip_mismatch_flags = 0

    print_examples_data: List[List[Dict[str, Any]]] = []

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Detect mismatch early for logging
            steps_raw = example.get(steps_field) or example.get("completions") or example.get("steps") or []
            labels_raw = example.get(labels_field) or example.get("labels") or []
            if isinstance(steps_raw, (list, tuple)) and isinstance(labels_raw, (list, tuple)):
                if len(steps_raw) != len(labels_raw):
                    zip_mismatch_flags += 1

            tokenized = tokenize_example(
                example=example,
                tokenizer=tokenizer,
                step_separator=step_separator,
                max_length=max_length,
                evaluate_n_steps=evaluate_n_steps,
                prompt_field=prompt_field,
                steps_field=steps_field,
                labels_field=labels_field,
                value_baseline=value_baseline,
                require_single_token_separator=not allow_multitoken_separator,
            )

            if tokenized["was_truncated"]:
                truncated_like_flags += 1

            # step count heuristic for debug
            if isinstance(steps_raw, (list, tuple)):
                raw_steps_per_chain.append(int(len(steps_raw)))
            else:
                raw_steps_per_chain.append(0)

            token_scores = forward_token_scores(
                model=model,
                tokenizer=tokenizer,
                input_ids=tokenized["input_ids"],
                prm_type=prm_type,
                regression_output=regression_output,
            )

            preds, labels, eval_steps_aligned = extract_valid_pairs(tokenized, token_scores)
            
            # EXCLUDE FINAL STEP (per user request to focus on intermediate steps)
            # if len(labels) > 0:
            #     preds = preds[:-1]
            #     labels = labels[:-1]
            #     eval_steps_aligned = eval_steps_aligned[:-1]

            valid_counts_per_chain.append(int(len(labels)))

            if len(labels) == 0:
                continue

            chain_preds_list.append(preds)
            chain_labels_list.append(labels)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            # step-wise aggregation
            for s_idx, p, y in zip(eval_steps_aligned, preds, labels):
                step_preds[int(s_idx)].append(float(p))
                step_labels[int(s_idx)].append(float(y))

            # first-positive metric (use preds as probabilities)
            first_positive_results.append(evaluate_first_positive_detection(preds, labels, thr=0.5))

            # capture a few examples with decoded segments
            if len(print_examples_data) < num_print_examples:
                # positions of labels in token sequence:
                labels_arr = np.asarray(tokenized["labels"], dtype=np.float32)
                valid_pos = np.where(labels_arr != -100.0)[0]
                # drop invalid labels (<0) to match preds/labels lengths
                valid_lbls = labels_arr[valid_pos]
                keep = valid_lbls >= 0.0
                valid_pos = valid_pos[keep]

                seq_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)

                example_steps = []
                start = 0
                L = min(len(valid_pos), len(preds), len(labels), len(eval_steps_aligned))
                for k in range(L):
                    end = int(valid_pos[k])
                    seg = seq_ids[start : end + 1]
                    text = tokenizer.decode(seg, skip_special_tokens=False)
                    example_steps.append({
                        "step_number": int(eval_steps_aligned[k]),
                        "prediction": float(preds[k]),
                        "label": float(labels[k]),
                        "text": text,
                    })
                    start = end + 1

                print_examples_data.append(example_steps)

        except Exception as e:
            print(f"[Error] Example {i}: {e}")
            continue

    # Coverage summary
    print("\n" + "=" * 60)
    print("EVAL COVERAGE DEBUG")
    print("=" * 60)
    if valid_counts_per_chain:
        vc = np.asarray(valid_counts_per_chain, dtype=np.int32)
        print(
            f"Valid evaluated points per chain: mean={vc.mean():.2f} std={vc.std():.2f} "
            f"min={vc.min()} p50={int(np.percentile(vc, 50))} p90={int(np.percentile(vc, 90))} max={vc.max()}"
        )
        print(f"Most common valid counts: {Counter(vc.tolist()).most_common(10)}")
    if raw_steps_per_chain:
        rs = np.asarray(raw_steps_per_chain, dtype=np.int32)
        print(
            f"Raw step count per chain: mean={rs.mean():.2f} std={rs.std():.2f} "
            f"min={rs.min()} p50={int(np.percentile(rs, 50))} p90={int(np.percentile(rs, 90))} max={rs.max()}"
        )
        print(f"Most common raw step counts: {Counter(rs.tolist()).most_common(10)}")
    print(f"Potential truncation (seq_len >= max_length): {truncated_like_flags}/{len(valid_counts_per_chain)}")
    print(f"Zip length mismatches (steps vs labels): {zip_mismatch_flags}")
    print("=" * 60 + "\n")

    return (
        np.asarray(all_preds, dtype=np.float32),
        np.asarray(all_labels, dtype=np.float32),
        first_positive_results,
        print_examples_data,
        dict(step_labels),
        dict(step_preds),
        chain_preds_list,
        chain_labels_list,
    )


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Robust PRM evaluation (classification or regression)")

    # Model
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--tokenizer_name_or_path", type=str, default=None)
    p.add_argument("--prm_type", type=str, choices=["classification", "regression"], required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (HF)
    p.add_argument("--dataset_repo", type=str, required=True, help="HF dataset repo, e.g. jacopo-minniti/MMLU-PUM-qwen-r1-distil-1.5B")
    p.add_argument("--dataset_config", type=str, default=None, help="HF dataset config/name (subset).")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_examples", type=int, default=None)

    # Fields
    p.add_argument("--prompt_field", type=str, default="prompt")
    p.add_argument("--steps_field", type=str, default="completions")
    p.add_argument("--labels_field", type=str, default="labels")

    # Formatting / evaluation
    p.add_argument("--max_length", type=int, default=8000)
    p.add_argument("--step_separator", type=str, default="<extra_0>")
    p.add_argument("--evaluate_n_steps", type=int, default=1)

    # Regression head interpretation
    p.add_argument("--regression_output", type=str, choices=["prob", "logit"], default="prob",
                   help="If regression head outputs logits, use --regression_output logit to apply sigmoid.")

    # Value Baseline
    p.add_argument("--value_baseline", type=str, default="none", choices=["none", "normalized", "cot_mean", "advantage"],
                   help="Apply baseline transformation to labels (must match training).")

    # Robustness knobs
    p.add_argument("--allow_multitoken_separator", action="store_true",
                   help="Allow separators that tokenize to multiple tokens (not recommended).")
    p.add_argument("--num_print_examples", type=int, default=3)

    # Plot
    p.add_argument("--plot_path", type=str, default="distribution_scores_vs_gt.png")

    args = p.parse_args()

    print("=" * 60)
    print("PRM EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"PRM type:   {args.prm_type}")
    print(f"Device:     {args.device}")
    print(f"Dataset:    {args.dataset_repo} config={args.dataset_config} split={args.split}")
    print(f"Max length: {args.max_length}")
    print(f"Separator:  {repr(args.step_separator)}")
    print(f"Stride:     {args.evaluate_n_steps}")
    print(f"Baseline:   {args.value_baseline}")
    if args.prm_type == "regression":
        print(f"Regression output interpretation: {args.regression_output}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        prm_type=args.prm_type,
        device=args.device,
    )

    print("Loading dataset...")
    if args.dataset_config is None:
        ds = load_dataset(args.dataset_repo, split=args.split)
    else:
        ds = load_dataset(args.dataset_repo, name=args.dataset_config, split=args.split)

    print(f"Dataset size: {len(ds)}")
    print_dataset_quick_stats(ds, num_preview=3)

    (
        preds,
        labels,
        first_positive_results,
        print_examples_data,
        step_labels,
        step_preds,
        chain_preds_list,
        chain_labels_list,
    ) = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        device=args.device,
        max_length=args.max_length,
        step_separator=args.step_separator,
        evaluate_n_steps=args.evaluate_n_steps,
        prompt_field=args.prompt_field,
        steps_field=args.steps_field,
        labels_field=args.labels_field,
        value_baseline=args.value_baseline,
        prm_type=args.prm_type,
        regression_output=args.regression_output,
        max_examples=args.max_examples,
        num_print_examples=args.num_print_examples,
        allow_multitoken_separator=args.allow_multitoken_separator,
    )

    if preds.size == 0:
        print("No valid predictions were generated. Check dataset formatting, separator token, and max_length.")
        return

    # Core metrics (treat preds as probabilities in [0,1] after any sigmoid in forward_token_scores)
    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    r2 = float(r2_score(labels, preds))

    # Correlation
    corr = float(np.corrcoef(labels.astype(np.float64), preds.astype(np.float64))[0, 1]) if len(labels) > 1 else float("nan")

    # Binary metrics
    y_bin = (labels > 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(y_bin, preds))
    except ValueError:
        auroc = 0.5

    ece_10 = calculate_expected_calibration_error(preds, y_bin, n_bins=10)
    ece_5 = calculate_expected_calibration_error(preds, y_bin, n_bins=5)
    ece_15 = calculate_expected_calibration_error(preds, y_bin, n_bins=15)

    # Hard metrics @ 0.5
    pred_bin = (preds > 0.5).astype(int)
    acc = float(accuracy_score(y_bin, pred_bin))
    f1 = float(f1_score(y_bin, pred_bin, zero_division=0))

    # Threshold sweep for best F1
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_thr = 0.5
    best_f1 = -1.0
    thr_rows = []
    for thr in thresholds:
        pb = (preds > thr).astype(int)
        f1_thr = float(f1_score(y_bin, pb, zero_division=0))
        acc_thr = float(accuracy_score(y_bin, pb))
        thr_rows.append((float(thr), f1_thr, acc_thr))
        if f1_thr > best_f1:
            best_f1 = f1_thr
            best_thr = float(thr)

    pb_best = (preds > best_thr).astype(int)
    acc_best = float(accuracy_score(y_bin, pb_best))
    f1_best = float(f1_score(y_bin, pb_best, zero_division=0))

    # First-positive metric
    valid_first_pos = [x for x in first_positive_results if x != -1]
    first_pos_acc = float(np.mean(valid_first_pos)) if valid_first_pos else float("nan")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Valid evaluated points (total): {len(preds)}")

    print("\n--- Soft metrics ---")
    print(f"MSE (Brier): {mse:.6f}")
    print(f"MAE:         {mae:.6f}")
    print(f"R²:          {r2:.6f}")
    print(f"Corr:        {corr:.6f}")
    print(f"AUROC:       {auroc:.6f}")

    print("\n--- Calibration ---")
    print(f"ECE (10 bins): {ece_10:.6f}")
    print(f"ECE (5 bins):  {ece_5:.6f}")
    print(f"ECE (15 bins): {ece_15:.6f}")

    print("\n--- Hard metrics ---")
    print(f"Accuracy @0.5: {acc:.6f}")
    print(f"F1 @0.5:       {f1:.6f}")
    print(f"Best threshold: {best_thr:.2f}")
    print(f"Accuracy @{best_thr:.2f}: {acc_best:.6f}")
    print(f"F1 @{best_thr:.2f}:       {f1_best:.6f}")

    if not np.isnan(first_pos_acc):
        print("\n--- First positive step detection ---")
        print(f"First-positive accuracy: {first_pos_acc:.6f} ({len(valid_first_pos)}/{len(first_positive_results)} chains used)")
    else:
        print("\n--- First positive step detection ---")
        print("No chains with positive labels available (or none evaluable).")

    print("\nLabel stats:")
    print(f"  mean={float(labels.mean()):.6f} std={float(labels.std()):.6f} min={float(labels.min()):.6f} max={float(labels.max()):.6f}")
    print("\nPrediction stats:")
    print(f"  mean={float(preds.mean()):.6f} std={float(preds.std()):.6f} min={float(preds.min()):.6f} max={float(preds.max()):.6f}")

    print("\n--- Threshold sweep (thr | F1 | Acc) ---")
    for thr, f1_thr, acc_thr in thr_rows:
        mark = " *" if abs(thr - best_thr) < 1e-9 else ""
        print(f"{thr:>4.2f} | {f1_thr:>7.4f} | {acc_thr:>7.4f}{mark}")

    # -----------------------------
    # Per-Step and Within-Chain Metrics
    # -----------------------------
    print("\n" + "=" * 60)
    print("GRANULAR METRICS (Per-Step & Within-Chain)")
    print("=" * 60)
    print("Explanation:")
    print("1. Per-Step Index Metrics: Metrics calculated by pooling data ONLY from a specific step number across all chains.")
    print("   (Addresses: 'How well does the model perform specifically at Step X?')")
    print("2. Within-Chain Metrics: Metrics calculated for each chain individually, then averaged.")
    print("   (Addresses: 'Does the model rank steps correctly WITHIN a CoT, or just predict the CoT average?')")

    # 1. Per-Step Index
    print("\n--- 1. Per-Step Index Metrics (Top 10 steps) ---")
    print(f"{'Step':<5} | {'Count':<6} | {'R²':<8} | {'AUROC':<8} | {'MSE':<8}")
    sorted_steps = sorted(step_labels.keys())
    for s in sorted_steps[:10]:  # Limit to first 10 steps to avoid spam
        y_s = np.array(step_labels[s])
        p_s = np.array(step_preds[s])
        
        if len(y_s) < 2:
            print(f"{s:<5} | {len(y_s):<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            continue

        # R2
        r2_s = r2_score(y_s, p_s) if len(y_s) > 1 else float("nan")
        # MSE
        mse_s = np.mean((y_s - p_s) ** 2)
        # AUROC
        try:
            # Need binary labels with both classes
            y_s_bin = (y_s > 0.5).astype(int)
            if len(np.unique(y_s_bin)) > 1:
                auroc_s = roc_auc_score(y_s_bin, p_s)
            else:
                auroc_s = float("nan")
        except:
            auroc_s = float("nan")

        print(f"{s:<5} | {len(y_s):<6} | {r2_s:>8.4f} | {auroc_s:>8.4f} | {mse_s:>8.4f}")

    if len(sorted_steps) > 10:
        print(f"... and {len(sorted_steps) - 10} more steps.")

    # 2. Within-Chain
    print("\n--- 2. Within-Chain Metrics (Averaged) ---")
    
    wc_r2s = []
    wc_aurocs = []
    wc_mses = []
    wc_vars = []

    for p_c, y_c in zip(chain_preds_list, chain_labels_list):
        if len(y_c) < 1:
            continue
        
        # MSE is always valid
        wc_mses.append(np.mean((y_c - p_c) ** 2))
        
        # GT Variance
        wc_vars.append(np.var(y_c))

        # R2 needs variance in y > 0 to be meaningful in standard sense, 
        # but sklearn returns 0.0 for constant prediction if constant label? No, 0 division.
        # If labels are constant, R2 is undefined (division by zero variance).
        if len(y_c) > 1 and np.var(y_c) > 1e-9:
            wc_r2s.append(r2_score(y_c, p_c))
        
        # AUROC needs 2 classes
        try:
             y_c_bin = (y_c > 0.5).astype(int)
             if len(np.unique(y_c_bin)) > 1:
                 val = roc_auc_score(y_c_bin, p_c)
                 wc_aurocs.append(val)
        except:
            pass

    def _stats(vals):
        if not vals:
            return "N/A", "N/A", 0
        return f"{np.mean(vals):.4f}", f"{np.median(vals):.4f}", len(vals)

    mean_r2, med_r2, n_r2 = _stats(wc_r2s)
    mean_roc, med_roc, n_roc = _stats(wc_aurocs)
    mean_mse, med_mse, n_mse = _stats(wc_mses)
    mean_var, med_var, n_var = _stats(wc_vars)

    print(f"{'Metric':<10} | {'Mean':<8} | {'Median':<8} | {'Valid Chains':<12}")
    print(f"{'R²':<10} | {mean_r2:>8} | {med_r2:>8} | {n_r2:<12} (Needs label variance)")
    print(f"{'AUROC':<10} | {mean_roc:>8} | {med_roc:>8} | {n_roc:<12} (Needs both classes)")
    print(f"{'MSE':<10} | {mean_mse:>8} | {med_mse:>8} | {n_mse:<12}")
    print(f"{'GT Var':<10} | {mean_var:>8} | {med_var:>8} | {n_var:<12}")
    
    print("=" * 60 + "\n")

    # -----------------------------
    # 3. Top X% High GT Variance
    # -----------------------------
    print("\n" + "=" * 60)
    print("HIGH GT VARIANCE CHAINS (Top 20%, 10%, 5%, 1%)")
    print("=" * 60)
    
    variances = np.array([np.var(y) if len(y) > 0 else 0.0 for y in chain_labels_list])
    
    if len(variances) > 0:
        for top_pct in [20, 10, 5, 1]:
            print(f"\n>>> Top {top_pct}% High Variance Chains")
            percentile_th = 100 - top_pct
            cutoff = np.percentile(variances, percentile_th)
            high_var_indices = np.where(variances >= cutoff)[0]
            
            subset_preds = [chain_preds_list[i] for i in high_var_indices]
            subset_labels = [chain_labels_list[i] for i in high_var_indices]

            print(f"Variance cutoff (p{percentile_th}): {cutoff:.6f}")
            print(f"Selected chains: {len(high_var_indices)} / {len(chain_labels_list)}")

            if not subset_preds:
                print("Subset empty.")
                print("-" * 30)
                continue

            # Calculate Within-Chain Metrics for this subset
            sub_wc_r2s = []
            sub_wc_aurocs = []
            sub_wc_mses = []

            for p_c, y_c in zip(subset_preds, subset_labels):
                if len(y_c) < 1:
                    continue
                
                # MSE
                sub_wc_mses.append(np.mean((y_c - p_c) ** 2))

                # R2 (needs variance)
                if len(y_c) > 1 and np.var(y_c) > 1e-9:
                    sub_wc_r2s.append(r2_score(y_c, p_c))
                
                # AUROC (needs both classes)
                try:
                    y_c_bin = (y_c > 0.5).astype(int)
                    if len(np.unique(y_c_bin)) > 1:
                        val = roc_auc_score(y_c_bin, p_c)
                        sub_wc_aurocs.append(val)
                except:
                    pass

            # Helper for stats
            def _sub_stats(vals):
                if not vals:
                    return "N/A", "N/A", 0
                return f"{np.mean(vals):.4f}", f"{np.median(vals):.4f}", len(vals)

            m_r2, med_r2, n_r2 = _sub_stats(sub_wc_r2s)
            m_roc, med_roc, n_roc = _sub_stats(sub_wc_aurocs)
            m_mse, med_mse, n_mse = _sub_stats(sub_wc_mses)

            print(f"{'Metric':<10} | {'Mean':<8} | {'Median':<8} | {'Valid Chains':<12}")
            print(f"{'R²':<10} | {m_r2:>8} | {med_r2:>8} | {n_r2:<12}")
            print(f"{'AUROC':<10} | {m_roc:>8} | {med_roc:>8} | {n_roc:<12}")
            print(f"{'MSE':<10} | {m_mse:>8} | {med_mse:>8} | {n_mse:<12}")
            print("-" * 30)

    else:
        print("No chains available for variance analysis.")
    print("=" * 60 + "\n")

    # -----------------------------
    # 4. Length Correlation
    # -----------------------------
    print("\n" + "=" * 60)
    print("LENGTH CORRELATION ANALYSIS")
    print("=" * 60)
    
    chain_lengths = [len(y) for y in chain_labels_list]
    
    # Needs at least 2 points for correlation
    valid_len_idx = [i for i, l in enumerate(chain_lengths) if len(chain_labels_list[i]) > 1]
    
    if len(valid_len_idx) > 5:
        # Pooled metrics (using average per chain as proxy for chain quality vs length)
        pooled_errors = [np.mean((chain_labels_list[i] - chain_preds_list[i])**2) for i in valid_len_idx]
        lengths = [chain_lengths[i] for i in valid_len_idx]
        
        len_mse_corr = np.corrcoef(lengths, pooled_errors)[0, 1]
        print(f"Correlation (Length vs Chain MSE): {len_mse_corr:.4f}")
        
        # Within-chain metrics vs Length
        # (Are metrics for longer chains better or worse?)
        
        # Filter respective metric lists for valid indices (assuming one-to-one mapping with chain_labels_list)
        # Note: wc_r2s etc were built by iterating over all chains. 
        # But some chains might have been skipped there (len < 1).
        # We need to rebuild aligned lists.
        
        aligned_lengths = []
        aligned_wc_r2s = []
        aligned_wc_mses = []
        
        for i in range(len(chain_labels_list)):
            y_c = chain_labels_list[i]
            p_c = chain_preds_list[i]
            if len(y_c) > 1 and np.var(y_c) > 1e-9:
                 aligned_wc_r2s.append(r2_score(y_c, p_c))
                 aligned_wc_mses.append(np.mean((y_c - p_c) ** 2))
                 aligned_lengths.append(len(y_c))

        if len(aligned_lengths) > 5:
            corr_len_r2 = np.corrcoef(aligned_lengths, aligned_wc_r2s)[0, 1]
            corr_len_mse = np.corrcoef(aligned_lengths, aligned_wc_mses)[0, 1]
            
            print(f"Correlation (Length vs Within-Chain R²): {corr_len_r2:.4f}")
            print(f"Correlation (Length vs Within-Chain MSE): {corr_len_mse:.4f}")
            print(f"(Positive R² correlation => Longer chains have better ranking.")
            print(f"(Positive MSE correlation => Longer chains have higher error.")
        else:
            print("Not enough valid chains with variance for ranking correlation.")
            
    else:
        print("Not enough chains for length correlation analysis.")
        
    print("=" * 60 + "\n")

    if print_examples_data:
        print("\n" + "=" * 60)
        print("STEP-BY-STEP EXAMPLES (decoded segments up to each eval point)")
        print("=" * 60)
        for i, steps in enumerate(print_examples_data):
            print(f"\nExample {i + 1}:")
            for s in steps:
                txt = s["text"].replace("\n", " ").strip()
                if len(txt) > 240:
                    txt = txt[:237] + "..."
                print(f"  Step {s['step_number']}: pred={s['prediction']:.4f} label={s['label']:.4f}")
                print(f"    {txt}")

    plot_step_distributions(step_labels, step_preds, args.plot_path)


if __name__ == "__main__":
    main()
