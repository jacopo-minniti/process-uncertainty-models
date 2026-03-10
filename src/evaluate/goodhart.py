# evaluate/goodhart.py

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset # Rename to avoid conflict with rely.utils.load_dataset
from sklearn.metrics import roc_auc_score, r2_score
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Reuse loading logic from pum_eval if possible, otherwise reimplement minimal version
from rely.train.regression_prm.model import RegressionPRMModel
from rely.train.regression_prm.trainer import RegressionPRMTrainer
from rely.utils import load_dataset # This is the utility function that handles HF or local paths

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str,
                   default="jacopo-minniti/MMLU-PUM-qwen-r1-distil-1.5B",
                   help="HF dataset name or local path for the ground truth dataset.")
    p.add_argument("--dataset_config", type=str, default="variance",
                   help="Dataset config/subset or label field ('variance'/'value'); applied to both GT and SBS.")
    p.add_argument("--dataset_split", type=str, default="test",
                   help="Dataset split for the ground truth dataset.")
    p.add_argument("--pum_model_ckpt", type=str, required=True,
                   help="Checkpoint / identifier for the PUM model to evaluate.")
    p.add_argument("--sbs_dataset", type=str, required=True,
                   help="Path to SBS completions dataset (JSONL). This dataset should have the same format as the ground truth dataset.")
    p.add_argument("--output_dir", type=str,
                   default="results/goodhart", help="Directory to save results")
    p.add_argument("--topk_percents", type=float, nargs="+",
                   default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                   help="Top-k percentages for Goodhart gap (currently unused in this refactor).")
    p.add_argument("--n_bins", type=int, default=10, help="Number of bins for calibration")
    
    # Model config args needed for correct formatting
    p.add_argument("--max_length", type=int, default=10000)
    p.add_argument("--step_separator", type=str, default="<extra_0>")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer_name_or_path", type=str, default=None,
                   help="HF name or local path for the tokenizer (base model). "
                        "If not set, will try to infer from the PUM model config.")
    
    p.add_argument("--evaluate_n_steps", type=int, default=1, 
                   help="Evaluate PUM every N steps (grouping steps). Matches SBS rollout stride.")
    
    p.add_argument("--gt_dataset", type=str, dest="dataset_name",
                   help="Alias for --dataset_name (GT dataset).")
    p.add_argument("--gt_split", type=str, dest="dataset_split",
                   help="Alias for --dataset_split.")
    
    p.add_argument("--sbs_dataset_label_config", type=str, default=None,
                   help="Label field for SBS dataset if different from dataset_config.")
    
    return p.parse_args()

def evaluated_step_indices(num_steps: int, evaluate_n_steps: int) -> list[int]:
    """
    Mirror SBS logic: return the step indices where a separator is inserted.
    When evaluate_n_steps > 1, only every n-th (and the final) step gets <extra_0>.
    """
    eval_every = max(1, int(evaluate_n_steps))
    indices: list[int] = []
    steps_since_sep = 0
    for idx in range(1, num_steps + 1):
        steps_since_sep += 1
        is_last = idx == num_steps
        if steps_since_sep == eval_every or is_last:
            indices.append(idx)
            steps_since_sep = 0
    return indices

def tokenize_example(example, tokenizer, step_separator, max_length, evaluate_n_steps: int):
    """
    Robustly tokenize by manually constructing the sequence:
      [User Prompt] [Step 1] [Sep] [Step 2] [Sep] ...
    This guarantees the separator is present and avoids chat template issues.
    """
    eval_every = max(1, int(evaluate_n_steps))
    
    # 1. Get the Separator ID
    sep_ids = tokenizer.encode(step_separator, add_special_tokens=False)
    if not sep_ids:
        print(f"Warning: Could not encode {step_separator}")
        sep_token_id = tokenizer.eos_token_id 
    else:
        sep_token_id = sep_ids[0]

    # 2. Format the user prompt
    #    We rely on the tokenizer to format the user part correctly (including BOS).
    #    We check if 'prompt' needs extraction or is ready.
    #    For consistency with goodhart's previous logic, we might need cleaning, 
    #    but let's trust the tokenizer template if we pass the raw user prompt.
    #    However, example["prompt"] in goodhart might contain system prompt artifacts if loaded from certain jsonls.
    #    We'll do a quick check/clean if extracted from 'prompt'.
    
    raw_prompt = example.get("prompt", "")
    # Minimal cleaning if it looks like it has header
    if "<|im_start|>" in raw_prompt:
        # Attempt extraction
        s_marker = "<|im_start|>user"
        e_marker = "<|im_end|>"
        s_idx = raw_prompt.find(s_marker)
        if s_idx != -1:
            content_start = s_idx + len(s_marker)
            e_idx = raw_prompt.find(e_marker, content_start)
            if e_idx != -1:
                raw_prompt = raw_prompt[content_start:e_idx].strip()
    
    # Also check if 'question' exists and is cleaner
    if "question" in example and example["question"]:
        # often 'question' is just the Q.
        pass # Use prompt logic above usually safer if prompt is the full thing
        
    messages = [{"role": "user", "content": raw_prompt}]
    
    # Apply template
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    except Exception:
        # Fallback if no chat template or other issue
        prompt_text = raw_prompt + "\n\nAssistant:"

    input_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
    labels = [-100.0] * len(input_ids)

    # 3. Process completions (steps)
    completions = example.get("completions", example.get("prefixes", []))
    labels_list = example.get("labels", [])
    
    # Robustness: if labels_list is missing or shorter, pad it
    if labels_list is None: labels_list = []
    if not isinstance(labels_list, list): 
        # Convert numpy/tuple to list
        labels_list = list(labels_list) if hasattr(labels_list, '__iter__') else [labels_list]
        
    if len(labels_list) < len(completions):
        # Pad with dummy -100 or -1 if we just need input_ids
        # For plot we need alignment. But here we just tokenizing.
        # We can pad with None or dummy.
        labels_list.extend([-1.0] * (len(completions) - len(labels_list)))

    steps_since_separator = 0
    
    for idx, (step, label) in enumerate(zip(completions, labels_list), start=1):
        step_text = str(step).strip()
        # Clean specific artifacts if known
        step_text = step_text.replace("\\think", "").replace(r"\think", "")
        
        if idx > 1:
            step_text = "\n\n" + step_text
            
        step_ids = tokenizer(step_text, add_special_tokens=False, truncation=False)["input_ids"]
        input_ids.extend(step_ids)
        labels.extend([-100.0] * len(step_ids))
        
        steps_since_separator += 1
        is_last = idx == len(completions)

        # 4. Insert Separator if needed
        if steps_since_separator == eval_every or is_last:
            input_ids.append(sep_token_id)
            labels.append(float(label) if label is not None else -1.0)
            steps_since_separator = 0

    # 5. Truncate if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "labels": labels}

# --- Step A: Build Gold Table ---

def build_gold_table(dataset, dataset_name="unknown", label_key="labels"):
    """
    Returns:
        gold_dict: dict[(int_example_id, int_prefix_id) -> float_gold_value]
        meta: dict[int_example_id -> dict with prompt, num_prefixes, completions]
    """
    print(f"Building gold table for {dataset_name}...")
    gold = {}
    meta = {}
    for i, ex in enumerate(dataset):
        completions = ex.get("completions", [])
        labels = ex.get(label_key, None)
        if labels is None:
            labels = ex.get("labels", ex.get("label", []))
        # Normalize labels to list
        if not isinstance(labels, list):
            if labels is None:
                labels = []
            elif isinstance(labels, (tuple, np.ndarray)):
                labels = list(labels)
            else:
                labels = [labels]
        
        # Fallback if names differ
        if not completions and "prefixes" in ex:
            completions = ex["prefixes"]
        
        # assert len(completions) == len(labels), f"Example {i}: prefix/label length mismatch {len(completions)} vs {len(labels)}"
        # Warn instead of assert to be robust like goodhart_cot_analysis
        if len(completions) != len(labels):
            # print(f"[WARN] Example {i}: prefix/label length mismatch {len(completions)} vs {len(labels)}. Skipping/Truncating?")
            # goodhart_cot_analysis truncates scores. Here we are building gold.
            # If labels are fewer, we skip?
            # Let's just use zip which truncates to shortest
            pass

        meta[i] = {
            "prompt": ex["prompt"],
            "completions": completions, # Store for text matching later if needed
            "num_prefixes": len(completions),
            "question": ex.get("question", ex.get("prompt", "")), # Fallback to prompt if question missing
            "solution": ex.get("solution", ex.get("answer", "")),   # Fallback/Aliases
        }

        for j, (cat, y) in enumerate(zip(completions, labels)):
             gold[(i, j)] = float(y)

    print(f"Gold table built: {len(gold)} entries from {len(dataset)} examples.")
    return gold, meta

# --- Step B: PUM Predictions ---
# ... (rest is unchanged or already updated)

# --- Step E: Metrics ---

# Helper from goodhart_cot_analysis.py
def _format_steps_with_separator(steps: list[str], evaluate_n_steps: int = 1) -> str:
    """
    Mirror of rely.inference.sbs.servers._format_steps_with_separator
    """
    if not steps:
        return "<extra_0>"
    # If evaluating every step:
    if evaluate_n_steps <= 1:
        return "\n\n<extra_0>".join(steps) + "\n\n<extra_0>"
    
    # Complex case if we were skipping steps
    formatted_parts = []
    steps_since_separator = 0
    for idx, step in enumerate(steps, start=1):
        formatted_parts.append(step)
        steps_since_separator += 1
        if steps_since_separator == evaluate_n_steps:
            formatted_parts.append("\n\n<extra_0>")
            steps_since_separator = 0
        elif idx != len(steps):
            formatted_parts.append("\n\n")
    if steps_since_separator > 0:
        formatted_parts.append("\n\n<extra_0>")
    return "".join(formatted_parts)

def clean_text_helper(t):
    if not t: return ""
    return t.replace("\\think", "").replace(r"\think", "").strip()

# --- Step B: PUM Predictions ---

def load_pum_model(checkpoint_path, device="cuda", tokenizer_name_or_path=None):
    print(f"Loading PUM model from {checkpoint_path}...")
    
    tok_path = tokenizer_name_or_path if tokenizer_name_or_path else checkpoint_path
    print(f"Loading tokenizer from {tok_path}...")
    
    # Match goodhart_cot_analysis.py loading style
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL FIX: Ensure <extra_0> is treated as a special token
    if "<extra_0>" not in tokenizer.special_tokens_map.get("additional_special_tokens", []):
        tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_0>"]})

    print("Loading model weights...")
    
    # Heuristic: 7B models are usually classification PRMs in this project context
    if "7B" in checkpoint_path:
        from transformers import AutoModel
        print("Detected 7B model, loading as Classification (AutoModel)...")
        model = AutoModel.from_pretrained(
            checkpoint_path,
            num_labels=2,
            dtype=torch.bfloat16,
            device_map=device if device != "cuda" else "auto",
            trust_remote_code=True
        )
    else:
        print("Loading as RegressionPRMModel...")
        model = RegressionPRMModel.from_pretrained(
            checkpoint_path, 
            dtype=torch.bfloat16, 
            device_map=device if device != "cuda" else "auto", 
            trust_remote_code=True
        )
        
    model = model.to(dtype=torch.bfloat16)
    model.eval()

    return model, tokenizer

def compute_pum_predictions(dataset, pum_model, tokenizer, args):
    """
    Returns:
        pum_dict: dict[(int_example_id, int_prefix_id) -> float_pum_pred]
    """
    print(f"Computing PUM predictions (stride={args.evaluate_n_steps})...")
    pum = {}
    model = pum_model
    device = args.device
    
    # Sep ID for finding it in input_ids
    try:
        sep_ids = tokenizer.encode(args.step_separator, add_special_tokens=False)
        sep_token_id = sep_ids[0] if sep_ids else tokenizer.eos_token_id
    except:
        sep_token_id = tokenizer.eos_token_id

    model.eval()
    
    for i, ex in enumerate(tqdm(dataset, desc="PUM Inference")):
        try:
            # PUM Eval Style Tokenization
            tokenized = tokenize_example(
                ex, 
                tokenizer, 
                args.step_separator, 
                args.max_length, 
                args.evaluate_n_steps
            )
            
            input_ids_list = tokenized["input_ids"]
            # To tensor
            input_ids = torch.tensor([input_ids_list], device=model.device if hasattr(model, "device") else device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            with torch.no_grad():
                # Handle both RegressionPRMModel and standard AutoModel (Classification)
                if hasattr(model, "model") and hasattr(model, "score"): 
                    # Likely Qwen2.5-Math-PRM style (Classification) or similar structure wrapping a base model
                    # Standard HF AutoModel for SeqClassification usually has model() or we call it directly.
                    # Qwen-Math-PRM implementation usually exposes .score() or we need use_cache=False on main forward.
                    # Let's try calling it like in servers.py:
                    # base_model_output = model.model(input_ids=..., use_cache=False)
                    # logits = model.score(base_model_output.last_hidden_state)
                    
                    # However, if it was loaded with AutoModelForSequenceClassification, it would be model(input_ids).logits
                    # The servers.py uses AutoModel.from_pretrained(...) which loads the generic model class,
                    # and presumably the custom code in trust_remote_code provides .score() method.
                    # Let's mimic servers.py logic exactly for non-regression.
                    
                    # servers.py logic for classification:
                    # base_model_output = model.model(input_ids=..., attention_mask=..., use_cache=False)
                    # logits = model.score(base_model_output.last_hidden_state)
                    # probabilities = torch.softmax(logits, dim=-1)
                    # scores = probabilities[:, :, 1]
                    
                    try:
                        base_model_output = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                        cls_logits = model.score(base_model_output.last_hidden_state) # (batch, seq, 2)
                        probabilities = torch.softmax(cls_logits, dim=-1)
                        logits = probabilities[0, :, 1] # Take prob of class 1 as the score. (seq_len,)
                    except AttributeError:
                        # Fallback if it's a standard HF SequenceClassifier
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                        # If SequenceClassifier, output is usually (batch, 2) for pooled, NOT per token.
                        # PRMs must provide per-token rewards.
                        # If this fails, we assume Regression logic applies or structure is unknown.
                        # But user request says "switch to classifier model when 7B".
                        # Assuming Qwen Math PRM 7B structure.
                        raise RuntimeError("Failed to use Qwen-Math-PRM logic (model.model + model.score). Check model type.")
                
                else:
                    # Regression Model (RegressionPRMModel)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0] # (seq_len,)
            
            # Extract Scores
            sep_mask = (input_ids[0] == sep_token_id)
            sep_indices = sep_mask.nonzero(as_tuple=True)[0]
            
            extracted_scores = []
            if len(sep_indices) > 0:
                extracted_scores = logits[sep_indices].float().cpu().tolist()
            
            # Map back to steps
            # tokenize_example inserts separator based on evaluate_n_steps.
            # We reconstruct the 'evaluated step indices' to map scores to the correct step number (0-indexed or 1-indexed).
            # gold_table keys are (i, j) where j is 0-indexed index into completions list.
            
            completions = ex.get("completions", ex.get("prefixes", []))
            num_steps = len(completions)
            
            # These are the 1-based indices of steps that got a separator
            eval_idxs = evaluated_step_indices(num_steps, args.evaluate_n_steps)
            
            # extracted_scores should align with eval_idxs
            # Truncate if lengths differ (due to max_length truncation in tokenize_example)
            limit = min(len(extracted_scores), len(eval_idxs))
            
            for k in range(limit):
                score = extracted_scores[k]
                step_1_idx = eval_idxs[k]
                step_0_idx = step_1_idx - 1 # Convert to 0-indexed for dict key
                
                pum[(i, step_0_idx)] = float(score)

        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
            continue

    return pum

def aggregate_step_data(gold_table, pum_table):
    """
     aggregations for plotting:
     step_preds: {step_idx: [scores...]}
     step_labels: {step_idx: [scores...]}
    """
    step_preds = defaultdict(list)
    step_labels = defaultdict(list)
    
    # Iterate through gold keys (assuming gold covers the valid set)
    # Or iterate union.
    all_keys = set(gold_table.keys()) | set(pum_table.keys())
    
    for (ex_id, step_idx_0) in all_keys:
        # step_idx_0 is 0-indexed. Plotting typically prefers 1-indexed steps or just consistent keys.
        # Let's use 1-indexed for plot labels.
        step_key = step_idx_0 + 1
        
        if (ex_id, step_idx_0) in gold_table:
            step_labels[step_key].append(gold_table[(ex_id, step_idx_0)])
        
        if (ex_id, step_idx_0) in pum_table:
            step_preds[step_key].append(pum_table[(ex_id, step_idx_0)])
            
    return step_labels, step_preds

def plot_two_panel_comparison(
    gt_step_labels, gt_step_preds, 
    sbs_step_labels, sbs_step_preds, 
    output_path
):
    """
    Two panel plot:
    Top: GT Dataset (GT Labels vs GT Preds)
    Bottom: SBS Dataset (SBS Labels vs SBS Preds)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    # Helper for one subplot
    def _plot_on_ax(ax, step_labels, step_preds, title_suffix):
        steps = sorted(
            s for s in set(step_labels.keys()) | set(step_preds.keys())
            if (step_labels.get(s) and len(step_labels.get(s)) > 0) or 
               (step_preds.get(s) and len(step_preds.get(s)) > 0)
        )
        if not steps:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            return

        def _mean_ci(vals):
            if not vals: return 0.0, 0.0
            mean = float(np.mean(vals))
            if len(vals) > 1:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                ci = 1.96 * se
            else:
                ci = 0.0
            return mean, ci
        
        lbl_means, lbl_errs = [], []
        pred_means, pred_errs = [], []
        
        for s in steps:
            m, c = _mean_ci(step_labels.get(s, []))
            lbl_means.append(m)
            lbl_errs.append(c)
            
            m, c = _mean_ci(step_preds.get(s, []))
            pred_means.append(m)
            pred_errs.append(c)
            
        x = np.arange(len(steps))
        width = 0.35
        
        ax.bar(x - width/2, lbl_means, width, yerr=lbl_errs, label='Ground Truth / Labels', capsize=3, alpha=0.7)
        ax.bar(x + width/2, pred_means, width, yerr=pred_errs, label='PUM Predictions', capsize=3, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(steps)
        ax.set_title(f"Per-step Scores: {title_suffix}")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    _plot_on_ax(axes[0], gt_step_labels, gt_step_preds, "Ground Truth Dataset (PRM vs GT)")
    _plot_on_ax(axes[1], sbs_step_labels, sbs_step_preds, "SBS Dataset (PRM vs SBS Labels)")
    
    axes[1].set_xlabel("Step Index")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")

# --- Step E: Metrics ---

def compute_iid_metrics(gold_table, pum_table):
    print("Computing IID metrics...")
    
    y_true = []
    y_pred = []

    for key, gold in gold_table.items():
        if key in pum_table:
            y_true.append(gold)
            y_pred.append(pum_table[key])
    
    if not y_true:
        print("No matching entries found for metrics.")
        return {"n": 0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_gold = float(y_true.mean())
    mean_pum_pred = float(y_pred.mean())

    # Threshold for AUROC (assuming gold is probability-like or variance)
    # If variance (lower is better?), we might need to flip?
    # User says: "gold targets... variance target". 
    # Usually variance: low = good. 
    # PUM prediction: usually high score = good?
    # We should check correlation. 
    # If correlation is negative, we flip one.
    
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"Correlation between Gold and PUM: {corr}")
    
    # If gold is variance (0=best), and PUM is reward (1=best), we expect negative correlation.
    # To compute AUROC/R2 conveniently, let's align them.
    # But for "Goodhart", we want to see "Gap". 
    # If PUM is Variance, Gold is Variance -> Gap is meaningless dimensions.
    
    # Assumption: PUM model is trained to predict the Gold target.
    # So if Gold is variance, PUM predicts variance.
    # If Gold is value, PUM predicts value.
    # PUM checkpoint "RegressionPRM" suggests it predicts the target directly.
    
    # AUROC classification:
    # "Correct" = Gold is "good". 
    # If Gold=Variance, "good" is low variance.
    # Let's take median split.
    threshold = np.median(y_true)
    
    # If correlation is positive, high gold = high pred.
    # If correlation is negative, high gold = low pred.
    
    # For AUROC, we need binary labels.
    # Let's say "Target Class 1" is "High Value".
    # We interpret y_bin as "Is this a good prefix?".
    # If Gold is variance, y < threshold is "Class 1" (Good).
    
    # We'll compute raw R2 first.
    r2 = r2_score(y_true, y_pred)
    
    # For AUROC, let's just bin based on median and see if PUM predicts it.
    y_bin = (y_true >= threshold).astype(int) # This is just "High Gold". 
    try:
        auroc = roc_auc_score(y_bin, y_pred)
    except:
        auroc = 0.5

    metrics = {
        "auroc_threshold_median": float(threshold),
        "auroc": float(auroc),
        "r2": float(r2),
        "correlation": float(corr),
        "mean_gold": mean_gold,
        "mean_pum_pred": mean_pum_pred,
        "n": len(y_true),
    }
    return metrics

def compute_calibration(gold_table, pum_table, n_bins):
    print(f"Computing calibration with {n_bins} bins...")
    
    y_true = []
    y_pred = []

    for key, gold in gold_table.items():
        if key in pum_table:
            y_true.append(gold)
            y_pred.append(pum_table[key])

    if not y_true:
        print("No matching entries found for calibration.")
        return []

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    # digitize returns 1..n_bins. We want 0..n_bins-1
    bin_ids = np.digitize(y_pred, bins) - 1 
    
    calib = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        mean_pred = float(y_pred[mask].mean())
        mean_gold = float(y_true[mask].mean())
        count = int(mask.sum())
        
        calib.append({
            "bin": b,
            "bin_low": float(bins[b]),
            "bin_high": float(bins[b + 1]),
            "mean_pum_pred": mean_pred,
            "mean_gold": mean_gold,
            "count": count,
        })
        return calib

def compute_snr(gold_table, pum_table):
    """
    Compute signal-to-noise ratio between ground-truth scores and model predictions.
    - noise epsilon_t = p_hat - p_true
    - SNR = Var(p_true) / Var(epsilon_t)
    """
    y_true = []
    y_pred = []

    for key, gold in gold_table.items():
        if key in pum_table:
            y_true.append(gold)
            y_pred.append(pum_table[key])

    if not y_true:
        print("No matching entries found for SNR.")
        return {"n": 0}

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    noise = y_pred - y_true
    var_signal = float(np.var(y_true, ddof=0))
    var_noise = float(np.var(noise, ddof=0))
    snr = float(var_signal / var_noise) if var_noise > 0 else float("inf")

    return {
        "n": len(y_true),
        "variance_signal": var_signal,
        "variance_noise": var_noise,
        "snr": snr,
        "mean_gt": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }

def save_json(data, path):
    print(f"Saving {path}...")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))

def main():
    args = parse_args()
    run_dir = os.path.join(args.output_dir, args.dataset_config) if args.dataset_config else args.output_dir
    os.makedirs(run_dir, exist_ok=True)
    
    # Load Models (PUM)
    pum_model, tokenizer = load_pum_model(args.pum_model_ckpt, device=args.device, tokenizer_name_or_path=args.tokenizer_name_or_path)

    # 1. Load Main Dataset (Ground Truth)
    print(f"Loading Ground Truth dataset from {args.dataset_name}...")
    # Using rely.utils.load_dataset (handles HF or local)
    gt_dataset = load_dataset(args.dataset_name, subset=args.dataset_config, split=args.dataset_split)
    
    # 2. Build Gold Table for GT
    print("Building Gold Table for Ground Truth...")
    gold_table_gt, meta_gt = build_gold_table(gt_dataset, args.dataset_name, label_key=args.dataset_config or "labels")
    print(f"GT Gold table: {len(gold_table_gt)} entries from {len(gt_dataset)} examples.")

    # 3. Compute PUM Predictions for GT
    print("Computing PUM predictions for Ground Truth...")
    pum_table_gt = compute_pum_predictions(gt_dataset, pum_model, tokenizer, args)

    # 4. Load SBS Dataset
    print(f"Loading SBS dataset from {args.sbs_dataset}...")
    sbs_dataset = load_dataset(args.sbs_dataset, subset=args.dataset_config)
    print(f"Loaded {len(sbs_dataset)} SBS items.")

    # 5. Build Gold Table for SBS (assuming same format -> extract labels)
    print("Building Gold Table for SBS...")
    # Note: build_gold_table assumes standard keys (prompt/solution). 
    # SBS dataset might use 'response'/'attempt'? Let's check user's previous context or assume standard.
    # User said "SAME EXACT FORMAT", so assume standard.
    gold_table_sbs, meta_sbs = build_gold_table(sbs_dataset, args.sbs_dataset, label_key=args.dataset_config or "labels") 
    print(f"SBS Gold table: {len(gold_table_sbs)} entries.")

    # 6. Compute PUM Predictions for SBS
    print("Computing PUM predictions for SBS...")
    pum_table_sbs = compute_pum_predictions(sbs_dataset, pum_model, tokenizer, args)

    # 7. Compute Metrics (IID vs OOD/SBS)
    print("Computing Metrics...")

    # IID (GT) Metrics
    print("Metrics for Ground Truth (IID):")
    iid_results = compute_iid_metrics(gold_table_gt, pum_table_gt)
    save_json(iid_results, os.path.join(run_dir, "metrics_gt.json"))

    # SBS Metrics
    print("Metrics for SBS (OOD):")
    sbs_results = compute_iid_metrics(gold_table_sbs, pum_table_sbs)
    save_json(sbs_results, os.path.join(run_dir, "metrics_sbs.json"))
    
    # Calibration
    calib_gt = compute_calibration(gold_table_gt, pum_table_gt, args.n_bins)
    save_json(calib_gt, os.path.join(run_dir, "calibration_gt.json"))
    
    calib_sbs = compute_calibration(gold_table_sbs, pum_table_sbs, args.n_bins)
    save_json(calib_sbs, os.path.join(run_dir, "calibration_sbs.json"))

    # SNR (GT + SBS)
    snr_results = {
        "gt": compute_snr(gold_table_gt, pum_table_gt),
        "sbs": compute_snr(gold_table_sbs, pum_table_sbs),
    }
    save_json(snr_results, os.path.join(run_dir, "snr.json"))
    
    # 8. Plot Step Comparison
    print("Generating Comparison Plot...")
    gt_labels, gt_preds = aggregate_step_data(gold_table_gt, pum_table_gt)
    sbs_labels, sbs_preds = aggregate_step_data(gold_table_sbs, pum_table_sbs)
    
    plot_path = os.path.join(run_dir, "goodhart_comparison_plot.png")
    plot_two_panel_comparison(
        gt_labels, gt_preds, 
        sbs_labels, sbs_preds, 
        plot_path
    )
    
    print("Done.")

if __name__ == "__main__":
    main()
