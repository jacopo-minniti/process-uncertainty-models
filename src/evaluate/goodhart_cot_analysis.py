#!/usr/bin/env python
# goodhart_cot_analysis.py

import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from rely.train.regression_prm.model import RegressionPRMModel
from rely.utils import load_dataset, prompt_pattern  # handles HF or local paths


def parse_args():
    p = argparse.ArgumentParser()

    # Ground-truth / original dataset (same format as SBS)
    p.add_argument(
        "--gt_dataset",
        type=str,
        required=True,
        help="HF dataset name or local path for the original (ground-truth) dataset.",
    )
    p.add_argument(
        "--gt_split",
        type=str,
        default="test",
        help="Split for the original dataset (ignored for plain JSONL/local lists).",
    )

    # SBS dataset (same format as GT)
    p.add_argument(
        "--sbs_dataset",
        type=str,
        required=True,
        help="HF dataset name or local path for the SBS dataset.",
    )

    # PUM model
    p.add_argument(
        "--pum_model_ckpt",
        type=str,
        required=True,
        help="Checkpoint / identifier for the PUM regression model.",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for PUM inference (per joined sequence).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Max sequence length for tokenization.",
    )

    p.add_argument(
        "--output_path",
        type=str,
        default="goodhart_info.json",
        help="Where to write the final info.json file.",
    )

    return p.parse_args()



def load_pum_model_and_tokenizer(ckpt: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CRITICAL FIX: Ensure <extra_0> is treated as a special token so it splits
    # correctly from following text. Use add_special_tokens to map to existing ID if in vocab.
    if "<extra_0>" not in tokenizer.special_tokens_map.get("additional_special_tokens", []):
        tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_0>"]})

    # PUM models often trained/served with flash-attn require fp16/bf16
    model = RegressionPRMModel.from_pretrained(ckpt, dtype=torch.bfloat16)
         
    model.to(device)
    model.to(dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model


def ensure_list(obj):
    """
    HF datasets sometimes give 'completions'/'labels' as non-list types.
    This makes sure we always work with Python lists.
    """
    if isinstance(obj, list):
        return obj
    # e.g., if it is a numpy array or similar
    try:
        return list(obj)
    except TypeError:
        return [obj]


def _format_steps_with_separator(steps: list[str], evaluate_n_steps: int = 1) -> str:
    """
    Mirror of rely.inference.sbs.servers._format_steps_with_separator
    but defaulting to 1 step evaluation since that's what we usually want 
    for detailed analysis.
    """
    if not steps:
        return "<extra_0>"
    # If evaluating every step:
    if evaluate_n_steps <= 1:
        return "\n\n<extra_0>".join(steps) + "\n\n<extra_0>"
    
    # Complex case if we were skipping steps (less relevant here but good for consistency)
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


@torch.no_grad()
def pum_scores_for_examples(examples, tokenizer, model, device, max_length: int, batch_size: int):
    """
    Given a list of examples (dicts with 'prompt', 'completions', 'labels'),
    return a dict: prompt -> dict with:
        {
            "prompt": str,
            "completions": [...],
            "labels": [...],
            "pum_scores": [...]
        }

    If anything mismatches (len(labels) vs len(pum_scores)), we print an error
    and skip that example.
    """
    results_by_prompt = {}

    # We'll process in simple batches; each batch element is a full joined sequence
    batch_texts = []
    batch_meta = []  # (prompt, completions, labels)

    def flush_batch():
        nonlocal batch_texts, batch_meta
        if not batch_texts:
            return

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # ASSUMPTION: RegressionPRMModel forward returns an object with
        # attribute `.logits` of shape [batch_size, seq_len]
        outputs = model(**enc)
        
        # Extract scores at <extra_0> tokens
        # Copy strictly from servers.py
        sep_token_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
        # print(f"[DEBUG] sep_token_id: {sep_token_id}")
        
        logits = outputs.logits
        input_ids = enc["input_ids"]
        seq_len = input_ids.size(1)
        
        all_scores = []
        for b_idx in range(input_ids.size(0)):
            # Find indices where the token is <extra_0>
            sep_mask = (input_ids[b_idx] == sep_token_id)
            sep_indices = sep_mask.nonzero(as_tuple=True)[0]
            
            # print(f"[DEBUG] Batch item {b_idx}: Found {len(sep_indices)} separators at indices {sep_indices.tolist()}")
            
            # Extract scores at these indices
            if len(sep_indices) > 0:
                scores_vec = logits[b_idx, sep_indices]
            else:
                scores_vec = torch.tensor([], device=logits.device)
            
            all_scores.append(scores_vec)
            
        for i, (prompt, completions, labels) in enumerate(batch_meta):
            scores = all_scores[i].detach().cpu().tolist()
            labels = ensure_list(labels)
            completions = ensure_list(completions)

            # Some models may predict more scores than completions (e.g., padding).
            # We assume the first `len(completions)` scores correspond to our steps.
            if len(scores) < len(completions):
                # Check for Truncation
                is_truncated = (seq_len >= max_length)
                
                print(f"\n[ERROR] PUM returned fewer scores ({len(scores)}) than completions ({len(completions)})")
                if is_truncated:
                    print(f"[CRITICAL] Truncation Detected! Sequence reached max_length ({max_length}).")
                    print("--> This is why scores are missing. The end of the sequence (containing separators) was cut off.")
                    print("--> Action: Increase --max_length in arguments.")
                
                print(f"Prompt preview: {prompt[:100]!r}")
                
                # Check the actual text used for this item (it was in batch_texts[i], need to recover)
                # Since we don't have easy access to the specific batch_text string here without re-constructing,
                # we will rely on checking the tokenizer output for this specific sequence if possible, 
                # or just look at the input_ids.
                
                raw_ids = input_ids[i].tolist()
                print(f"DEBUG: sep_token_id used: {sep_token_id}")
                sep_count_in_ids = raw_ids.count(sep_token_id)
                print(f"DEBUG: Count of sep_token_id in input_ids: {sep_count_in_ids}")
                
                # Try to decode the input_ids back to text to see what the tokenizer "saw"
                if not is_truncated:
                     # Only print detailed debug if it wasn't just simple truncation, to save log space
                     decoded = tokenizer.decode(raw_ids, skip_special_tokens=False)
                     print(f"DEBUG: Decoded input_ids (snippet): {decoded[-500:]!r}")
                
                print("Skipping this item.")
                continue

            if len(labels) != len(completions):
                # Trim scores to the number of completions
                print("Trimming scores")
                scores = scores[: len(completions)]

            results_by_prompt[prompt] = {
                "prompt": prompt,
                "completions": completions,
                "labels": labels,
                "pum_scores": scores,
            }

        batch_texts = []
        batch_meta = []

    for ex in tqdm(examples, desc="Scoring with PUM"):
        prompt = ex.get("prompt", "")
        # We assume completions are a list of step strings
        completions = ensure_list(ex.get("completions", []))
        labels = ensure_list(ex.get("labels", []))

        if not completions:
            print(f"[ERROR] No completions in example with prompt:\n{prompt[:200]!r}\nSkipping.")
            continue

        if len(completions) != len(labels):
            print(
                f"[ERROR] Example has {len(completions)} completions but {len(labels)} labels "
                f"for prompt:\n{prompt[:200]!r}\nSkipping."
            )
            continue

        # Cleaning function as requested
        def clean_text(t):
            if not t: return ""
            return t.replace("\\think", "").replace(r"\think", "").strip()

        # Clean the prompt
        cleaned_prompt = clean_text(ex.get("prompt", ""))
        
        # Robust extraction: "match only the prompt itself"
        # We look for the user message content specifically, ignoring system prompts.
        user_msg = cleaned_prompt
        s_marker = "<|im_start|>user"
        e_marker = "<|im_end|>"
        
        s_idx = cleaned_prompt.find(s_marker)
        if s_idx != -1:
            content_start = s_idx + len(s_marker)
            e_idx = cleaned_prompt.find(e_marker, content_start)
            if e_idx != -1:
                user_msg = cleaned_prompt[content_start:e_idx].strip()
        
        # Use default system prompt since we are ignoring the input's system prompt
        from rely.utils import MMLU_SYSTEM_PROMPT
        system_msg = MMLU_SYSTEM_PROMPT

        # Prepare the formatted assistant content
        # preserving alignment: map clean_text but do NOT filter out empty
        clean_steps = [clean_text(c) for c in completions]
        formatted_content = _format_steps_with_separator(clean_steps, evaluate_n_steps=1)
        
        # METHOD CHANGE: apply_chat_template with the assistant message included seems to be 
        # modifying/stripping our content (e.g. seeing multiple <extra_0> as artifacts to clean?).
        # Instead, we will format [System, User] -> generate prompt -> append our exact content.
        
        messages = [
            {"role": "system", "content": system_msg}, 
            {"role": "user", "content": user_msg}, 
        ]
        
        # usage: generate the "header" for the assistant response
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Append our carefully constructed content directly
        text += formatted_content
        
        # Final sanity check (should always pass now unless formatted_content itself is wrong)
        expected_separators = formatted_content.count("<extra_0>")
        actual_separators = text.count("<extra_0>")

        if actual_separators < expected_separators:
             print(f"[WARN] Separator mismatch! Expected {expected_separators}, got {actual_separators} in text.")
             print(f"[WARN] Formatted content end: {formatted_content[-50:]!r}")
             print(f"[WARN] Final text end: {text[-50:]!r}")

        batch_texts.append(text)
        batch_meta.append((prompt, completions, labels))

        if len(batch_texts) >= batch_size:
            flush_batch()

    flush_batch()
    return results_by_prompt


def to_python_list(dataset_obj):
    """
    Make sure we always deal with a plain Python list of dicts,
    regardless of whether load_dataset returned an HF Dataset or
    already a list.
    """
    if isinstance(dataset_obj, list):
        return dataset_obj
    try:
        # HuggingFace Dataset has .to_list()
        return dataset_obj.to_list()
    except AttributeError:
        # Fallback: iterate
        return [x for x in dataset_obj]


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading PUM model from {args.pum_model_ckpt}")
    tokenizer, model = load_pum_model_and_tokenizer(args.pum_model_ckpt, device)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print(f"[INFO] Loading GT dataset from {args.gt_dataset}")
    gt_raw = load_dataset(args.gt_dataset, subset="variance", split=args.gt_split)
    gt_examples = to_python_list(gt_raw)
    print(f"[INFO] GT dataset size: {len(gt_examples)}")

    print(f"[INFO] Loading SBS dataset from {args.sbs_dataset}")
    sbs_raw = load_dataset(args.sbs_dataset)
    sbs_examples = to_python_list(sbs_raw)
    print(f"[INFO] SBS dataset size: {len(sbs_examples)}")

    # ------------------------------------------------------------------
    # Run PUM separately on GT and SBS
    # ------------------------------------------------------------------
    print("[INFO] Scoring GT dataset with PUM...")
    gt_by_prompt = pum_scores_for_examples(
        gt_examples, tokenizer, model, device, args.max_length, args.batch_size
    )

    print("[INFO] Scoring SBS dataset with PUM...")
    sbs_by_prompt = pum_scores_for_examples(
        sbs_examples, tokenizer, model, device, args.max_length, args.batch_size
    )

    # ------------------------------------------------------------------
    # Build combined info.json paired by prompt
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Build combined info.json paired by prompt
    # ------------------------------------------------------------------
    print("[INFO] Building combined info.json...")
    print(f"[INFO] Scored entries - GT: {len(gt_by_prompt)}, SBS: {len(sbs_by_prompt)}")

    # Normalize keys for robust matching
    def normalize_key(p):
        # Maybe clean more aggressively if needed
        return p.strip()

    gt_map = {normalize_key(p): p for p in gt_by_prompt}
    sbs_map = {normalize_key(p): p for p in sbs_by_prompt}
    
    print(f"[INFO] Map sizes after normalization - GT: {len(gt_map)}, SBS: {len(sbs_map)}")

    all_norm_keys = set(gt_map.keys()) | set(sbs_map.keys())
    results = []

    missing_gt = 0
    missing_sbs = 0
    mismatched = 0
    
    missing_gt_examples = []
    missing_sbs_examples = []

    for norm_k in sorted(list(all_norm_keys)):
        original_gt_prompt = gt_map.get(norm_k)
        original_sbs_prompt = sbs_map.get(norm_k)

        if original_gt_prompt is None:
            missing_gt += 1
            if len(missing_gt_examples) < 3:
                missing_gt_examples.append(norm_k)
            continue
        if original_sbs_prompt is None:
            missing_sbs += 1
            if len(missing_sbs_examples) < 3:
                missing_sbs_examples.append(norm_k)
            continue
            
        gt_entry = gt_by_prompt[original_gt_prompt]
        sbs_entry = sbs_by_prompt[original_sbs_prompt]

        # Use the GT prompt as the canonical one for output
        prompt = original_gt_prompt

        # We do NOT require same number of completions between GT and SBS.
        # We just require that within each side, len(labels) == len(pum_scores),
        # which is already enforced in pum_scores_for_examples.

        results.append(
            {
                "prompt": prompt,
                "gt": {
                    "completions": gt_entry["completions"],
                    "labels": gt_entry["labels"],
                    "pum_scores": gt_entry["pum_scores"],
                },
                "sbs": {
                    "completions": sbs_entry["completions"],
                    "labels": sbs_entry["labels"],
                    "pum_scores": sbs_entry["pum_scores"],
                },
            }
        )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote {len(results)} paired entries to {args.output_path}")
    if missing_gt or missing_sbs or mismatched:
        print(
            f"[INFO] Skipped due to issues -> "
            f"missing_gt: {missing_gt}, missing_sbs: {missing_sbs}, mismatched: {mismatched}"
        )
        if missing_gt_examples:
            print("[INFO] Examples present in SBS but MISSING in GT:")
            for e in missing_gt_examples:
                print(f"   - {e[:100]!r}...")
        if missing_sbs_examples:
            print("[INFO] Examples present in GT but MISSING in SBS:")
            for e in missing_sbs_examples:
                print(f"   - {e[:100]!r}...")


if __name__ == "__main__":
    main()
