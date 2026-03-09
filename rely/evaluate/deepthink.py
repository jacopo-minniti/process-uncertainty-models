# rely/evaluate/score_self_consistency.py

import os
import json
import argparse
from collections import Counter
import math
import re

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
import matplotlib.pyplot as plt

from rely.utils import normalize_answer, MATH_SYSTEM_PROMPT, prompt_pattern
from rely.train import RegressionPRMModel

# --- PUM Model Loading and Uncertainty Calculation ---

def load_pum_model(model_path):
    """Loads the PUM model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # PRM
    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    # )
    # SOFT PRM
    model = RegressionPRMModel.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = model.to(dtype=torch.bfloat16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    
    print("PUM uncertainty model loaded.")
    return model, tokenizer, device

@torch.no_grad()
def get_trace_uncertainties(model, tokenizer, device, question, solutions, aggregation_method="product"):
    """
    Calculates the uncertainty for a batch of solution traces.
    
    Args:
        aggregation_method: Either "product" or "max" to determine how to aggregate uncertainty scores
    """
    conversation_strs = []
    step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]

    for solution in solutions:
        assistant_response = solution['solution_path']
        steps = [s.strip() for s in assistant_response.split("\n\n") if s.strip()]
        # Ensure there's at least one step for the <extra_0> token
        if not steps:
            steps = [""]
        
        formatted_content = "<extra_0>".join(steps) + "<extra_0>"
        
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": formatted_content}
        ]
        conv_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        conversation_strs.append(conv_str)

    if not conversation_strs:
        return []

    all_uncertainties = []
    batch_size = 8 

    for i in range(0, len(conversation_strs), batch_size):
        batch_conversations = conversation_strs[i:i + batch_size]
        try:
            inputs = tokenizer(batch_conversations, return_tensors="pt", padding=True, truncation=True, max_length=7000).to(device)
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

            # PRM
            # uncertainty_probs = torch.softmax(outputs.logits, dim=-1)[:, :, 1]
            # SOFT PRM
            uncertainty_probs = outputs.logits
            
            print(f"DEBUG: Batch {i//batch_size + 1} - uncertainty_probs shape: {uncertainty_probs.shape}")
            print(f"DEBUG: uncertainty_probs min/max: {uncertainty_probs.min().item():.4f}/{uncertainty_probs.max().item():.4f}")
            
            token_masks = (inputs.input_ids == step_sep_id)
            
            print(f"DEBUG: step_sep_id: {step_sep_id}")
            print(f"DEBUG: token_masks shape: {token_masks.shape}")
            print(f"DEBUG: total separator tokens found: {token_masks.sum().item()}")
            for seq_idx in range(token_masks.shape[0]):
                sep_count = token_masks[seq_idx].sum().item()
                print(f"DEBUG: Sequence {seq_idx}: {sep_count} separator tokens")
            
            has_separator = token_masks.any(dim=1)
            default_uncertainty = torch.tensor(0.5, device=device)
            
            # Calculate uncertainty based on aggregation method
            if aggregation_method == "product":
                masked_probs = uncertainty_probs * token_masks.float() + (1 - token_masks.float())
                calculated_uncertainties = torch.prod(masked_probs, dim=1)
                print(f"DEBUG: Product aggregation - calculated_uncertainties: {calculated_uncertainties.cpu().tolist()}")
            elif aggregation_method == "max":
                masked_probs = uncertainty_probs * token_masks.float()
                calculated_uncertainties = torch.max(masked_probs, dim=1)[0]
                print(f"DEBUG: Max aggregation - calculated_uncertainties: {calculated_uncertainties.cpu().tolist()}")
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            final_uncertainties = torch.where(has_separator, calculated_uncertainties, default_uncertainty)
            print(f"DEBUG: Final uncertainties for batch: {final_uncertainties.cpu().tolist()}")
            all_uncertainties.extend(final_uncertainties.cpu().tolist())

            del inputs, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Error during batch uncertainty calculation: {e}")
            # Add default uncertainty for the failed batch
            all_uncertainties.extend([0.5] * len(batch_conversations))

    return all_uncertainties


# --- Evaluation Logic ---

def get_majority_vote(solutions):
    """Calculates majority vote from a list of solutions."""
    valid_answers = [s['final_answer'] for s in solutions if s.get('final_answer') and str(s['final_answer']).strip() and s['final_answer'] != "Not found"]
    if not valid_answers:
        return "N/A"
    
    normalized_answers = [normalize_answer(ans) for ans in valid_answers]
    normalized_answers = [ans for ans in normalized_answers if ans.strip()]
    
    if not normalized_answers:
        return "N/A"
        
    return Counter(normalized_answers).most_common(1)[0][0]

def get_confidence_weighted_majority_vote(solutions, confidences):
    """Calculates confidence-weighted majority vote from solutions and their confidence scores."""
    if len(solutions) != len(confidences):
        raise ValueError("Number of solutions must match number of confidence scores")
    
    # Filter valid answers and their corresponding confidences
    valid_data = []
    for sol, conf in zip(solutions, confidences):
        final_ans = sol.get('final_answer')
        if final_ans and str(final_ans).strip() and final_ans != "Not found":
            normalized_ans = normalize_answer(final_ans)
            if normalized_ans.strip():
                valid_data.append((normalized_ans, conf))
    
    if not valid_data:
        return "N/A"
    
    # Normalize the confidences so they sum to 1.0
    total_confidence = sum(conf for _, conf in valid_data)
    if total_confidence == 0:
        return "N/A"
    
    normalized_confidences = [(answer, conf / total_confidence) for answer, conf in valid_data]
    
    # Calculate weighted votes for each unique answer
    answer_weights = {}
    for answer, normalized_confidence in normalized_confidences:
        if answer in answer_weights:
            answer_weights[answer] += normalized_confidence
        else:
            answer_weights[answer] = normalized_confidence
    
    # Return answer with highest weighted vote
    if answer_weights:
        return max(answer_weights.items(), key=lambda x: x[1])[0]
    else:
        return "N/A"

def evaluate_all_percentages(pum_data, percentages):
    """Evaluate correctness for multiple top-k percentages."""
    if pum_data is None:
        return {p: {'pum_correct': None, 'pum_weighted_correct': None, 'bon_correct': None} for p in percentages}
    
    scored_solutions = pum_data['scored_solutions']
    confidences = pum_data['confidences']
    normalized_gt = pum_data['normalized_gt']
    
    results = {}
    
    for top_k_percent in percentages:
        # --- Top-k percent majority vote ---
        num_to_keep = math.ceil(len(scored_solutions) * (top_k_percent / 100.0))
        top_k_solutions = [sol for conf, sol in scored_solutions[:num_to_keep]]
        
        pum_majority_vote = get_majority_vote(top_k_solutions)
        is_pum_correct = False
        if pum_majority_vote != "N/A":
            normalized_pum_mv = normalize_answer(str(pum_majority_vote))
            if normalized_gt == normalized_pum_mv:
                is_pum_correct = True
        
        # --- Confidence weighted majority vote (all solutions) ---
        all_solutions = [sol for conf, sol in scored_solutions]
        pum_weighted_majority_vote = get_confidence_weighted_majority_vote(all_solutions, confidences)
        is_pum_weighted_correct = False
        if pum_weighted_majority_vote != "N/A":
            normalized_pum_weighted_mv = normalize_answer(str(pum_weighted_majority_vote))
            if normalized_gt == normalized_pum_weighted_mv:
                is_pum_weighted_correct = True
        
        # --- Best-of-N (BoN) score based on highest confidence ---
        is_bon_correct = False
        if scored_solutions:
            highest_conf_solution = scored_solutions[0][1]  # Solution with highest confidence
            bon_answer = highest_conf_solution.get('final_answer')
            
            if bon_answer and str(bon_answer).strip() and bon_answer != "Not found":
                normalized_bon_answer = normalize_answer(str(bon_answer))
                if normalized_gt == normalized_bon_answer:
                    is_bon_correct = True
        
        results[top_k_percent] = {
            'pum_correct': is_pum_correct,
            'pum_weighted_correct': is_pum_weighted_correct,
            'bon_correct': is_bon_correct
        }
    
    return results


def process_json_file(path, pum_model, pum_tokenizer, pum_device, use_pum, aggregation_method="product"):
    """
    Processes a single JSON file to check for correctness.
    Now returns data for multiple top-k percentages.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truth = data.get('ground_truth')
        if ground_truth is None:
            return None # Skip if no ground truth

        normalized_gt = normalize_answer(str(ground_truth))
        if not normalized_gt:
            return None # Skip if ground truth is empty

        solutions = data.get('solutions', [])
        if not solutions:
            return {'normal_correct': False, 'pum_data': None} # No solutions, so incorrect

        # --- 1. Normal Majority Vote ---
        majority_vote = get_majority_vote(solutions)
        is_normal_correct = False
        if majority_vote != "N/A":
            normalized_mv = normalize_answer(str(majority_vote))
            if normalized_gt == normalized_mv:
                is_normal_correct = True

        result = {'normal_correct': is_normal_correct}

        # --- 2. PUM Analysis ---
        if use_pum:
            if not pum_model or not pum_tokenizer:
                raise ValueError("PUM model and tokenizer must be loaded to use PUM-weighted scoring.")
            
            question = data.get('question')
            uncertainties = get_trace_uncertainties(pum_model, pum_tokenizer, pum_device, question, solutions, aggregation_method)
            
            confidences = [1.0 - u for u in uncertainties]
            
            # Pair solutions with their confidence scores and sort
            scored_solutions = sorted(zip(confidences, solutions), key=lambda x: x[0], reverse=True)
            
            # Store PUM data for later analysis
            result['pum_data'] = {
                'scored_solutions': scored_solutions,
                'confidences': confidences,
                'normalized_gt': normalized_gt
            }

        return result

    except Exception as e:
        print(f"Warning: Failed to process file {path}: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate self-consistency outputs with optional PUM-weighted majority voting.')
    parser.add_argument('input_dir', help='Directory containing result subfolders to process.')
    parser.add_argument('--use-pum', action='store_true', help='Enable PUM-weighted majority vote scoring.')
    parser.add_argument('--uncertainty_model_path', type=str, help='Path to the PUM uncertainty model (required if --use-pum).')
    parser.add_argument('--aggregation-method', type=str, default='product', choices=['product', 'max'], 
                       help='Method to aggregate uncertainty scores within a trace: "product" or "max".')
    
    args = parser.parse_args()

    if args.use_pum and not args.uncertainty_model_path:
        parser.error("--uncertainty_model_path is required when --use-pum is specified.")

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"❌ Error: The directory does not exist: '{input_dir}'")
        exit(1)

    # Define percentages to evaluate: 10%, 15%, 20%, ..., 100%
    percentages = list(range(10, 101, 5))
    
    # --- Load PUM model if needed ---
    pum_model, pum_tokenizer, pum_device = None, None, None
    if args.use_pum:
        pum_model, pum_tokenizer, pum_device = load_pum_model(args.uncertainty_model_path)

    # --- Find and process summary files ---
    json_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f == 'summary.json']

    if not json_files:
        print(f"❌ No 'summary.json' files found in '{input_dir}' or its subfolders.")
        exit(1)

    # --- Initialize counters for each percentage ---
    total_files = 0
    normal_correct_count = 0
    
    # Counters for each percentage
    pum_results = {p: {'correct': 0, 'weighted_correct': 0, 'bon_correct': 0} for p in percentages}

    for path in tqdm(sorted(json_files), desc="Evaluating summaries"):
        result = process_json_file(path, pum_model, pum_tokenizer, pum_device, args.use_pum, args.aggregation_method)
        
        if result is None:
            continue # Skip file if it had errors or was invalid
            
        total_files += 1
        
        if result['normal_correct']:
            normal_correct_count += 1
        
        # Evaluate all percentages if PUM is enabled
        if args.use_pum and result['pum_data'] is not None:
            percentage_results = evaluate_all_percentages(result['pum_data'], percentages)
            
            for p in percentages:
                if percentage_results[p]['pum_correct']:
                    pum_results[p]['correct'] += 1
                if percentage_results[p]['pum_weighted_correct']:
                    pum_results[p]['weighted_correct'] += 1
                if percentage_results[p]['bon_correct']:
                    pum_results[p]['bon_correct'] += 1

    # --- Calculate and Print Results ---
    normal_accuracy = (normal_correct_count / total_files) if total_files > 0 else 0.0
    
    # Extract N from directory name (e.g., "n_32")
    dir_name = os.path.basename(input_dir.rstrip('/'))
    n_samples = 0
    match = re.search(r'n_(\d+)', dir_name)
    if match:
        n_samples = int(match.group(1))

    print(f"\n--- Evaluation Summary ---")
    print(f"Directory: {dir_name}")
    print(f"Number of samples evaluated: {total_files}")
    
    result_summary = {
        "N": n_samples,
        "maj_accuracy": round(normal_accuracy, 4)
    }
    
    if args.use_pum and total_files > 0:
        # Calculate accuracies for each percentage
        pum_accuracies = {}
        weighted_accuracies = {}
        bon_accuracies = {}
        
        best_pum_percent = None
        best_pum_accuracy = 0.0
        
        for p in percentages:
            pum_acc = pum_results[p]['correct'] / total_files
            weighted_acc = pum_results[p]['weighted_correct'] / total_files
            bon_acc = pum_results[p]['bon_correct'] / total_files
            
            pum_accuracies[p] = pum_acc
            weighted_accuracies[p] = weighted_acc
            bon_accuracies[p] = bon_acc
            
            if pum_acc > best_pum_accuracy:
                best_pum_accuracy = pum_acc
                best_pum_percent = p
        
        print(f"\nBest PUM accuracy: {best_pum_accuracy:.4f} at top-{best_pum_percent}%")
        print(f"Baseline majority vote accuracy: {normal_accuracy:.4f}")
        
        # Create plots
        os.makedirs('/scratch/jacopo04/rely/assets/figures', exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(percentages, [pum_accuracies[p] for p in percentages], 'b-o', label='PUM Top-k% Majority Vote', linewidth=2, markersize=6)
        plt.plot(percentages, [weighted_accuracies[p] for p in percentages], 'g-s', label='PUM Weighted Majority Vote', linewidth=2, markersize=6)
        plt.plot(percentages, [bon_accuracies[p] for p in percentages], 'r-^', label='Best-of-N (Highest Confidence)', linewidth=2, markersize=6)
        plt.axhline(y=normal_accuracy, color='orange', linestyle='--', linewidth=2, label='Baseline Majority Vote')
        
        plt.xlabel('Top-k Percentage (%)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'PUM Evaluation Results - {dir_name} (N={n_samples})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(10, 100)
        plt.ylim(0, 1)
        
        # Add annotation for best performance
        if best_pum_percent is not None:
            plt.annotate(f'Best: {best_pum_accuracy:.3f} at {best_pum_percent}%', 
                        xy=(best_pum_percent, best_pum_accuracy), 
                        xytext=(best_pum_percent + 15, best_pum_accuracy + 0.05),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                        fontsize=10, color='blue')
        
        plt.tight_layout()
        
        plot_path = f'/scratch/jacopo04/rely/assets/figures/pum_evaluation_{dir_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        
        result_summary.update({
            "best_pum_percent": best_pum_percent,
            "best_pum_accuracy": round(best_pum_accuracy, 4),
            "pum_weighted_accuracy": round(weighted_accuracies[50], 4),  # Report 50% as representative
            "bon_accuracy": round(bon_accuracies[10], 4),  # Report 10% as representative for BoN
            "aggregation_method": args.aggregation_method,
            "pum_model_path": args.uncertainty_model_path
        })

    print("\nResults:")
    print(json.dumps(result_summary, indent=4))
