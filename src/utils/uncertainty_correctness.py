#!/usr/bin/env python
# plot_correlation.py

"""
Plots the correlation between PUM trace uncertainty (using 'max' aggregation)
and the correctness of individual traces.
"""

import os
import json
import argparse
import re
from collections import Counter

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from rely.train import RegressionPRMModel
# --- PUM Model Loading (from inspiration script) ---

def load_pum_model(model_path):
    """Loads the PUM model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Use the SOFT PRM (RegressionPRMModel) as in the inspiration script
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
    (Adapted from inspiration script, removed debug prints)
    """
    conversation_strs = []
    # This ID is for the '<extra_0>' token
    step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]

    # Use a fixed system prompt (as in the inspiration script)
    MATH_SYSTEM_PROMPT = "Follow the user's instructions to solve the math problem, thinking step by step."

    for solution in solutions:
        assistant_response = solution['solution_path']
        steps = [s.strip() for s in assistant_response.split("\n\n") if s.strip()]
        if not steps:
            steps = [""] # Ensure at least one step
        
        # Format with <extra_0> separators
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
    batch_size = 16 # Adjusted batch size for efficiency

    for i in range(0, len(conversation_strs), batch_size):
        batch_conversations = conversation_strs[i:i + batch_size]
        try:
            inputs = tokenizer(batch_conversations, return_tensors="pt", padding=True, truncation=True, max_length=7000).to(device)
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

            # SOFT PRM (RegressionPRMModel) output
            uncertainty_probs = outputs.logits.squeeze(-1) # Shape (batch_size, seq_len)
            
            # Mask for separator tokens
            token_masks = (inputs.input_ids == step_sep_id)
            
            has_separator = token_masks.any(dim=1)
            # Use a default uncertainty (e.g., 0.5) if no separators are found
            default_uncertainty = torch.tensor(0.5, device=device) 
            
            if aggregation_method == "product":
                # Apply mask: probabilities at separator tokens, 1.0 elsewhere
                masked_probs = torch.where(token_masks, uncertainty_probs, torch.tensor(1.0, device=device))
                calculated_uncertainties = torch.prod(masked_probs, dim=1)
            elif aggregation_method == "max":
                # Apply mask: probabilities at separator tokens, 0.0 elsewhere
                masked_probs = torch.where(token_masks, uncertainty_probs, torch.tensor(0.0, device=device))
                calculated_uncertainties = torch.max(masked_probs, dim=1)[0]
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            # If a sequence had no separators, its 'max' would be 0 or 'product' 1.
            # Replace these with the default uncertainty.
            final_uncertainties = torch.where(has_separator, calculated_uncertainties, default_uncertainty)
            all_uncertainties.extend(final_uncertainties.cpu().tolist())

            del inputs, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Error during batch uncertainty calculation: {e}")
            all_uncertainties.extend([0.5] * len(batch_conversations)) # Default on error

    return all_uncertainties

# --- New Logic for Correlation ---

def normalize_answer(s):
    """
    Basic normalization for numerical answers.
    Copied from rely.utils or similar.
    """
    s = str(s).strip()
    s = s.replace(",", "")
    # Find the last number (integer or decimal) in the string
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if matches:
        return matches[-1]
    return s # Return original string if no number is found

def process_files_for_correlation(json_files, pum_model, pum_tokenizer, pum_device):
    """
    Processes all json files to extract trace uncertainty and correctness, grouped by problem.
    Returns problem-level data to enable proper train/test splitting.
    """
    problems_data = []  # List of dictionaries, one per problem
    
    print("Collecting uncertainty and correctness data...")
    for path in tqdm(json_files, desc="Processing files"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ground_truth = data.get('ground_truth')
            if ground_truth is None:
                continue
            normalized_gt = normalize_answer(str(ground_truth))
            if not normalized_gt:
                continue
                
            solutions = data.get('solutions', [])
            question = data.get('question')
            if not solutions or not question:
                continue
                
            # Get uncertainties for all traces in this file using MAX aggregation
            uncertainties = get_trace_uncertainties(
                pum_model, pum_tokenizer, pum_device, 
                question, solutions, aggregation_method="max" # <-- Hard-coded 'max'
            )
            
            if len(uncertainties) != len(solutions):
                print(f"Warning: Mismatch in solutions ({len(solutions)}) and uncertainties ({len(uncertainties)}) for {path}. Skipping.")
                continue
                
            # Process all traces for this problem
            problem_uncertainties = []
            problem_correctness = []
            
            for sol, unc in zip(solutions, uncertainties):
                final_ans = sol.get('final_answer')
                
                if final_ans is None or final_ans == "Not found" or str(final_ans).strip() == "":
                    normalized_ans = "" # Treat as incorrect
                else:
                    normalized_ans = normalize_answer(str(final_ans))
                
                # 1.0 if correct, 0.0 if incorrect
                is_correct = 1.0 if (normalized_ans == normalized_gt and normalized_ans != "") else 0.0
                
                problem_uncertainties.append(unc)
                problem_correctness.append(is_correct)
            
            # Store problem-level data
            problems_data.append({
                'problem_id': path,  # Use file path as unique problem identifier
                'uncertainties': problem_uncertainties,
                'correctness': problem_correctness
            })
                
        except Exception as e:
            print(f"Warning: Failed to process file {path}: {e}")
    
    return problems_data

def train_and_evaluate_model(problems_data, output_path):
    """
    Trains a logistic regression model to predict correctness from uncertainty
    with proper problem-level train/test splitting to avoid data leakage.
    """
    if not problems_data:
        print("No valid data to train model.")
        return

    # Split problems into train/test sets (not individual traces)
    problem_ids = [p['problem_id'] for p in problems_data]
    
    # Calculate class balance for stratification (majority class per problem)
    problem_labels = []
    for p in problems_data:
        # Use majority vote of correctness for each problem for stratification
        majority_correct = sum(p['correctness']) > len(p['correctness']) / 2
        problem_labels.append(1 if majority_correct else 0)
    
    try:
        train_problems, test_problems, _, _ = train_test_split(
            problems_data, problem_labels, test_size=0.2, random_state=42, 
            stratify=problem_labels
        )
    except ValueError:
        # If stratification fails (e.g., too few samples), split without stratification
        print("Warning: Could not stratify by problem-level correctness. Using random split.")
        train_problems, test_problems = train_test_split(
            problems_data, test_size=0.2, random_state=42
        )
    
    # Flatten train and test data to individual traces
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for p in train_problems:
        X_train.extend(p['uncertainties'])
        y_train.extend(p['correctness'])
    
    for p in test_problems:
        X_test.extend(p['uncertainties'])
        y_test.extend(p['correctness'])
    
    # Convert to numpy arrays
    X_train = np.array(X_train).reshape(-1, 1)
    y_train = np.array(y_train)
    X_test = np.array(X_test).reshape(-1, 1)
    y_test = np.array(y_test)
    
    # Combine all data for correlation and visualization
    all_uncertainties = X_train.flatten().tolist() + X_test.flatten().tolist()
    all_correctness = y_train.tolist() + y_test.tolist()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total problems: {len(problems_data)}")
    print(f"   Train problems: {len(train_problems)}")
    print(f"   Test problems: {len(test_problems)}")
    print(f"   Total traces: {len(all_correctness)}")
    print(f"   Train traces: {len(y_train)}")
    print(f"   Test traces: {len(y_test)}")
    
    correct_count = int(sum(all_correctness))
    print(f"   Correct traces: {correct_count} ({(correct_count/len(all_correctness))*100:.1f}%)")
    print(f"   Incorrect traces: {len(all_correctness) - correct_count} ({((len(all_correctness) - correct_count)/len(all_correctness))*100:.1f}%)")
    
    # Calculate correlation for reference
    try:
        corr, p_value = pearsonr(all_uncertainties, all_correctness)
        print(f"   Point-Biserial Correlation: r = {corr:.4f} (p = {p_value:.2e})")
    except ValueError as e:
        print(f"   Could not calculate correlation: {e}")
    
    print(f"\n🔧 Training logistic regression model with problem-level splitting...")
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Model Performance (Problem-Level Split):")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   AUROC: {auroc:.4f}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))
    
    # Print model coefficients
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    print(f"\n🔍 Model Parameters:")
    print(f"   Coefficient (uncertainty): {coef:.4f}")
    print(f"   Intercept: {intercept:.4f}")
    print(f"   Interpretation: {'Higher' if coef > 0 else 'Lower'} uncertainty → {'Higher' if coef > 0 else 'Lower'} probability of correctness")
    
    # Create visualization
    plot_model_results(all_uncertainties, all_correctness, model, output_path)

def plot_model_results(uncertainties, correctness, model, output_path):
    """
    Creates a visualization of the model predictions and data distribution.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Distribution of uncertainties by correctness
    data = {
        'Uncertainty': uncertainties,
        'Correctness': ['Correct' if c == 1.0 else 'Incorrect' for c in correctness]
    }
    df = pd.DataFrame(data)
    
    sns.violinplot(x='Correctness', y='Uncertainty', data=df, 
                   palette={"Correct": "#5cb85c", "Incorrect": "#d9534f"}, 
                   inner='quartile', order=['Correct', 'Incorrect'], ax=ax1)
    ax1.set_title("Distribution of Uncertainty by Correctness")
    ax1.set_ylabel("Uncertainty Score")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Right plot: Logistic regression curve
    X_range = np.linspace(min(uncertainties), max(uncertainties), 100).reshape(-1, 1)
    y_pred_proba = model.predict_proba(X_range)[:, 1]
    
    ax2.scatter([u for u, c in zip(uncertainties, correctness) if c == 0], 
               [0] * sum(1 for c in correctness if c == 0), 
               alpha=0.6, color='#d9534f', label='Incorrect', s=20)
    ax2.scatter([u for u, c in zip(uncertainties, correctness) if c == 1], 
               [1] * sum(1 for c in correctness if c == 1), 
               alpha=0.6, color='#5cb85c', label='Correct', s=20)
    ax2.plot(X_range.flatten(), y_pred_proba, 'b-', linewidth=2, label='Logistic Regression')
    ax2.set_xlabel("Uncertainty Score")
    ax2.set_ylabel("Probability of Correctness")
    ax2.set_title("Logistic Regression Model")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Model visualization saved to: {output_path}")

# --- Main execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot correlation between trace uncertainty (max method) and correctness.')
    parser.add_argument('input_dir', 
                        help='Directory containing result subfolders (with summary.json files).')
    parser.add_argument('--uncertainty_model_path', type=str, required=True, 
                        help='Path to the PUM uncertainty model (RegressionPRMModel).')
    parser.add_argument('--output_path', type=str, 
                        default='assets/figures/corr_uncertainty_correctness.png',
                        help='Path to save the output correlation plot.')
    
    args = parser.parse_args()

    # --- 1. Load PUM model ---
    if not args.uncertainty_model_path or not os.path.isdir(args.uncertainty_model_path):
         print(f"❌ Error: The uncertainty model path does not exist or is not a directory: '{args.uncertainty_model_path}'")
         exit(1)
    pum_model, pum_tokenizer, pum_device = load_pum_model(args.uncertainty_model_path)

    # --- 2. Find summary files ---
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: The input directory does not exist: '{args.input_dir}'")
        exit(1)
    json_files = [os.path.join(root, f) for root, _, files in os.walk(args.input_dir) for f in files if f == 'summary.json']
    
    if not json_files:
        print(f"❌ No 'summary.json' files found in '{args.input_dir}' or its subfolders.")
        exit(1)
    
    print(f"Found {len(json_files)} 'summary.json' files.")
    
    # --- 3. Process files ---
    problems_data = process_files_for_correlation(
        json_files, pum_model, pum_tokenizer, pum_device
    )
    
    # --- 4. Train and evaluate model ---
    if not problems_data:
        print("❌ No valid traces were processed. Cannot train model.")
    else:
        train_and_evaluate_model(problems_data, args.output_path)