"""
Create fork points based on entropy thresholds.

This module provides functions for creating fork points from model outputs
based on entropy thresholds and extracting corresponding activations.
"""

import torch
import json
import os
import random
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from unsloth import FastLanguageModel


SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about general knowledge. Think step by step and then finish your answer with 'The correct answer is (X)' where X is the correct letter choice.

EXAMPLE

Question: The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the other, the detector will detect photons
Options:
(A) an average of 10 times, with an rms deviation of about 4
(B) an average of 10 times, with an rms deviation of about 3
(C) an average of 10 times, with an rms deviation of about 1
(D) an average of 10 times, with an rms deviation of about 0.1

## Your Example Answer
[...] The correct answer is (B).
"""


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy from logits.
    
    Args:
        logits: Model logits
        
    Returns:
        Entropy values
    """
    probs = torch.softmax(logits, dim=-1)
    probs_clamped = torch.clamp(probs, min=1e-9)
    entropy = -torch.sum(probs * torch.log(probs_clamped), dim=-1)
    return entropy


def create_forks_from_dataset(
    input_file: str,
    output_file: str,
    entropy_threshold: float = 0.5195,
    n_forks_per_item: int = 10,
    use_top_entropy: bool = True,
    model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    gpu_id: int = 0,
    dataset_percentage: float = 100.0
) -> None:
    """
    Create fork points from a dataset based on entropy thresholds.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output PT file
        entropy_threshold: Entropy threshold for forking
        n_forks_per_item: Number of forks to create per item
        use_top_entropy: Whether to use top entropy points or random sampling
        model_name: Name of the model to use
        gpu_id: GPU device ID to use
        dataset_percentage: Percentage of dataset to process (0.0-100.0)
    """
    print(f"Loading model and tokenizer on GPU {gpu_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=30_000,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    # Load dataset
    with open(input_file, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Shuffle the dataset first
    random.shuffle(dataset)
    
    # Apply dataset percentage if less than 100%
    if dataset_percentage < 100.0:
        num_items_to_process = int(len(dataset) * dataset_percentage / 100.0)
        dataset = dataset[:num_items_to_process]
        print(f"Processing {dataset_percentage}% of dataset: {num_items_to_process} items out of {len(dataset)}")
    
    output_data = []
    
    print(f"Processing {len(dataset)} items (GPU {gpu_id})...")
    for idx, item in enumerate(tqdm(dataset, desc=f"GPU {gpu_id}")):
        question = item["question"]
        attempt = item["attempt"]
        solution = item["solution"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<think>\n"+attempt}
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(f"cuda:{gpu_id}")
        
        assistant_prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt")
        assistant_start_index = assistant_prompt.shape[1]

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        second_to_last_layer_activations = hidden_states[-2].squeeze(0)
        
        forking_points = []
        for i in range(assistant_start_index, input_ids.shape[1]):
            token_logits = logits[0, i - 1, :]
            entropy = calculate_entropy(token_logits)

            if entropy.item() > entropy_threshold:
                forking_points.append(i)

        if not forking_points:
            continue

        num_samples = min(len(forking_points), n_forks_per_item)
        if use_top_entropy:
            # Compute entropies for all forking points
            entropies = []
            for fork_token_index in forking_points:
                token_logits = logits[0, fork_token_index - 1, :]
                entropy = calculate_entropy(token_logits)
                entropies.append((entropy.item(), fork_token_index))
            # Sort by entropy descending and pick top N
            entropies.sort(reverse=True)
            selected_indices = [idx for _, idx in entropies[:num_samples]]
        else:
            selected_indices = random.sample(forking_points, num_samples)

        for fork_token_index in selected_indices:
            cut_token_ids = input_ids[0, :fork_token_index]
            
            assistant_token_ids = cut_token_ids[assistant_start_index:]
            cut_cot_text = tokenizer.decode(assistant_token_ids, skip_special_tokens=True)

            activation_tensor = second_to_last_layer_activations[fork_token_index - 1].cpu()

            new_item = {
                "question": question,
                "cut_cot": cut_cot_text,
                "cut_cot_activations": activation_tensor,
                "solution": solution
            }
            output_data.append(new_item)

    torch.save(output_data, output_file)
    print(f"GPU {gpu_id}: Processing complete. Generated {len(output_data)} new data points.")
    print(f"Dataset saved to: {output_file}")


def process_dataset_chunk(
    start_idx: int,
    end_idx: int,
    output_path: str,
    gpu_id: int,
    input_file: str = "generations-mmlu-qwen3-1.7B.jsonl",
    entropy_threshold: float = 0.5195,
    n_forks_per_item: int = 10,
    use_top_entropy: bool = True,
    model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    dataset_percentage: float = 100.0
) -> None:
    """
    Process a chunk of the dataset for forking analysis.
    
    Args:
        start_idx: Starting index for dataset chunk
        end_idx: Ending index for dataset chunk
        output_path: Output file path
        gpu_id: GPU device ID to use
        input_file: Input JSONL file path
        entropy_threshold: Entropy threshold for forking
        n_forks_per_item: Number of forks to create per item
        use_top_entropy: Whether to use top entropy points or random sampling
        model_name: Name of the model to use
        dataset_percentage: Percentage of the dataset chunk to process (0.0-100.0)
    """
    # Validate dataset percentage
    if dataset_percentage < 0.0 or dataset_percentage > 100.0:
        raise ValueError("dataset_percentage must be between 0.0 and 100.0")
    
    # Set the CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"Loading model and tokenizer on GPU {gpu_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=30_000,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    with open(input_file, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Shuffle the dataset first
    random.shuffle(dataset)
    
    # Select the chunk of data to process
    dataset_chunk = dataset[start_idx:end_idx]
    
    # Apply dataset percentage if less than 100%
    if dataset_percentage < 100.0:
        num_items_to_process = int(len(dataset_chunk) * dataset_percentage / 100.0)
        dataset_chunk = dataset_chunk[:num_items_to_process]
        print(f"Processing {dataset_percentage}% of dataset chunk: {num_items_to_process} items out of {len(dataset[start_idx:end_idx])}")
    
    output_data = []
    
    print(f"Processing items {start_idx} to {start_idx + len(dataset_chunk) - 1} (GPU {gpu_id})...")
    for idx, item in enumerate(tqdm(dataset_chunk, desc=f"GPU {gpu_id}")):
        question = item["question"]
        attempt = item["attempt"]
        solution = item["solution"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<think>\n"+attempt}
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(f"cuda:{gpu_id}")
        
        assistant_prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt")
        assistant_start_index = assistant_prompt.shape[1]

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        second_to_last_layer_activations = hidden_states[-2].squeeze(0)
        
        forking_points = []
        for i in range(assistant_start_index, input_ids.shape[1]):
            token_logits = logits[0, i - 1, :]
            entropy = calculate_entropy(token_logits)

            if entropy.item() > entropy_threshold:
                forking_points.append(i)

        if not forking_points:
            continue

        num_samples = min(len(forking_points), n_forks_per_item)
        if use_top_entropy:
            # Compute entropies for all forking points
            entropies = []
            for fork_token_index in forking_points:
                token_logits = logits[0, fork_token_index - 1, :]
                entropy = calculate_entropy(token_logits)
                entropies.append((entropy.item(), fork_token_index))
            # Sort by entropy descending and pick top N
            entropies.sort(reverse=True)
            selected_indices = [idx for _, idx in entropies[:num_samples]]
        else:
            selected_indices = random.sample(forking_points, num_samples)

        for fork_token_index in selected_indices:
            cut_token_ids = input_ids[0, :fork_token_index]
            
            assistant_token_ids = cut_token_ids[assistant_start_index:]
            cut_cot_text = tokenizer.decode(assistant_token_ids, skip_special_tokens=True)

            activation_tensor = second_to_last_layer_activations[fork_token_index - 1].cpu()

            new_item = {
                "question": question,
                "cut_cot": cut_cot_text,
                "cut_cot_activations": activation_tensor,
                "solution": solution
            }
            output_data.append(new_item)

    torch.save(output_data, output_path)
    print(f"GPU {gpu_id}: Processing complete. Generated {len(output_data)} new data points.")
    print(f"Dataset saved to: {output_path}")


def main():
    """Command-line interface for process_dataset_chunk."""
    parser = argparse.ArgumentParser(description='Process dataset chunks for forking analysis')
    parser.add_argument('--start_idx', type=int, required=True, help='Starting index for dataset chunk')
    parser.add_argument('--end_idx', type=int, required=True, help='Ending index for dataset chunk')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--input_file', type=str, default="generations-mmlu-qwen3-1.7B.jsonl", help='Input JSONL file path')
    parser.add_argument('--entropy_threshold', type=float, default=0.5195, help='Entropy threshold for forking')
    parser.add_argument('--n_forks_per_item', type=int, default=10, help='Number of forks per item')
    parser.add_argument('--use_top_entropy', action='store_true', help='Use top entropy points instead of random sampling')
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit", help='Model name to use')
    parser.add_argument('--dataset_percentage', type=float, default=100.0, 
                       help='Percentage of the dataset chunk to process (0.0-100.0, default: 100.0)')
    
    args = parser.parse_args()
    
    process_dataset_chunk(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        output_path=args.output_path,
        gpu_id=args.gpu_id,
        input_file=args.input_file,
        entropy_threshold=args.entropy_threshold,
        n_forks_per_item=args.n_forks_per_item,
        use_top_entropy=args.use_top_entropy,
        model_name=args.model_name,
        dataset_percentage=args.dataset_percentage
    )


if __name__ == "__main__":
    main() 