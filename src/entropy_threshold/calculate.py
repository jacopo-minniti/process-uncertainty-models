import os
os.environ["HF_HOME"] = "/scratch/jacopo04/rely/.cache"

from rely.utils import load_dataset, MMLU_SYSTEM_PROMPT
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy
from tqdm import tqdm
import random

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Adjust model name as needed
print(f"Loading model: {model_name}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model = model.to(device)

# Load dataset
dataset = load_dataset("data/mmlu/generations_r1_distil.jsonl")
random.seed(42)
random.shuffle(dataset)
dataset = dataset[:int(len(dataset)/2)]
print(f"Loaded {len(dataset)} items from r1-qwen-distil dataset")

def calculate_token_entropies_batch(questions_and_cots, batch_size=32):
    """Calculate entropy for tokens in CoTs using batched processing"""
    all_entropies = []
    
    for i in tqdm(range(0, len(questions_and_cots), batch_size), desc="Processing batches"):
        batch = questions_and_cots[i:i+batch_size]
        
        # Prepare batch data
        batch_texts = []
        batch_cot_starts = []
        
        for question, cot_text in batch:
            # Create conversation
            messages = [
                {"role": "system", "content": MMLU_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": cot_text}
            ]
            
            # Get full text and CoT start position
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            system_user_text = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True
            )
            
            batch_texts.append(full_text)
            # We'll calculate exact token position after tokenization
            batch_cot_starts.append(len(tokenizer.encode(system_user_text)))
        
        # Tokenize batch with padding
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=10_000).to(device)
        
        # Forward pass for entire batch
        with torch.no_grad():
            outputs = model(**inputs)
            # Calculate token entropies efficiently on GPU
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            token_entropies = -(log_probs.exp() * log_probs).sum(dim=-1)  # [batch_size, seq_len-1]
        
        # Process each item in the batch
        for batch_idx, (_, cot_text) in enumerate(batch):
            try:
                cot_start_idx = batch_cot_starts[batch_idx]
                attention_mask = inputs["attention_mask"][batch_idx]
                
                # Get valid sequence length (excluding padding)
                valid_length = attention_mask.sum().item()
                
                # Extract entropies for CoT tokens only
                cot_end_idx = min(valid_length - 1, token_entropies.shape[1])
                if cot_start_idx < cot_end_idx:
                    item_entropies = token_entropies[batch_idx, cot_start_idx:cot_end_idx]
                    all_entropies.extend(item_entropies.cpu().numpy().tolist())
                
            except Exception as e:
                print(f"Error processing batch item {batch_idx}: {e}")
                continue
    
    return all_entropies

# Prepare all question-CoT pairs
questions_and_cots = []
for item in dataset:
    question = item.get("question", "")
    attempt = item.get("attempt", "")
    
    if question and attempt:
        questions_and_cots.append((question, attempt))

print(f"Prepared {len(questions_and_cots)} question-CoT pairs for processing")

# Process all items with batching
all_entropies = calculate_token_entropies_batch(questions_and_cots, batch_size=2)  # Reduced batch size for safety

print(f"Total token entropies collected: {len(all_entropies)}")

# Calculate percentiles
if all_entropies:
    percentiles = [50, 60, 70, 80, 90, 95, 96, 97, 98, 98.5, 99, 99.5, 99.9]
    entropy_percentiles = {}
    
    for p in percentiles:
        entropy_percentiles[f"p{p}"] = float(np.percentile(all_entropies, p))
    
    # Add some basic stats
    entropy_percentiles["mean"] = float(np.mean(all_entropies))
    entropy_percentiles["std"] = float(np.std(all_entropies))
    entropy_percentiles["min"] = float(np.min(all_entropies))
    entropy_percentiles["max"] = float(np.max(all_entropies))
    entropy_percentiles["total_tokens"] = len(all_entropies)
    entropy_percentiles["processed_items"] = len(questions_and_cots)
    
    # Save results
    with open("data/mmlu/entropy_percentiles_r1_distil.json", "w") as f:
        json.dump(entropy_percentiles, f, indent=2)
    
    print(f"Results saved to entropy_results.json")
    print(f"Percentiles: {entropy_percentiles}")
else:
    print("No entropies calculated")

