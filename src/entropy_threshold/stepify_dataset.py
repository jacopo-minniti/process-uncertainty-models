import os
os.environ["HF_HOME"] = "/scratch/jacopo04/rely/.cache"

# Hyperparameter for entropy threshold
ENTROPY_THRESHOLD = 1.8505859375

from rely.utils import load_dataset, save_dataset, MMLU_SYSTEM_PROMPT
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy

import json
import re
import torch
import numpy as np

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
print(f"Loaded {len(dataset)} items from qwen3 dataset")

def add_extra_tokens_batch(questions_and_cots, batch_size=32):
    """Add <extra_0> tokens after high entropy tokens at natural boundaries"""
    enriched_items = []
    
    for i in tqdm(range(0, len(questions_and_cots), batch_size), desc="Processing batches"):
        batch = questions_and_cots[i:i+batch_size]
        
        # Prepare batch data
        batch_texts = []
        batch_cot_starts = []
        batch_original_cots = []
        
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
            batch_cot_starts.append(len(tokenizer.encode(system_user_text)))
            batch_original_cots.append(cot_text)
        
        # Tokenize batch with padding
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=10_000).to(device)
        
        # Forward pass for entire batch
        with torch.no_grad():
            outputs = model(**inputs)
            # Calculate token entropies efficiently on GPU
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            token_entropies = -(log_probs.exp() * log_probs).sum(dim=-1)  # [batch_size, seq_len-1]
        
        # Process each item in the batch
        for batch_idx, (question, cot_text) in enumerate(batch):
            try:
                cot_start_idx = batch_cot_starts[batch_idx]
                attention_mask = inputs["attention_mask"][batch_idx]
                
                # Get valid sequence length (excluding padding)
                valid_length = attention_mask.sum().item()
                
                # Extract entropies for CoT tokens only
                cot_end_idx = min(valid_length - 1, token_entropies.shape[1])
                
                if cot_start_idx < cot_end_idx:
                    item_entropies = token_entropies[batch_idx, cot_start_idx:cot_end_idx].cpu().numpy()
                    
                    # Tokenize just the CoT to get token-to-text mapping
                    cot_tokens = tokenizer.encode(cot_text, add_special_tokens=False)
                    cot_token_strings = [tokenizer.decode([token]) for token in cot_tokens]
                    
                    # Find high entropy tokens and add <extra_0> at appropriate boundaries
                    enriched_cot = add_extra_tokens_to_text(cot_text, cot_token_strings, item_entropies)
                    enriched_items.append({"question": question, "attempt": enriched_cot})
                else:
                    # If we can't process, keep original
                    enriched_items.append({"question": question, "attempt": cot_text})
                
            except Exception as e:
                print(f"Error processing batch item {batch_idx}: {e}")
                # Keep original on error
                enriched_items.append({"question": batch[batch_idx][0], "attempt": batch[batch_idx][1]})
                continue
    
    return enriched_items

def add_extra_tokens_to_text(cot_text, token_strings, entropies):
    """Add <extra_0> tokens to text based on entropy thresholds"""
    if len(token_strings) != len(entropies):
        # Adjust for length mismatch (common with tokenization differences)
        min_len = min(len(token_strings), len(entropies))
        token_strings = token_strings[:min_len]
        entropies = entropies[:min_len]
    
    # Find high entropy token positions
    high_entropy_positions = []
    for i, entropy_val in enumerate(entropies):
        if entropy_val > ENTROPY_THRESHOLD:
            high_entropy_positions.append(i)
    
    if not high_entropy_positions:
        return cot_text
    
    # Group nearby high entropy tokens
    grouped_positions = []
    current_group = [high_entropy_positions[0]]
    
    for pos in high_entropy_positions[1:]:
        if pos - current_group[-1] <= 2:  # Within 2 tokens of each other
            current_group.append(pos)
        else:
            grouped_positions.append(current_group)
            current_group = [pos]
    grouped_positions.append(current_group)
    
    # For each group, find the next \n\n boundary and mark for <extra_0>
    insertion_points = set()
    
    for group in grouped_positions:
        # Start looking from the last token in the group
        start_pos = group[-1]
        
        # Reconstruct text up to this point to find character position
        char_pos = 0
        for i in range(start_pos + 1):
            if i < len(token_strings):
                char_pos += len(token_strings[i])
        
        # Find next \n\n after this character position
        remaining_text = cot_text[char_pos:]
        
        # Look for \n\n patterns
        match = re.search(r'\n\n', remaining_text)
        if match:
            # Position <extra_0> just before the \n\n
            boundary_pos = char_pos + match.start()
            insertion_points.add(boundary_pos)
    
    # Insert <extra_0> tokens at identified points (in reverse order to maintain positions)
    enriched_text = cot_text
    for pos in sorted(insertion_points, reverse=True):
        if pos < len(enriched_text):
            enriched_text = enriched_text[:pos] + "<extra_0>" + enriched_text[pos:]
    
    return enriched_text

# Prepare all question-CoT pairs
questions_and_cots = []
for item in dataset:
    question = item.get("question", "")
    attempt = item.get("attempt", "")
    
    if question and attempt:
        questions_and_cots.append((question, attempt))

print(f"Prepared {len(questions_and_cots)} question-CoT pairs for processing")

# Process all items with batching to add <extra_0> tokens
enriched_data = add_extra_tokens_batch(questions_and_cots, batch_size=2)  # Reduced batch size for safety

print(f"Processed {len(enriched_data)} items with <extra_0> tokens")

# Save enriched dataset in the same format as original
save_dataset(enriched_data, "data/mmlu/generations_r1_distil_separator_99.5.jsonl")

print(f"Enriched dataset saved to data/mmlu/generations_qwen3_separator.jsonl")
print(f"Using entropy threshold: {ENTROPY_THRESHOLD}")

# Show a sample of the enriched data
if enriched_data:
    print("\nSample enriched item:")
    sample_item = enriched_data[0]
    print(f"Question: {sample_item['question'][:100]}...")
    print(f"Original length: {len(questions_and_cots[0][1])}")
    print(f"Enriched length: {len(sample_item['attempt'])}")
    print(f"Number of <extra_0> tokens added: {sample_item['attempt'].count('<extra_0>')}")
    if "<extra_0>" in sample_item['attempt']:
        print("\nFirst few <extra_0> positions:")
        attempt_text = sample_item['attempt']
        positions = []
        start = 0
        for _ in range(min(3, attempt_text.count('<extra_0>'))):
            pos = attempt_text.find('<extra_0>', start)
            if pos != -1:
                context_start = max(0, pos - 50)
                context_end = min(len(attempt_text), pos + 60)
                context = attempt_text[context_start:context_end].replace('\n', '\\n')
                positions.append(f"...{context}...")
                start = pos + 1
        for i, context in enumerate(positions):
            print(f"{i+1}: {context}")
else:
    print("No data processed")

