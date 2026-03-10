import argparse
import json
import os
import random

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def collate_fn(batch, tokenizer, separator='<extra_0>'):
    """
    Collates data, applying the chat template to the prompt and preparing for step-wise evaluation.
    """
    input_ids_list = []
    labels_list = []
    score_ids_list = []
    separator_ids = tokenizer.encode(separator, add_special_tokens=False, return_tensors='pt')

    for item in batch:
        # 1. Apply chat template for the user prompt
        messages = [{"role": "user", "content": item['prompt']}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        score_ids_for_item = []
        # Sequentially append completions and separators, tracking separator positions
        for completion in item['completions']:
            completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors='pt')['input_ids']
            # Ensure all tensors have the same dtype (long) and device before concatenation
            completion_ids = completion_ids.to(dtype=torch.long, device=prompt_ids.device)
            separator_ids_device = separator_ids.to(dtype=torch.long, device=prompt_ids.device)
            prompt_ids = torch.cat([prompt_ids, completion_ids, separator_ids_device], dim=-1)
            # The score is taken at the last token of the separator (assuming a single-token separator)
            score_ids_for_item.append(prompt_ids.size(-1) - 1)
        
        input_ids_list.append(prompt_ids)
        score_ids_list.append(score_ids_for_item)
        
        # 2. Process the 'labels' field (list of bools) and convert to integers
        labels_list.append([int(label) for label in item['labels']])
    
    # Right-pad the input_ids to the max length in the batch
    pad_token_id = tokenizer.pad_token_id
    max_len = max(ids.size(-1) for ids in input_ids_list)
    
    padded_input_ids = []
    for ids in input_ids_list:
        padding_len = max_len - ids.size(-1)
        padding = torch.full((padding_len,), pad_token_id, dtype=torch.long, device=ids.device)
        padded_input_ids.append(torch.cat([ids.squeeze(0), padding]))
        
    input_ids = torch.stack(padded_input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels_list, # This is a ragged list of ints
        "score_ids": score_ids_list # This is a ragged list of indices
    }
    

def gather_objects(data, accelerator):
    """
    Gathers a list of objects from all processes.
    """
    if accelerator.num_processes == 1:
        return data
        
    output_objects = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(output_objects, data)
    
    # Flatten the list of lists
    return [item for sublist in output_objects for item in sublist]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    bs = args.batch_size
    num_of_workers = args.num_of_workers
    separator = args.separator
    model_path = args.model
    subset = args.subset

    model_name = os.path.basename(model_path)
    save_dir = f'outputs/{model_name}'
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator()
    print(f'Loading model from {model_path}')
    # This script assumes a token classification head, which is common for Process Reward Models (PRMs)
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = accelerator.prepare(model)
    model.eval()

    # Load the dataset
    dataset = load_dataset('jacopo-minniti/MATH-PUM-qwen2.5-1.5B', subset, split="test")
    dataset = dataset.shuffle(seed=42).select(range(2000))

    sampler = None
    if accelerator.distributed_type == "MULTI_GPU":
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
        )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=bs, 
        collate_fn=lambda x: collate_fn(x, tokenizer, separator), 
        num_workers=num_of_workers,
        sampler=sampler,
        drop_last=False,
    )

    all_labels = []
    all_predictions = []
    all_scores = []

    for batch in tqdm(dataloader, disable=not accelerator.is_main_process, desc="Evaluating"):
        input_ids = batch['input_ids'].to(accelerator.device)
        labels = batch['labels']
        score_ids = batch['score_ids']
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        for i, score_id_list in enumerate(score_ids):
            if not score_id_list:
                continue
            
            true_labels = labels[i]
            step_logits = logits[i, score_id_list]
            
            # Predicted classes (e.g., 0 for incorrect, 1 for correct)
            pred_steps = torch.argmax(step_logits, dim=-1)
            
            # Probability scores for the positive class (class 1) for AUROC calculation
            scores = torch.softmax(step_logits, dim=-1)[:, 1]

            all_labels.extend(true_labels)
            all_predictions.extend(pred_steps.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())
    
    accelerator.wait_for_everyone()

    gathered_labels = gather_objects(all_labels, accelerator)
    gathered_predictions = gather_objects(all_predictions, accelerator)
    gathered_scores = gather_objects(all_scores, accelerator)

    if accelerator.is_main_process:
        if gathered_labels:
            accuracy = accuracy_score(gathered_labels, gathered_predictions)
            f1 = f1_score(gathered_labels, gathered_predictions)
            auroc = roc_auc_score(gathered_labels, gathered_scores)
            
            print(f'\n--- Evaluation Results for subset: {subset} ---')
            print(f'Total steps evaluated: {len(gathered_labels)}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'AUROC:    {auroc:.4f}')
            print('-------------------------------------------------')
        else:
            print("No data was gathered to evaluate.")

    if accelerator.distributed_type == "MULTI_GPU":
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a Process Reward Model with step-wise metrics.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-w", "--num_of_workers", type=int, default=4)
    parser.add_argument("-s", "--separator", type=str, default="<extra_0>", help="Separator token used between steps")
    parser.add_argument("--subset", type=str, required=True, help="Dataset subset to evaluate on (e.g., 'validation')")
    args = parser.parse_args()

    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)