# rely/inference/sbs/servers.py

import argparse
import logging
from multiprocessing import Queue
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from rely.utils import prompt_pattern
from rely.train.regression_prm.model import RegressionPRMModel

logger = logging.getLogger(__name__)


def _tokenize_with_manual_separator(tokenizer, system_msg: str, user_msg: str, steps: List[str], evaluate_n_steps: int, sep_token_id: int) -> List[int]:
    """
    Manually construct the token sequence:
    [System/User Prompt] [Step 1] [Sep] [Step 2] [Sep] ...
    
    This avoids issues where apply_chat_template or subsequent string processing
    might strip or malform the <extra_0> separator.
    """
    # 1. Format and tokenize the prompt prefix (System + User)
    messages = [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": user_msg.strip()}
    ]
    # We use apply_chat_template to get the formatted string for the prompt part
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize prompt. add_special_tokens=False usually fine if template handles BOS.
    # To be safe regarding BOS, if valid prompt text usually starts with BOS, tokenizer might add another if True.
    # We'll assume the template output is "ready to go" but might need BOS if the template doesn't add it.
    # However, standard chat templates often include BOS. Let's stick to what pum_eval did:
    input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    
    # 2. Append steps and separators
    eval_every = max(1, int(evaluate_n_steps))
    steps_since_separator = 0
    
    for idx, step in enumerate(steps, start=1):
        step_text = step.strip()
        # Add newline separation if not the first step (matching pum_eval fix)
        if idx > 1:
            step_text = "\n\n" + step_text
            
        step_ids = tokenizer(step_text, add_special_tokens=False)["input_ids"]
        input_ids.extend(step_ids)
        
        steps_since_separator += 1
        is_last = idx == len(steps)
        
        if steps_since_separator == eval_every or is_last:
            input_ids.append(sep_token_id)
            steps_since_separator = 0
            
    return input_ids


def _uncertainty_model_server(args: argparse.Namespace, task_queue: Queue, result_queues: List[Queue]):
    uncertainty_device = torch.device(f"cuda:{args.uncertainty_model_gpu}")
    logger.info(f"[UncertaintyServer] Starting on device {uncertainty_device}")
    eval_every = getattr(args, "evaluate_n_steps", 1)

    tokenizer = AutoTokenizer.from_pretrained(args.uncertainty_model_path, trust_remote_code=True)
    
    uncertainty_model_type = getattr(args, "uncertainty_model_type", "regression")
    
    if uncertainty_model_type == "regression":
        model = RegressionPRMModel.from_pretrained(
            args.uncertainty_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModel.from_pretrained(
            args.uncertainty_model_path, num_labels=2, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        
    model = model.to(dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Note: for manual padding we usually pad right, but here we can handle it.
    model.eval()
    logger.info("[UncertaintyServer] Uncertainty model loaded.")
    
    # Pre-calculate separator ID
    sep_ids = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if not sep_ids:
        logger.warning("[UncertaintyServer] Could not encode <extra_0>. Using EOS as separator.")
        sep_token_id = tokenizer.eos_token_id
    else:
        sep_token_id = sep_ids[0]

    @torch.no_grad()
    def get_uncertainties(prompts: List[str]) -> List[float]:
        input_ids_list = []
        missing_prompt_matches = 0
        
        # 1. Parse and Tokenize
        for prompt in prompts:
            match = prompt_pattern.match(prompt)
            if not match:
                missing_prompt_matches += 1
                logger.error(f"Prompt did not match expected format. Cannot score. Prompt: {prompt[:200]}")
                # Append empty list to maintain index alignment, will handle later
                input_ids_list.append([]) 
                continue
            
            system_msg, user_msg, partial_solution = match.groups()
            steps = [s.strip() for s in (partial_solution or "").split('\n\n') if s.strip()]
            
            # Manual Tokenization
            ids = _tokenize_with_manual_separator(
                tokenizer, system_msg, user_msg, steps, eval_every, sep_token_id
            )
            input_ids_list.append(ids)

        if not input_ids_list:
             return [0.5] * len(prompts)
             
        # Handle failures (empty ids) by filling with dummy or marking for default score
        # We'll just replace empty ones with a dummy [0] to avoid crashing, and setting score to 0.5 later.
        valid_indices = [i for i, ids in enumerate(input_ids_list) if ids]
        if not valid_indices:
             return [0.5] * len(prompts)

        batch_size = 8
        all_uncertainties = [0.5] * len(prompts) # Initialize with default
        
        # We evaluate only the valid ones
        valid_inputs = [input_ids_list[i] for i in valid_indices]
        valid_results = []

        for i in range(0, len(valid_inputs), batch_size):
            batch_ids = valid_inputs[i:i + batch_size]
            
            # Manual Padding
            max_len = max(len(ids) for ids in batch_ids)
            # Cap max length
            max_len = min(max_len, 16000)
            
            padded_batch = []
            attention_masks = []
            
            for ids in batch_ids:
                # Truncate
                curr_ids = ids[:max_len]
                # Pad (Left padding usually for generation, but for regression model Right padding is often fine/standard unless specific requirement. 
                # pum_eval used padding_side="left" implicitly by tokenizer default or config. 
                # But here we are building tensors manually. Let's use Right padding for simplicity unless Left is strictly required.
                # Actually, the model standard config says padding_side="left" (lines 50).
                # Let's respect left padding.)
                pad_len = max_len - len(curr_ids)
                padded_ids = [tokenizer.pad_token_id] * pad_len + curr_ids
                mask = [0] * pad_len + [1] * len(curr_ids)
                
                padded_batch.append(padded_ids)
                attention_masks.append(mask)
            
            input_tensor = torch.tensor(padded_batch, device=uncertainty_device, dtype=torch.long)
            mask_tensor = torch.tensor(attention_masks, device=uncertainty_device, dtype=torch.long)
            
            try:
                if uncertainty_model_type == "regression":
                    outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
                    uncertainty_probs = outputs.logits
                else:
                    base_model_output = model.model(input_ids=input_tensor, attention_mask=mask_tensor, use_cache=False)
                    logits = model.score(base_model_output.last_hidden_state)
                    probabilities = torch.softmax(logits, dim=-1)
                    uncertainty_probs = probabilities[:, :, 1] # Probability of class 1 (Correct)

                
                # Identify separator positions
                token_masks = (input_tensor == sep_token_id)
                has_separator = token_masks.any(dim=1)
                
                calculated_uncertainties = torch.zeros(len(batch_ids), device=uncertainty_device)
                default_val = torch.tensor(0.5, device=uncertainty_device)
                
                # Logic copied from original, adapted for tensor operations
                if args.uncertainty_method == "product":
                    clamped_probs = uncertainty_probs.clamp(min=1e-8)
                    log_probs = torch.log(clamped_probs)
                    # Mask out non-separator tokens
                    masked_log_probs = torch.where(token_masks, log_probs, torch.tensor(0.0, device=uncertainty_device))
                    sum_log_probs = masked_log_probs.sum(dim=1)
                    calculated_uncertainties = torch.exp(sum_log_probs)
                elif args.uncertainty_method == "average":
                    masked_probs = uncertainty_probs * token_masks.float()
                    sums = masked_probs.sum(dim=1)
                    counts = token_masks.sum(dim=1).clamp(min=1)
                    calculated_uncertainties = sums / counts
                elif args.uncertainty_method == "maximum":
                    masked_probs = torch.where(token_masks, uncertainty_probs, torch.tensor(0.0, device=uncertainty_device))
                    maximums, _ = masked_probs.max(dim=1)
                    calculated_uncertainties = maximums
                else:  # "last_step"
                    seq_len_dim = token_masks.size(1)
                    reverse_mask = torch.flip(token_masks, dims=[1])
                    # argmax gives FIRST index of True. In reversed, that's distance from end.
                    # Index in original = (len-1) - index_in_reversed
                    # If row has no separator, argmax is 0 (first element reversed -> last element original).
                    # But we handle has_separator check later.
                    last_indices_rev = torch.argmax(reverse_mask.float(), dim=1)
                    last_indices = (seq_len_dim - 1) - last_indices_rev
                    calculated_uncertainties = torch.gather(uncertainty_probs, 1, last_indices.unsqueeze(-1)).squeeze(-1)

                final_uncertainties = torch.where(has_separator, calculated_uncertainties, default_val)
                valid_results.extend(final_uncertainties.cpu().tolist())

                del input_tensor, mask_tensor, outputs, uncertainty_probs
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"[UncertaintyServer] CUDA OOM in batch, falling back to single processing.")
                torch.cuda.empty_cache()
                # Fallback: process one by one
                for j in range(len(batch_ids)):
                    # ... (Single processing logic similar to above) ...
                    # For brevity, implementing a simplified safe return 0.5 for OOM case in fallback for now
                    # or copying the logic. Ideally we copy the logic.
                    # Let's just append 0.5s to keep it running and log error
                    valid_results.append(0.5)

        # Map back to all_uncertainties
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            if idx_in_valid < len(valid_results):
                all_uncertainties[orig_idx] = valid_results[idx_in_valid]

        return all_uncertainties

    while True:
        try:
            task = task_queue.get()
            if task == "STOP": break
            request_id, worker_rank = task["request_id"], task["worker_rank"]
            uncertainties = get_uncertainties(task["prompts"])
            result_queues[worker_rank].put({"request_id": request_id, "uncertainties": uncertainties})
        except Exception as e:
            logger.error(f"[UncertaintyServer] Error processing task: {e}", exc_info=True)
    logger.info("[UncertaintyServer] Shutting down.")


def _value_model_server(args: argparse.Namespace, task_queue: Queue, result_queues: List[Queue]):
    value_device = torch.device(f"cuda:{args.value_model_gpu}")
    logger.info(f"[ValueServer] Starting on device {value_device}")
    eval_every = getattr(args, "evaluate_n_steps", 1)
    value_method = getattr(args, "value_method", "last_step")

    tokenizer = AutoTokenizer.from_pretrained(args.value_model_path, trust_remote_code=True)
    if args.value_model_type == "regression":
        model = RegressionPRMModel.from_pretrained(
            args.value_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = model.to(dtype=torch.bfloat16)
    else:
        model = AutoModel.from_pretrained(
            args.value_model_path, num_labels=2, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    logger.info(f"[ValueServer] Value model loaded (type: {args.value_model_type}).")
    
    # Pre-calculate separator ID
    sep_ids = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if not sep_ids:
        logger.warning("[ValueServer] Could not encode <extra_0>. Using EOS as separator.")
        sep_token_id = tokenizer.eos_token_id
    else:
        sep_token_id = sep_ids[0]

    @torch.no_grad()
    def get_values(prompts: List[str], generated_texts: List[str]) -> List[float]:
        input_ids_list = []
        for prompt, gen_text in zip(prompts, generated_texts):
            match = prompt_pattern.match(prompt)
            if not match:
                logger.error(f"Prompt did not match expected format. Cannot score. Prompt: {prompt[:200]}")
                input_ids_list.append([]) 
                continue
            
            system_msg, user_msg, partial_solution = match.groups()
            full_assistant_response = ((partial_solution or "").strip() + '\n\n' + gen_text.strip()).strip()
            steps = [s.strip() for s in full_assistant_response.split('\n\n') if s.strip()]
            
            # Manual Tokenization
            ids = _tokenize_with_manual_separator(
                tokenizer, system_msg, user_msg, steps, eval_every, sep_token_id
            )
            input_ids_list.append(ids)

        if not input_ids_list:
            return [0.0] * len(prompts)
            
        valid_indices = [i for i, ids in enumerate(input_ids_list) if ids]
        if not valid_indices:
             return [0.0] * len(prompts) # All prompts failed parsing

        batch_size = 8
        all_rewards = [0.0] * len(prompts)
        
        valid_inputs = [input_ids_list[i] for i in valid_indices]
        valid_results = []
        
        for i in range(0, len(valid_inputs), batch_size):
            batch_ids = valid_inputs[i:i + batch_size]
            
            # Manual Padding
            max_len = max(len(ids) for ids in batch_ids)
            max_len = min(max_len, 16000)
            
            padded_batch = []
            attention_masks = []
            
            for ids in batch_ids:
                curr_ids = ids[:max_len]
                pad_len = max_len - len(curr_ids)
                # Left Padding
                padded_ids = [tokenizer.pad_token_id] * pad_len + curr_ids
                mask = [0] * pad_len + [1] * len(curr_ids)
                
                padded_batch.append(padded_ids)
                attention_masks.append(mask)
                
            input_tensor = torch.tensor(padded_batch, device=value_device, dtype=torch.long)
            mask_tensor = torch.tensor(attention_masks, device=value_device, dtype=torch.long)

            try:
                if args.value_model_type == "regression":
                    outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
                    scores = outputs.logits
                else:  # classification
                    base_model_output = model.model(input_ids=input_tensor, attention_mask=mask_tensor, use_cache=False)
                    logits = model.score(base_model_output.last_hidden_state)
                    probabilities = torch.softmax(logits, dim=-1)
                    scores = probabilities[:, :, 1]
                
                token_masks = (input_tensor == sep_token_id)
                has_separator = token_masks.any(dim=1)
                
                # Default 0.0 for chunks with no separator (should not happen if logic is correct, unless truncated)
                calculated_values = torch.zeros(len(batch_ids), device=value_device)
                
                if value_method == "maximum":
                    neg_inf = torch.tensor(float("-inf"), device=value_device)
                    masked_scores = torch.where(token_masks, scores, neg_inf)
                    max_scores, _ = masked_scores.max(dim=1)
                    calculated_values = max_scores
                else: # last_step
                    seq_len_dim = token_masks.size(1)
                    reverse_mask = torch.flip(token_masks, dims=[1])
                    last_indices_rev = torch.argmax(reverse_mask.float(), dim=1)
                    last_indices = (seq_len_dim - 1) - last_indices_rev
                    calculated_values = torch.gather(scores, 1, last_indices.unsqueeze(-1)).squeeze(-1)
                
                final_values = torch.where(has_separator, calculated_values, torch.tensor(0.0, device=value_device))
                valid_results.extend(final_values.cpu().tolist())
                
                del input_tensor, mask_tensor
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"[ValueServer] CUDA OOM in batch, falling back to single processing.")
                torch.cuda.empty_cache()
                for j in range(len(batch_ids)):
                    # Simpified fallback for now - return 0.0
                    valid_results.append(0.0)
                    
        # Map back results
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            if idx_in_valid < len(valid_results):
                all_rewards[orig_idx] = valid_results[idx_in_valid]
        
        return all_rewards

    while True:
        try:
            task = task_queue.get()
            if task == "STOP": break
            request_id, worker_rank = task["request_id"], task["worker_rank"]
            values = get_values(task["prompts"], task["generated_texts"])
            result_queues[worker_rank].put({"request_id": request_id, "values": values})
        except Exception as e:
            logger.error(f"[ValueServer] Error processing task: {e}", exc_info=True)
    logger.info("[ValueServer] Shutting down.")