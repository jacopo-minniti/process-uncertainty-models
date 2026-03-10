import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json
import logging
import os
from pathlib import Path
from collections import Counter
import torch
import numpy as np

from vllm import LLM, SamplingParams
from rely.utils import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Import helpers
from rely.utils import MMLU_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT, format_prompt, extract_final_answer, normalize_answer
from rely.train.regression_prm.model import RegressionPRMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BoNConfig:
    """Configuration for Best-of-N inference."""
    # Generation Config
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    num_samples: int = 20
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.6 # Reduced to allow space for PRM if on same GPU
    
    # PRM Config
    prm_model_name: str = ""
    prm_model_type: str = "classification" # "classification" or "regression"
    prm_aggregation: str = "product" # "product", "average", "maximum", "last_step"
    prm_device: str = "cuda:0"
    step_separator: str = "<extra_0>"
    evaluate_n_steps: int = 1

class BoNInference:
    """Inference class for Best-of-N with PRM scoring."""

    def __init__(self, config: BoNConfig):
        self.config = config
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.prm_model = None
        self.prm_tokenizer = None
        
        self._initialize_generation_model()
        self._initialize_prm_model()

    def _initialize_generation_model(self):
        try:
            logger.info(f"Loading generation model {self.config.model_name} with vLLM...")
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                enable_prefix_caching=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=10000
            )
            # Initialize tokenizer for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            logger.info("Generation model loaded.")
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            raise

    def _initialize_prm_model(self):
        try:
            logger.info(f"Loading PRM model {self.config.prm_model_name} on {self.config.prm_device}...")
            self.prm_tokenizer = AutoTokenizer.from_pretrained(self.config.prm_model_name, trust_remote_code=True)
            if self.prm_tokenizer.pad_token is None:
                self.prm_tokenizer.pad_token = self.prm_tokenizer.eos_token
            
            if self.config.prm_model_type == "regression":
                self.prm_model = RegressionPRMModel.from_pretrained(
                    self.config.prm_model_name, 
                    dtype=torch.bfloat16, 
                    trust_remote_code=True
                )
            else:
                self.prm_model = AutoModel.from_pretrained(
                    self.config.prm_model_name, 
                    num_labels=2, 
                    dtype=torch.bfloat16, 
                    trust_remote_code=True
                )
            
            self.prm_model.to(self.config.prm_device)
            self.prm_model.eval()
            logger.info("PRM model loaded.")
            
            # Pre-calculate separator ID
            sep_ids = self.prm_tokenizer.encode(self.config.step_separator, add_special_tokens=False)
            if not sep_ids:
                logger.warning(f"Could not encode {self.config.step_separator}. Using EOS as separator.")
                self.sep_token_id = self.prm_tokenizer.eos_token_id
            else:
                self.sep_token_id = sep_ids[0]

        except Exception as e:
            logger.error(f"Failed to load PRM model: {e}")
            raise

    def _tokenize_for_prm(self, system_msg: str, user_msg: str, text: str) -> List[int]:
        """Tokenize prompt + generated text with separators for PRM."""
        # 1. Prompt
        messages = [
            {"role": "system", "content": system_msg.strip()},
            {"role": "user", "content": user_msg.strip()}
        ]
        prompt_text = self.prm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.prm_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        
        # 2. Steps with separators
        steps = [s.strip() for s in text.split("\n\n") if s.strip()]
        eval_every = max(1, self.config.evaluate_n_steps)
        steps_since_separator = 0
        
        for idx, step in enumerate(steps, start=1):
            step_text = step
            if idx > 1:
                step_text = "\n\n" + step_text
            
            step_ids = self.prm_tokenizer(step_text, add_special_tokens=False)["input_ids"]
            input_ids.extend(step_ids)
            
            steps_since_separator += 1
            is_last = (idx == len(steps))
            
            if steps_since_separator == eval_every or is_last:
                input_ids.append(self.sep_token_id)
                steps_since_separator = 0
                
        return input_ids

    @torch.no_grad()
    def _score_generation(self, input_ids: List[int]) -> float:
        """Score a single tokenized sequence."""
        # This simple implementation scores one by one. Batching could be added for speed.
        if not input_ids:
            return 0.0
            
        device = self.config.prm_device
        # Truncate if too long (simple handling)
        if len(input_ids) > 16000: 
            input_ids = input_ids[:16000]
            
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor, device=device)
        
        if self.config.prm_model_type == "regression":
            outputs = self.prm_model(input_ids=input_tensor, attention_mask=attention_mask)
            scores = outputs.logits # [1, seq_len]
        else:
            base_out = self.prm_model.model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=False)
            logits = self.prm_model.score(base_out.last_hidden_state)
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, :, 1] # [1, seq_len] Probability of "Good"
            
        # Filter for separator tokens
        token_masks = (input_tensor == self.sep_token_id)
        has_separator = token_masks.any()
        
        if not has_separator:
            return 0.0 # Fallback
            
        method = self.config.prm_aggregation
        # Get scores at separator positions
        separator_scores = torch.masked_select(scores, token_masks)
        
        if method == "product":
            if self.config.prm_model_type == "regression":
                # For regression, 'product' might imply sum of values ? 
                # Or assume they are log-probs? 
                # User instructions: "product of all the scores". 
                # If they are arbitrary values, product is unstable. 
                # Assuming standard PRM usage where values are ~probs.
                return torch.prod(separator_scores).item()
            else:
                return torch.prod(separator_scores).item()
        elif method == "average":
            return torch.mean(separator_scores).item()
        elif method == "maximum":
            return torch.max(separator_scores).item()
        elif method == "last_step":
            return separator_scores[-1].item()
        else:
            return separator_scores[-1].item()

    def _generate(self, prompt: str, max_tokens: int, n: int = 1) -> List[str]:
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
            n=n,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return [output.text for output in outputs[0].outputs]

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def run_inference(self, 
                     user_question: str, 
                     ground_truth: Optional[str] = None, 
                     base_path: Optional[str] = None,
                     system_prompt: str = MATH_SYSTEM_PROMPT) -> Dict[str, Any]:
        
        logger.info(f"Running BoN for '{user_question[:50]}...'")
        
        base_prompt = format_prompt(user_question, system_prompt)
        generated_texts = self._generate(base_prompt, self.config.max_new_tokens, n=self.config.num_samples)
        
        solutions = []
        scores = []
        
        # Parse, Score, and Collect
        for i, text in enumerate(generated_texts):
            answer = extract_final_answer(text)
            
            # Score
            try:
                # Extract solution part (remove prompt if it was part of text, but vLLM usually returns only completion)
                # But _tokenize_for_prm needs clean inputs. 
                # `text` here is the COMPLETION.
                input_ids = self._tokenize_for_prm(system_prompt, user_question, text)
                score = self._score_generation(input_ids)
            except Exception as e:
                logger.error(f"Error scoring sample {i}: {e}")
                score = float("-inf")
            
            scores.append(score)
            
            solution_data = {
                "beam_index": i + 1,
                "final_answer": answer,
                "solution_path": text,
                "value": score,
                "prm_score": score # Redundant but for compatibility
            }
            solutions.append(solution_data)
            
        # Sort solutions by score for BoN
        # But we keep original order in `solutions` list for consistency with beam index
        # We find best one
        
        best_idx = np.argmax(scores)
        best_solution = solutions[best_idx]
        
        # Calc Total Tokens
        prompt_tokens = self._count_tokens(base_prompt)
        completion_tokens = sum(self._count_tokens(t) for t in generated_texts)
        total_tokens = prompt_tokens + completion_tokens

        # Save if needed
        if base_path:
            save_bon_result(solutions, base_path, user_question, ground_truth, total_tokens)

        return {
            "question": user_question, 
            "ground_truth": ground_truth, 
            "solutions": solutions, 
            "total_tokens": total_tokens,
            "best_solution": best_solution
        }

def create_bon_summary(solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str], total_tokens: int = 0) -> Dict[str, Any]:
    """Create summary mirroring SBS/Majority Voting but including BoN."""
    normalized_ground_truth = normalize_answer(ground_truth)
    gt_usable = normalized_ground_truth != ""

    normalized_answers = []
    
    # 1. Majority Voting Logic
    for sol in solutions:
        raw_answer = sol.get("final_answer")
        normalized_answer = normalize_answer(raw_answer)
        sol["normalized_answer"] = normalized_answer
        if normalized_answer:
            normalized_answers.append(normalized_answer)

    majority_vote = None
    is_majority_correct = False
    if normalized_answers:
        counts = Counter(normalized_answers)
        most_common = counts.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            majority_vote = most_common[0][0]
            if gt_usable and majority_vote == normalized_ground_truth:
                is_majority_correct = True
        else:
            majority_vote = "TIE"

    # 2. BoN Logic
    # Find best score
    best_sol = max(solutions, key=lambda x: x.get("value", float("-inf")))
    bon_answer = best_sol.get("normalized_answer")
    is_bon_correct = False
    if gt_usable and bon_answer == normalized_ground_truth:
        is_bon_correct = True

    # Accuracy
    accuracy = None
    if gt_usable and normalized_answers:
        correct_answers = sum(1 for ans in normalized_answers if ans == normalized_ground_truth)
        accuracy = f"{correct_answers / len(normalized_answers):.2%}"

    return {
        "question": question,
        "ground_truth": ground_truth,
        "normalized_ground_truth": normalized_ground_truth,
        "gt_usable": gt_usable,
        "majority_vote": majority_vote,
        "is_majority_correct": is_majority_correct,
        "best_of_n_answer": bon_answer,
        "is_best_of_n_correct": is_bon_correct,
        "accuracy": accuracy,
        "solutions": solutions,
        "total_tokens": total_tokens,
        "error": None
    }

def save_bon_result(final_solutions: List[Dict[str, Any]], base_path: str, question: str, ground_truth: Optional[str], total_tokens: int = 0):
    os.makedirs(base_path, exist_ok=True)
    summary_path = os.path.join(base_path, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(create_bon_summary(final_solutions, question, ground_truth, total_tokens=total_tokens), f, indent=4)

def _is_run_valid(summary_path: str) -> bool:
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        gt_usable = bool(data.get('gt_usable', False))
        majority_vote = data.get('majority_vote')
        # Check if BoN is present? 
        # For now validity usually implies completion.
        maj_valid = gt_usable and majority_vote not in (None, "", "TIE", "None")
        return gt_usable # Simplest check: if we have GT and finished, it's valid enough.
    except Exception:
        return False

def run_bon(
    user_questions: List[str],
    config: BoNConfig,
    ground_truths: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    system_prompt: str = MATH_SYSTEM_PROMPT,
    start_idx: int = 0,
    end_idx: Optional[int] = None
):
    inference = BoNInference(config)
    
    if ground_truths is None:
        ground_truths = [None] * len(user_questions)

    full_dataset = [
        {"question": q, "ground_truth": gt, "original_index": i}
        for i, (q, gt) in enumerate(zip(user_questions, ground_truths))
    ]
    
    dataset = full_dataset[start_idx:end_idx]
    logger.info(f"Processing {len(dataset)} items.")

    # Dedup logic (skip existing)
    if output_dir and os.path.exists(output_dir):
        processed_indices = set()
        for folder_name in os.listdir(output_dir):
            if folder_name.startswith("q_"):
                try:
                    idx = int(folder_name.split("_")[1])
                    summary_path = os.path.join(output_dir, folder_name, "summary.json")
                    if os.path.exists(summary_path) and _is_run_valid(summary_path):
                        processed_indices.add(idx)
                except: continue
        dataset = [item for item in dataset if item['original_index'] not in processed_indices]
        logger.info(f"Remaining items after skipping: {len(dataset)}")

    for item in tqdm(dataset, desc="Running BoN"):
        question = item["question"]
        ground_truth = item["ground_truth"]
        original_index = item["original_index"]

        base_path = os.path.join(output_dir, f"q_{original_index:04d}") if output_dir else None
        
        try:
            inference.run_inference(
                user_question=question,
                ground_truth=ground_truth,
                base_path=base_path,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Error on {original_index}: {e}")
            if base_path:
                os.makedirs(base_path, exist_ok=True)
                with open(os.path.join(base_path, "error.log"), 'w') as f:
                    f.write(str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Generation args
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    
    # PRM args
    parser.add_argument("--prm_model_name", type=str, required=True)
    parser.add_argument("--prm_model_type", type=str, default="regression", choices=["classification", "regression"])
    parser.add_argument("--prm_aggregation", type=str, default="product", choices=["product", "average", "maximum", "last_step"])
    parser.add_argument("--prm_device", type=str, default="cuda:0")
    
    args = parser.parse_args()

    # dataset_name = "data/mmlu/inference_test.jsonl"
    dataset_name = "data/math/inference_test.jsonl"
    dataset = load_dataset(dataset_name)
    
    questions, ground_truths = [], []
    for item in dataset:
        questions.append(item["question"])
        ground_truths.append(item["solution"])

    config = BoNConfig(
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prm_model_name=args.prm_model_name,
        prm_model_type=args.prm_model_type,
        prm_aggregation=args.prm_aggregation,
        prm_device=args.prm_device
    )

    system_prompt = MMLU_SYSTEM_PROMPT if "mmlu" in dataset_name.lower() else MATH_SYSTEM_PROMPT

    run_bon(
        user_questions=questions,
        ground_truths=ground_truths,
        config=config,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        system_prompt=system_prompt
    )
