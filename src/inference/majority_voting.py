import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json
import logging
import os
from pathlib import Path
from collections import Counter

from vllm import LLM, SamplingParams
from rely.utils import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Import helpers from sbs.py
from rely.utils import MMLU_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT, format_prompt, extract_final_answer, normalize_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency inference."""
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    num_samples: int = 20
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8


@dataclass
class SelfConsistencyResult:
    """Holds results from self-consistency inference."""
    answers: List[str]
    generated_texts: List[str]
    distribution: Dict[str, int]
    most_consistent_answer: Optional[str]
    config: SelfConsistencyConfig



class SelfConsistencyInference:
    """Efficient self-consistency inference with batch follow-up processing."""

    def __init__(self, config: SelfConsistencyConfig):
        self.config = config
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info(f"Loading model {self.config.model_name} with vLLM…")
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                enable_prefix_caching=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=15000
            )
            # Initialize tokenizer for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate(self, prompt: str, max_tokens: int, n: int = 1) -> List[str]:
        """Generate text from the model."""
        if self.llm is None:
            raise ValueError("LLM not initialized")

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
            n=n,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        
        return [output.text for output in outputs[0].outputs]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        if self.tokenizer is None:
            # Fallback to rough word count if tokenizer is not available
            return len(text.split())
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback to word count if tokenization fails
            return len(text.split())

    def _get_consistent_samples(self, system_prompt: str, user_question: str) -> List[tuple[str, str]]:
        """
        Generates N samples with single generation only.

        Returns:
            A list of tuples: (answer, full_generated_text).
        """
        base_prompt = format_prompt(user_question, system_prompt)

        # Generate all N samples
        generated_texts = self._generate(base_prompt, self.config.max_new_tokens, n=self.config.num_samples)

        results = []
        for generated_text in generated_texts:
            full_text = base_prompt + generated_text
            answer = extract_final_answer(full_text)
            # Ensure we handle None and empty strings properly
            final_answer = answer if (answer is not None and answer.strip()) else None
            results.append((final_answer, full_text))

        return results

    def run_inference(self, 
                     user_question: str, 
                     ground_truth: Optional[str] = None, 
                     base_path: Optional[str] = None,
                     system_prompt: str = MATH_SYSTEM_PROMPT) -> Dict[str, Any]:
        """Run self-consistency for a single question and return aggregated results in SBS format."""
        logger.info(f"Running self-consistency for '{user_question[:50]}...' with {self.config.num_samples} samples...")
        
        # `results` now contains (answer, text)
        results = self._get_consistent_samples(system_prompt, user_question)

        # Unpack the results into separate lists
        answers = [res[0] for res in results]
        generated_texts = [res[1] for res in results]

        for i, answer in enumerate(answers):
            logger.debug(f"Sample {i + 1}/{self.config.num_samples} ⇒ {answer}")

        # Prepare solutions in the exact same format as SBS
        base_prompt = format_prompt(user_question, system_prompt)
        prompt_tokens = self._count_tokens(base_prompt)
        
        completion_texts = [text.replace(base_prompt, "", 1) for text in generated_texts]
        completion_tokens = sum(self._count_tokens(comp) for comp in completion_texts)
        total_tokens = prompt_tokens + completion_tokens

        solutions = []
        for i, (answer, text) in enumerate(zip(answers, generated_texts)):
            # Extract just the assistant response part for solution_path (like SBS does)
            solution_path = text.split("<|im_start|>assistant\n")[-1] if "<|im_start|>assistant\n" in text else text
            
            termination_reason = "answer_found" if answer is not None else "max_tokens"
            solution_data = {
                "beam_index": i + 1,
                "final_answer": answer,
                "depth": 1,  # Self-consistency is single-step, so depth is always 1
                "termination_reason": termination_reason,
                "solution_path": solution_path
            }
            solutions.append(solution_data)

        # Save results if base_path is provided (following SBS pattern)
        saved_files = []
        if base_path:
            saved_files = save_self_consistency_result(solutions, base_path, user_question, ground_truth, total_tokens)

        return {"question": user_question, "ground_truth": ground_truth, "solutions": solutions, "total_tokens": total_tokens}


def create_self_consistency_summary(solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str], total_tokens: int = 0) -> Dict[str, Any]:
    """Create summary mirroring SBS fields needed for majority voting (no best-of-n/value fields)."""
    normalized_ground_truth = normalize_answer(ground_truth)
    gt_usable = normalized_ground_truth != ""

    normalized_answers = []

    for sol in solutions:
        raw_answer = sol.get("final_answer") or sol.get("answer")
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
        "accuracy": accuracy,
        "solutions": solutions,
        "total_tokens": total_tokens,
        "error": None
    }


def save_self_consistency_result(final_solutions: List[Dict[str, Any]], base_path: str, question: str, ground_truth: Optional[str], total_tokens: int = 0) -> List[str]:
    """Save results in exact same format as SBS."""
    import os
    saved_files = []
    os.makedirs(base_path, exist_ok=True)

    summary_path = os.path.join(base_path, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(create_self_consistency_summary(final_solutions, question, ground_truth, total_tokens), f, indent=4)
    saved_files.append(summary_path)
    
    return saved_files


def _is_run_valid(summary_path: str) -> bool:
    """
    Checks if an existing summary.json is considered 'valid'.
    Valid means: GT is usable AND Majority Vote is valid (not None/TIE).
    """
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gt_usable = bool(data.get('gt_usable', False))
        
        majority_vote = data.get('majority_vote')
        maj_applicable = gt_usable and majority_vote not in (None, "", "TIE", "None")
        
        is_valid = gt_usable and maj_applicable
        
        if not is_valid:
            logger.info(f"Invalid item found at {summary_path}: gt_usable={gt_usable}, maj={majority_vote}")
        
        return is_valid
    except Exception as e:
        logger.warning(f"Failed to check validity of {summary_path}: {e}")
        return False


def run_self_consistency(
    user_questions: List[str],
    config: Optional[SelfConsistencyConfig] = None,
    ground_truths: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    system_prompt: str = MATH_SYSTEM_PROMPT,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run self-consistency for a list of questions."""
    if config is None:
        config = SelfConsistencyConfig()

    inference = SelfConsistencyInference(config)
    results = []

    if ground_truths is None:
        ground_truths = [None] * len(user_questions)

    # Combine questions, ground truths, and original indices
    full_dataset = [
        {"question": q, "ground_truth": gt, "original_index": i}
        for i, (q, gt) in enumerate(zip(user_questions, ground_truths))
    ]

    # 1. Slice dataset based on start and end indices
    dataset = full_dataset[start_idx:end_idx]
    logger.info(f"Processing dataset slice from {start_idx} to {end_idx if end_idx is not None else len(full_dataset)}. Total items: {len(dataset)}")

    # 2. Exclude already processed questions if output_dir is provided
    if output_dir and os.path.exists(output_dir):
        processed_indices = set()
        for folder_name in os.listdir(output_dir):
            if folder_name.startswith("q_"):
                try:
                    idx = int(folder_name.split("_")[1])
                    summary_path = os.path.join(output_dir, folder_name, "summary.json")
                    
                    # Only exclude if the run exists AND is valid
                    if os.path.exists(summary_path):
                        if _is_run_valid(summary_path):
                            processed_indices.add(idx)
                        else:
                            logger.warning(f"Run {folder_name} exists but is INVALID. Will be re-run.")
                    else:
                        logger.warning(f"Run {folder_name} exists but missing summary.json. Will be re-run.")
                        
                except (IndexError, ValueError):
                    continue
        
        original_count = len(dataset)
        dataset = [item for item in dataset if item['original_index'] not in processed_indices]
        if original_count > len(dataset):
            logger.info(f"Excluded {original_count - len(dataset)} already processed questions. Remaining: {len(dataset)}")

    # Main loop
    for item in tqdm(dataset, desc="Running Self-Consistency"):
        question = item["question"]
        ground_truth = item["ground_truth"]
        original_index = item["original_index"]

        base_path = os.path.join(output_dir, f"q_{original_index:04d}") if output_dir else None
        
        # Check again if the specific directory exists, as a safeguard
        if base_path:
            summary_path = os.path.join(base_path, "summary.json")
            if os.path.exists(summary_path):
                if _is_run_valid(summary_path):
                    logger.info(f"Skipping item {original_index} as it is already completed and valid.")
                    continue
                else:
                    logger.warning(f"Item {original_index} exists but is invalid/incomplete. Overwriting.")

        try:
            result = inference.run_inference(
                user_question=question,
                ground_truth=ground_truth,
                base_path=base_path,
                system_prompt=system_prompt,
            )
            results.append(result)
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"FATAL ERROR processing item {original_index}: {e}\n{error_traceback}")
            if base_path:
                os.makedirs(base_path, exist_ok=True)
                error_summary = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "error": f"{e}\n{error_traceback}",
                    "majority_vote": None,
                    "is_majority_correct": False,
                    "accuracy": None,
                    "solutions": [],
                    "total_tokens": 0,
                    "gt_usable": False
                }
                with open(os.path.join(base_path, "summary.json"), 'w') as f:
                    json.dump(error_summary, f, indent=4)
                
                with open(os.path.join(base_path, "error.log"), 'w') as f:
                    f.write(f"Error on question index {original_index}:\n{e}\n\n{error_traceback}")
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run self-consistency inference on a dataset.")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Name of the model to use.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples to generate for each question.")
    parser.add_argument("--max_new_tokens", type=int, default=20000, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--start_idx", type=int, default=0, help="Index to start processing from in the dataset.")
    parser.add_argument("--end_idx", type=int, default=None, help="Index to end processing at in the dataset.")
    
    args = parser.parse_args()

    # dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    # questions, ground_truths = [], []

    # for item in dataset:
    #     questions.append(item["problem"])
    #     ground_truths.append(item["answer"])

    dataset_name = "data/mmlu/inference_test.jsonl"
    dataset = load_dataset(dataset_name)
    questions, ground_truths = [], []
    
    for item in dataset:
        questions.append(item["question"])
        ground_truths.append(item["solution"])

    # Create the config object from args
    sc_config = SelfConsistencyConfig(
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    system_prompt = MMLU_SYSTEM_PROMPT if "mmlu" in dataset_name.lower() else MATH_SYSTEM_PROMPT

    run_self_consistency(
        user_questions=questions,
        ground_truths=ground_truths,
        config=sc_config,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        system_prompt=system_prompt,
    )
