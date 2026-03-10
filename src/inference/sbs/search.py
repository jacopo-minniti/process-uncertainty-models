# rely/inference/sbs/search.py

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from typing import List, Dict, Any, Optional

from openai import OpenAI
from transformers import AutoTokenizer

from rely.utils import MATH_SYSTEM_PROMPT, extract_final_answer, normalize_answer
from .strategies import SamplingStrategy
from .utils import SBSConfig, SBSNode
from .clients import ValueClient

logger = logging.getLogger(__name__)

# --- OpenAI Client Configuration ---
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"

class StepBeamSearch:
    """Step-level Beam Search implementation with pluggable sampling strategies."""
    def __init__(self,
                 inference_model_name: str,
                 config: SBSConfig,
                 strategy: SamplingStrategy,
                 value_client: ValueClient,
                 worker_rank: int):
        self.config = config
        if self.config.max_depth is None and self.config.budget is None:
            raise ValueError("Either max_depth or budget must be specified for the search.")
        
        self.inference_model_name = inference_model_name
        self.strategy = strategy
        self.worker_rank = worker_rank
        self.value_client = value_client
        self.error_reasons: list[str] = []

        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        logger.info(f"[Rank {self.worker_rank}] Client initialized to connect to {OPENAI_API_BASE}")
        
        self.root: Optional[SBSNode] = None
        self.active_beams: List[SBSNode] = []
        self.completed_beams: List[SBSNode] = []
        self.current_beam_width: int = config.step_beam_width
        self._prompt_cache = {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.inference_model_name, trust_remote_code=True)
        self.total_generated_tokens = 0
        if config.evaluate_n_steps < 1:
            raise ValueError("evaluate_n_steps must be at least 1.")
        self.evaluate_n_steps = config.evaluate_n_steps
        self.current_step = 0

    def clear_cache(self):
        self._prompt_cache.clear()
        self.total_generated_tokens = 0
        self.current_step = 0
        self.error_reasons = []

    def _record_error(self, reason: str):
        """Track run-level errors so they can be written to summary.json."""
        if reason:
            self.error_reasons.append(reason)

    def create_prompt(self, question: str, partial_solution: str = "") -> str:
        cache_key = (question, partial_solution)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        prompt = f"<|im_start|>system\n{MATH_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{partial_solution}"
        self._prompt_cache[cache_key] = prompt
        return prompt

    def should_evaluate_depth(self, depth: int) -> bool:
        return depth % self.evaluate_n_steps == 0

    def should_log_scored_steps(self) -> bool:
        """
        When ``evaluate_n_steps`` > 1, most intermediate nodes reuse their parent's score
        and are not re-scored. To reduce noisy logs, only emit per-step verbose logs on
        iterations where at least one newly generated child will actually be scored.
        """
        if not self.config.verbose:
            return False
        if self.evaluate_n_steps <= 1:
            return True
        if not self.active_beams:
            return False
        return any(self.should_evaluate_depth(beam.depth + 1) for beam in self.active_beams)

    def should_evaluate_current_step(self) -> bool:
        """
        Determine whether active beams should refresh their uncertainty scores.
        We only have meaningful uncertainty targets once a beam has produced at
        least ``evaluate_n_steps`` steps (otherwise no <extra_0> separator is injected
        for the PUM models). Therefore, trigger the refresh based on the depth of the
        current beams rather than the loop counter.
        """
        if not self.active_beams:
            return False
        return any(
            beam.depth > 0 and self.should_evaluate_depth(beam.depth)
            for beam in self.active_beams
        )

    def _get_values(self, prompts: List[str], generated_texts: List[str]) -> List[float]:
        """Get values by making a call to the value client."""
        return self.value_client.get_values(prompts, generated_texts)

    def _make_api_request_with_samples(self, prompt: str, n_samples: int) -> List[Dict[str, Any]]:
        if n_samples <= 0:
            return []
        
        request_params = {
            "model": self.inference_model_name,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stop": ["\n\n"],
            "n": n_samples,
        }

        try:
            completion = self.client.completions.create(**request_params)
            results = []
            for choice in completion.choices:
                results.append({
                    'text': choice.text,
                    'logprobs': choice.logprobs
                })
                self.total_generated_tokens += len(self.tokenizer.encode(choice.text, add_special_tokens=False))
            return results
        except Exception as e:
            logger.error(f"[Rank {self.worker_rank}] Error during API call: {e}")
            self._record_error(f"vllm_api_error: {e}")
            return []

    def _generate_and_score_candidates(self, question: str) -> List[SBSNode]:
        if not self.active_beams:
            return []

        sample_distribution = self.strategy.distribute_samples(self, question)

        all_candidates, seen_full_texts = [], set()
        prompts = [self.create_prompt(question, node.full_text) for node in self.active_beams]
        
        generated_outputs = [[] for _ in self.active_beams]
        with ThreadPoolExecutor(max_workers=len(self.active_beams)) as executor:
            future_to_index = {executor.submit(self._make_api_request_with_samples, prompt, n_samples): i 
                               for i, (prompt, n_samples) in enumerate(zip(prompts, sample_distribution))}
                
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    generated_outputs[index] = future.result()
                except Exception as exc:
                    logger.error(f"[Rank {self.worker_rank}] API request generated an exception: {exc}")
                    self._record_error(f"candidate_generation_error: {exc}")

        value_request_prompts, value_request_texts, candidate_data = [], [], []
        for i, gen_results in enumerate(generated_outputs):
            parent_node = self.active_beams[i]
            for gen_result in gen_results:
                candidate_depth = parent_node.depth + 1
                needs_eval = self.should_evaluate_depth(candidate_depth)
                if needs_eval:
                    value_request_prompts.append(prompts[i])
                    value_request_texts.append(gen_result['text'])
                candidate_data.append({
                    'parent_node': parent_node,
                    'generation_result': gen_result,
                    'needs_eval': needs_eval
                })

        values = self._get_values(value_request_prompts, value_request_texts) if value_request_prompts else []
        if self.should_log_scored_steps() and value_request_prompts:
            logger.info(
                f"[Rank {self.worker_rank}] Value scoring requests: count={len(value_request_prompts)} "
                f"depths={[c['parent_node'].depth + 1 for c in candidate_data if c['needs_eval']]}"
            )
            logger.info(f"[Rank {self.worker_rank}] Value scores (first 10): {[round(v,4) for v in values[:10]]}")
        eval_idx = 0
        for data in candidate_data:
            parent_node = data['parent_node']
            gen_result = data['generation_result']
            gen_text = gen_result['text']
            needs_eval = data['needs_eval']
            
            if needs_eval:
                new_step_value = values[eval_idx]
                eval_idx += 1
            else:
                if self.config.value_method == 'product':
                    new_step_value = 1.0
                else:
                    new_step_value = parent_node.value if parent_node.depth > 0 else 0.0
            snippet = gen_text.rstrip() + '\n\n'
            child_node = parent_node.add_child(snippet)
            child_node.prm_score = new_step_value if needs_eval else None
            
            if self.config.value_method == 'product':
                child_node.value = parent_node.value * new_step_value
            elif self.config.value_method == 'maximum':
                child_node.value = max(parent_node.value, new_step_value) if parent_node.depth > 0 else new_step_value
            else:
                child_node.value = new_step_value
            
            self.strategy.update_candidate_uncertainty(child_node, gen_result)

            if ans := extract_final_answer(gen_text):
                child_node.is_terminal, child_node.final_answer = True, ans
                
            if self.config.remove_duplicate:
                if child_node.full_text in seen_full_texts:
                    continue
                seen_full_texts.add(child_node.full_text)
                
            all_candidates.append(child_node)
        
        if self.should_log_scored_steps() and all_candidates:
            logger.info(f"Generated {len(all_candidates)} unique candidates this step.")
        if not all_candidates:
            self._record_error("no_candidates_generated")

        return all_candidates

    def _update_beams(self, candidates: List[SBSNode]) -> int:
        if not candidates:
            self.active_beams, self.current_beam_width = [], 0
            return 0
        
        ranked = self.strategy.rank_candidates(self, candidates)
        new_active_beams, newly_completed = [], 0
        
        for cand in ranked[:self.current_beam_width]:
            max_depth_reached = self.config.max_depth is not None and cand.depth >= self.config.max_depth
            if cand.is_terminal or max_depth_reached:
                self.completed_beams.append(cand)
                newly_completed += 1
            else:
                new_active_beams.append(cand)
                
        self.active_beams = new_active_beams
        self.current_beam_width = len(self.active_beams)
        return newly_completed

    def _create_summary(self, solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str]) -> Dict[str, Any]:
        # 1. Normalize Ground Truth
        normalized_gt = normalize_answer(ground_truth)
        gt_usable = normalized_gt != ""

        # 2. Normalize Solutions and Prepare for Voting/BoN
        normalized_answers = []
        best_solution = None
        max_value = float('-inf')
        
        for sol in solutions:
            # Normalize answer and store it back in solution
            raw_ans = sol.get('final_answer') or sol.get('answer')
            norm_ans = normalize_answer(raw_ans)
            sol['normalized_answer'] = norm_ans
            
            if norm_ans:
                normalized_answers.append(norm_ans)
                
                # BoN Logic: Find solution with max value
                try:
                    val = float(sol.get('value', float('-inf')))
                    if val > max_value:
                        max_value = val
                        best_solution = sol
                except (ValueError, TypeError):
                    pass

        # 3. Strict Majority Vote
        majority_vote = None
        is_majority_correct = False
        if normalized_answers:
            counts = Counter(normalized_answers)
            most_common = counts.most_common()
            # Check for ties
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                majority_vote = most_common[0][0]
                if gt_usable and majority_vote == normalized_gt:
                    is_majority_correct = True
            else:
                majority_vote = "TIE" # Keep TIE as it is a specific state
        
        # 4. Best of N
        best_of_n_answer = None
        is_best_of_n_correct = False
        if best_solution:
            best_of_n_answer = best_solution['normalized_answer']
            if gt_usable and best_of_n_answer == normalized_gt:
                is_best_of_n_correct = True

        # 5. Accuracy (Raw percentage of correct beams)
        accuracy = None
        if gt_usable and normalized_answers:
            correct_count = sum(1 for ans in normalized_answers if ans == normalized_gt)
            accuracy = f"{correct_count / len(normalized_answers):.2%}"

        return {
            "question": question,
            "ground_truth": ground_truth,
            "normalized_ground_truth": normalized_gt,
            "gt_usable": gt_usable,
            "majority_vote": majority_vote,
            "is_majority_correct": is_majority_correct,
            "best_of_n_answer": best_of_n_answer,
            "is_best_of_n_correct": is_best_of_n_correct,
            "accuracy": accuracy,
            "solutions": solutions,
            "total_tokens": self.total_generated_tokens,
            "error": "; ".join(self.error_reasons) if self.error_reasons else None
        }

    def _save_results(self, final_beams: List[SBSNode], base_path: str, question: str, ground_truth: Optional[str]):
        solutions = []
        os.makedirs(base_path, exist_ok=True)

        for idx, node in enumerate(final_beams):
            if not node.final_answer:
                node.final_answer = extract_final_answer(node.full_text) or None
            
            solution_data = {
                "beam_index": idx + 1, 
                "value": node.value, # Restoring 'value' for BoN logic (maps to mean_score)
                "mean_score": node.value, 
                "var_score": node.uncertainty, # Stored in uncertainty field
                "final_answer": node.final_answer, "depth": node.depth, "solution_path": node.full_text
            }
            solutions.append(solution_data)

        summary_path = os.path.join(base_path, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self._create_summary(solutions, question, ground_truth), f, indent=4)
        
        return solutions



    def run(self, question: str, ground_truth: Optional[str] = None, base_path: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"[Rank {self.worker_rank}] Starting SBS for question: {question[:100]}...")
        self.question = question # Store for strategy access
        self.clear_cache()
        self.root = SBSNode(text="", depth=0)
        if self.config.value_method == "product":
            self.root.value = 1.0
        else:
            self.root.value = 0.0
        self.root.prm_score = 0.0
            
        self.active_beams = [self.root]
        self.completed_beams = []
        self.current_beam_width = self.config.step_beam_width
        
        step = 0
        while self.active_beams:
            if (self.config.max_depth and step >= self.config.max_depth) or \
               (self.config.budget and self.total_generated_tokens >= self.config.budget):
                logger.info(f"[Rank {self.worker_rank}] Search limit reached. Stopping search.")
                break

            step += 1
            self.current_step = step
            if self.should_log_scored_steps():
                logger.info(f"\n--- [Rank {self.worker_rank}] SBS Step {step} | Active Beams: {len(self.active_beams)} ---")
            
            candidates = self._generate_and_score_candidates(question)
            if not candidates:
                logger.warning(f"[Rank {self.worker_rank}] No new candidates generated. Stopping search.")
                break
                
            self._update_beams(candidates)

        self.completed_beams.extend(self.active_beams)
        final_beams = sorted(self.completed_beams, key=lambda x: x.value, reverse=True)[:self.config.step_beam_width]
        
        solutions = []
        if base_path:
            solutions = self._save_results(final_beams, base_path, question, ground_truth)
        
        if self.config.verbose and solutions:
            best = solutions[0]
            logger.info(f"--- [Rank {self.worker_rank}] SBS Complete --- Best solution Mean: {best['mean_score']:.4f}, Var: {best['var_score']:.4f}, Answer: {best['final_answer']}")

        return {"question": question, "ground_truth": ground_truth, "solutions": solutions}
