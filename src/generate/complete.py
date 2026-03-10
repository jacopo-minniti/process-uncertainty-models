import os
import random
from multiprocessing import Process
from time import sleep
import logging
from typing import Optional, Literal, Union

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from pydantic import BaseModel

from rely.utils import format_prompt, load_dataset, save_dataset, merge, MMLU_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)


class CompleterConfig(BaseModel):
    model: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"  # Change this to your local model path
    tp_size: int = 1
    dp_size: int = 8
    max_num_seqs: int = 512
    forking_strategy: Literal["entropy", "newline"] = "newline"
    completion_type: Literal["short", "long"] = "long"
    system_prompt: str = MATH_SYSTEM_PROMPT
    dataset: str  # Can be local file path or HF dataset name
    subset: Optional[str] = None  # For HF datasets
    split: str = "train"  # For HF datasets
    question_field: str = "question"
    max_steps: Optional[int] = None  # Skip attempts with more than N steps
    step_separator: str = "\n\n"  # Separator used to split attempts into steps. When cutting at step N, includes everything up to and including step N, so the separator becomes the "ending token" of each step.
    every_n_steps: int = 1  # Sample every n steps (1 means every step, 5 means steps 0, 4, 9, etc.)


class Completer:

    def __init__(self, config: CompleterConfig):
        self.config = config


    def _format_prompt_with_completion_type(self, question: str, cut_cot: str) -> str:
        """
        Format the prompt based on completion_type.
        
        Args:
            question: The question text to format.
            cut_cot: The partial chain of thought to include.
            
        Returns:
            A formatted prompt string.
        """
        if self.config.completion_type == "short":
            # For short completion, add forcing text to make the model provide a final answer
            forcing_text = "\n\nI reasoned enough, the user wants a final answer.\n## Final Answer\n\\boxed{"
            modified_cot = cut_cot + forcing_text
            return format_prompt(question, system_prompt=self.config.system_prompt, cot=modified_cot)
        else:
            # For long completion (default behavior), use the original format
            return format_prompt(question, system_prompt=self.config.system_prompt, cot=cut_cot)


    def _get_num_steps(self, item: dict) -> int:
        """Get the number of steps in the CoT for an item."""
        if self.config.forking_strategy == "entropy":
            # For entropy strategy, we don't have step information
            return 1
        else:
            attempt = item.get("attempt", "")
            if not attempt:
                return 0
            steps = attempt.split(self.config.step_separator)
            return len(steps)


    def split_attempt(self, item: dict, used_steps: Optional[set] = None) -> str:
        if self.config.forking_strategy == "entropy":
            return item.get("cut_cot", "")
        else:
            attempt = item.get("attempt", "")
            if not attempt:
                return ""
            steps = attempt.split(self.config.step_separator)
            if len(steps) <= 1:
                return attempt
            
            max_step = len(steps) - 1
            
            # Ensure different samples when used_steps is provided
            if used_steps is not None:
                available_steps = [i for i in range(max_step + 1) if i not in used_steps]
                if not available_steps:
                    # If all steps have been used, fall back to random selection
                    sampled_step = random.randint(0, max_step) if max_step > 0 else 0
                else:
                    sampled_step = random.choice(available_steps)
                used_steps.add(sampled_step)
            else:
                sampled_step = random.randint(0, max_step) if max_step > 0 else 0
                
            cut_steps = steps[:sampled_step + 1]

            return self.config.step_separator.join(cut_steps)


    def generate(
        self,
        output_file: str,
        n_completions_per_item: int = 100,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        every_n_steps: int = 1,  # Sample every n steps (1 means every step)
    ):
        """
        Generate completions from the dataset, sampling every n steps of the CoT.
        
        Args:
            output_file: Path to save the generated completions.
            n_completions_per_item: Number of completions to generate for each CoT step.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature for generation.
            every_n_steps: Sample every n steps (1 means every step, 5 means steps 0, 4, 9, etc.).
        """
        node_size = 1
        node_rank = 0
        master_addr = "127.0.0.1"
        master_port = 13345
        gpu_memory_utilization = 0.9
        tp_size = self.config.tp_size
        dp_size = self.config.dp_size
        max_num_seqs = self.config.max_num_seqs
        model = self.config.model

        if dp_size > 1:
            if node_size > 1:
                _master_addr = master_addr
                _master_port = master_port
            else:
                _master_addr = "127.0.0.1"
                _master_port = get_open_port()
                logging.info(f"Single node run. Using master address {_master_addr}:{_master_port}")
            if dp_size % node_size != 0:
                raise ValueError("dp_size must be divisible by node_size.")
            dp_per_node = dp_size // node_size
            processes = []
            for local_dp_rank in range(dp_per_node):
                global_dp_rank = node_rank * dp_per_node + local_dp_rank
                proc = Process(
                    target=self._worker,
                    args=(
                        output_file,
                        n_completions_per_item,
                        max_new_tokens,
                        temperature,
                        every_n_steps,
                        model,
                        dp_size,
                        tp_size,
                        gpu_memory_utilization,
                        max_num_seqs,
                        global_dp_rank,
                        local_dp_rank,
                        _master_addr,
                        _master_port,
                    ),
                )
                proc.start()
                processes.append(proc)
            exit_code = 0
            for proc in processes:
                proc.join(timeout=35000)
                if proc.exitcode is None:
                    logging.warning(f"Process {proc.pid} timed out. Killing...")
                    proc.kill()
                    exit_code = 1
                elif proc.exitcode != 0:
                    logging.error(f"Process {proc.pid} exited with error code {proc.exitcode}.")
                    exit_code = proc.exitcode
            if exit_code == 0:
                self._merge_output_files(output_file, dp_size)
            else:
                logging.warning("One or more processes failed. Skipping file merge.")
            if exit_code != 0:
                raise RuntimeError(f"Completion failed with exit code {exit_code}")
        else:
            logging.info("Running in single-process mode (dp_size=1).")
            self._worker(
                output_file,
                n_completions_per_item,
                max_new_tokens,
                temperature,
                every_n_steps,
                model,
                dp_size,
                tp_size,
                gpu_memory_utilization,
                max_num_seqs,
                0,  # global_dp_rank
                0,  # local_dp_rank
                '127.0.0.1',
                get_open_port()
            )
            base, ext = os.path.splitext(output_file)
            os.rename(f"{base}.rank0{ext}", output_file)
            logging.info(f"Processing complete! Output written to {output_file}")


    def _worker(self, output_file, n_completions_per_item, max_new_tokens, temperature, every_n_steps, model, dp_size, tp_size, gpu_memory_utilization, max_num_seqs, global_dp_rank, local_dp_rank, master_addr, master_port):
        logging.info(f"Starting worker: global_rank={global_dp_rank}, local_rank={local_dp_rank}")
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = master_addr
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
        
        # Let SLURM handle GPU allocation - don't override CUDA_VISIBLE_DEVICES

        all_data = load_dataset(
            self.config.dataset,
            subset=self.config.subset,
            split=self.config.split,
            question_field=self.config.question_field,
        )
        if all_data is None:
            return
        
        total_items = len(all_data)
        chunk_size = total_items // dp_size
        remainder = total_items % dp_size

        # Calculate data chunk for this DP rank
        start_idx = global_dp_rank * chunk_size + min(global_dp_rank, remainder)
        end_idx = (global_dp_rank + 1) * chunk_size + min(global_dp_rank + 1, remainder)
        data_chunk = all_data[start_idx:end_idx]

        if not data_chunk:
            logging.warning(f"DP rank {global_dp_rank} has no items to process. Exiting worker.")
            return
        logging.info(f"DP rank {global_dp_rank} preparing {len(data_chunk)} items.")
        logging.info(f"DP rank {global_dp_rank} using completion_type: {self.config.completion_type}")

        prompts_to_process = []
        source_indices = []

        for item_idx, item in enumerate(data_chunk):
            question = item.get("question", "") + r" \think"
            if not question:
                logging.warning(f"Skipping item on rank {global_dp_rank}: missing question")
                continue
            
            # Handle entropy strategy which doesn't have multiple steps
            if self.config.forking_strategy == "entropy":
                cut_cot = item.get("cut_cot", "")
                prompt = self._format_prompt_with_completion_type(question, cut_cot)
                prompts_to_process.append(prompt)
                source_indices.append({
                    "item_idx": item_idx,
                    "cut_cot": cut_cot,
                    "sample_idx": 0
                })
                continue
            
            # For "newline" strategy, sample every step of the CoT
            attempt = item.get("attempt", "")
            if not attempt:
                continue
            
            # Skip attempts that don't contain "</think>" substring
            # if "</think>" not in attempt:
            #     continue

            steps = attempt.split(self.config.step_separator)
            num_steps = len(steps)
            if num_steps == 0:
                continue
            
            # Skip attempts with more than max_steps if specified
            if self.config.max_steps is not None and num_steps > self.config.max_steps:
                continue
            
            # Create a prompt for every n steps
            for step_index in range(0, num_steps, self.config.every_n_steps):
                cut_cot = self.config.step_separator.join(steps[:step_index + 1])
                prompt = self._format_prompt_with_completion_type(question, cut_cot)
                prompts_to_process.append(prompt)
                
                source_indices.append({
                    "item_idx": item_idx, 
                    "cut_cot": cut_cot,
                    "sample_idx": step_index
                })

        if not prompts_to_process:
            logging.warning(f"DP rank {global_dp_rank} has no valid prompts to process after filtering. Exiting.")
            return
        
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            max_model_len=12000,
        )

        logging.info(f"DP rank {global_dp_rank} starting generation for {len(prompts_to_process)} prompts...")
        
        # Adjust max_tokens based on completion_type
        adjusted_max_tokens = max_new_tokens
        if self.config.completion_type == "short":
            # For short completions, we expect much shorter outputs (just the final answer)
            adjusted_max_tokens = min(50, max_new_tokens)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=adjusted_max_tokens,
            n=n_completions_per_item,
        )

        all_outputs = llm.generate(prompts_to_process, sampling_params)
        base, ext = os.path.splitext(output_file)
        output_file_path = f"{base}.rank{global_dp_rank}{ext}"

        processed_outputs = {}

        for source_info, output_group in zip(source_indices, all_outputs):
            item_idx = source_info["item_idx"]
            
            if item_idx not in processed_outputs:
                processed_outputs[item_idx] = {
                    "original_item": data_chunk[item_idx],
                    "samples": []
                }
            
            completions = [out.text for out in output_group.outputs]
            processed_outputs[item_idx]["samples"].append({
                "sample_idx": source_info["sample_idx"],
                "cut_cot": source_info["cut_cot"],
                "completions": completions
            })

        output_data = list(processed_outputs.values())
        
        save_dataset(output_data, output_file_path)
        
        logging.info(f"DP rank {global_dp_rank} finished. Processed {len(output_data)} original items.")
        logging.info(f"Output for rank {global_dp_rank} written to {output_file_path}")
        
        sleep(1)


    def _merge_output_files(self, output_file: str, dp_size: int) -> None:
        base, ext = os.path.splitext(output_file)
        input_files = [f"{base}.rank{rank}{ext}" for rank in range(dp_size)]
        logging.info(f"Merging output files: {input_files}")
        merge(input_files, output_file)
        logging.info(f"Final merged output written to {output_file}")


def main():
    """
    Main function that runs the complete pipeline:
    """    

    input_dataset = "data/mmlu/generations_qwen2.5.jsonl"
    completions_file = "data/mmlu/completions_qwen2.5.jsonl"  
    
    completer_config = CompleterConfig(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        tp_size=1,
        dp_size=4,
        max_num_seqs=128,
        forking_strategy="newline",
        completion_type="long",
        dataset=input_dataset,
        question_field="question",
        max_steps=150,
        step_separator="\n\n",
        every_n_steps=1
    )

    completer = Completer(completer_config)
    completer.generate(
        output_file=completions_file,
        n_completions_per_item=8,
        max_new_tokens=5000,
        temperature=1.0,
    )

if __name__ == "__main__":
    exit(main())