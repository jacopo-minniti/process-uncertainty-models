"""
Generate completions from local JSONL files using vLLM.

This module provides functions for generating model completions from local JSONL files
with support for data parallelism and distributed inference.
"""

import argparse
import json
import os
import logging
from multiprocessing import Process
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port

from rely.utils.text_utils import format_prompt, MMLU_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_from_dataset(
    input_file: str,
    output_file: str,
    model: str = "Qwen/Qwen3-1.7B",
    question_field: str = "question",
    answer_field: str = "solution",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tp_size: int = 1,
    dp_size: int = 8,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 128,
    dtype: str = "bfloat16",
    system_prompt: str = MMLU_SYSTEM_PROMPT,
    n_generations_per_cot: int = 1,
) -> None:
    """
    Generate completions from a local JSONL file using vLLM with data parallelism.

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file
        model: The model identifier for vLLM
        question_field: The field in the dataset for the question text
        answer_field: The field in the dataset for the correct answer's index
        temperature: Sampling temperature for the model
        max_tokens: Maximum number of tokens to generate
        tp_size: Number of tensor parallel replicas PER data parallel rank
        dp_size: Total data parallel size (number of ranks across all nodes)
        gpu_memory_utilization: GPU memory utilization fraction
        max_num_seqs: Maximum number of sequences per batch
        dtype: Model data type
        system_prompt: The system prompt to use for formatting prompts.
        n_generations_per_cot: Number of independent Chain-of-Thought generations per prompt.
    """
    # Create argument namespace for compatibility with existing code
    class Args:
        def __init__(self):
            self.input_file = input_file
            self.output_file = output_file
            self.model = model
            self.question_field = question_field
            self.answer_field = answer_field
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.tp = tp_size
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_num_seqs = max_num_seqs
            self.dtype = dtype
            self.system_prompt = system_prompt
            self.n_generations_per_cot = n_generations_per_cot

    args = Args()

    # Multiprocessing setup for data parallelism
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    
    for i in range(dp_size):
        proc = Process(target=_generate_worker,
                      args=(args, dp_size, i, dp_master_ip, dp_master_port))
        proc.start()
        procs.append(proc)
    
    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            logging.error(f"Process {proc.pid} exited with code {proc.exitcode}")
            exit_code = proc.exitcode

    if exit_code != 0:
        raise RuntimeError(f"Generation failed with exit code {exit_code}")


def _generate_worker(args, dp_size: int, dp_rank: int, dp_master_ip: str, dp_master_port: int) -> None:
    """
    Worker function for data parallel generation.
    
    Args:
        args: Arguments namespace
        dp_size: Total data parallel size
        dp_rank: Current data parallel rank
        dp_master_ip: Master node IP address
        dp_master_port: Master node port
    """
    # Set Environment Variables for Data Parallelism
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # Initialize vLLM
    logging.info(f"[Rank {dp_rank}] Initializing vLLM...")
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            dtype=args.dtype
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        logging.error(f"[Rank {dp_rank}] Failed to initialize vLLM: {e}")
        return

    # Load local JSONL file
    logging.info(f"[Rank {dp_rank}] Loading dataset from '{args.input_file}'...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    except Exception as e:
        logging.error(f"[Rank {dp_rank}] Failed to load dataset: {e}")
        return

    # Determine Data Slice for this Rank
    total_samples = len(dataset)
    samples_per_rank = total_samples // dp_size
    start_idx = dp_rank * samples_per_rank
    end_idx = (dp_rank + 1) * samples_per_rank
    if dp_rank == dp_size - 1:
        end_idx = total_samples  # Ensure the last rank gets all remaining samples

    # Extract and Format Prompts for this Rank's Slice
    full_prompts = []
    formatted_llm_prompts = []
    original_records = [] 
    prompt_generation_indices = []  # Track which generation this is for each prompt
    
    logging.info(f"[Rank {dp_rank}] Preparing prompts from dataset slice (indices {start_idx} to {end_idx-1})...")
    logging.info(f"[Rank {dp_rank}] Generating {args.n_generations_per_cot} independent CoTs per prompt...")
    
    # Create a list of indices for this rank to process
    rank_indices = list(range(start_idx, end_idx))
    
    for i in tqdm(rank_indices, desc=f"[Rank {dp_rank}] Preparing Prompts"):
        example = dataset[i]
        
        question = example.get(args.question_field) + r" \think"
        
        if not question or not isinstance(question, str):
            logging.warning(f"\n[Rank {dp_rank}] Skipping sample {i} due to missing/invalid question field.")
            continue
        
        # Generate multiple independent CoTs for each prompt
        for generation_idx in range(args.n_generations_per_cot):
            full_prompts.append(question)
            formatted_llm_prompts.append(format_prompt(question, args.system_prompt))
            original_records.append(example)
            prompt_generation_indices.append(generation_idx)

    if not formatted_llm_prompts:
        logging.error(f"[Rank {dp_rank}] No valid prompts were generated for this rank. Exiting.")
        return
        
    unique_questions = len(formatted_llm_prompts) // args.n_generations_per_cot
    logging.info(f"[Rank {dp_rank}] Generated {len(formatted_llm_prompts)} prompts to process ({unique_questions} unique questions × {args.n_generations_per_cot} generations each).")

    # Run vLLM Inference
    logging.info(f"[Rank {dp_rank}] Starting vLLM inference...")
    outputs = llm.generate(formatted_llm_prompts, sampling_params)
    logging.info(f"[Rank {dp_rank}] Inference complete.")

    # Save Results
    base, ext = os.path.splitext(args.output_file)
    rank_output_file = f"{base}_{dp_rank}{ext}"

    with open(rank_output_file, 'w', encoding='utf-8') as outfile:
        sorted_outputs = sorted(outputs, key=lambda x: int(x.request_id))
        
        for i, output in enumerate(tqdm(sorted_outputs, desc=f"[Rank {dp_rank}] Saving Results")):
            assistant_response = output.outputs[0].text.strip()
            
            original_record = original_records[i]
            full_question_prompt = full_prompts[i]
            generation_idx = prompt_generation_indices[i]
            
            answer_idx = original_record.get(args.answer_field)

            result = {
                "question": full_question_prompt,
                "attempt": assistant_response,
                "solution": answer_idx,
                "generation_index": generation_idx
            }
            outfile.write(json.dumps(result) + '\n')

    logging.info(f"[Rank {dp_rank}] Processing complete. Results saved to {rank_output_file}")


def main():
    """Command-line interface for generate_from_dataset."""
    parser = argparse.ArgumentParser(description="Query a local vLLM model with a local JSONL file using Data Parallelism.")

    # Dataset Arguments
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--question-field", type=str, default="question", help="The field in the dataset for the question text.")
    parser.add_argument("--answer-field", type=str, default="solution", help="The field in the dataset for the correct answer.")

    # vLLM & Model Arguments
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="The model identifier for vLLM.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSONL file. Rank will be appended.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the model.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum number of tokens to generate.")
    parser.add_argument("--tp-size", type=int, default=1, help="Number of tensor parallel replicas PER data parallel rank.")
    parser.add_argument("--dp-size", type=int, default=4, help="Total data parallel size.")
    parser.add_argument("--n-generations-per-cot", type=int, default=1, help="Number of independent Chain-of-Thought generations per prompt.")

    args = parser.parse_args()

    generate_from_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        question_field=args.question_field,
        answer_field=args.answer_field,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        n_generations_per_cot=args.n_generations_per_cot
    )


if __name__ == "__main__":
    main() 