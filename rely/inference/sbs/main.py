# rely/inference/sbs/main.py

import logging
import time
import os
import traceback
from multiprocessing import Process, Queue, set_start_method
import argparse
import json

from tqdm import tqdm
from rely.utils import load_dataset

from rely.inference.sbs.strategies import (
    SamplingStrategy, UniformStrategy, PumStrategy, 
)
from rely.inference.sbs.utils import SBSConfig
from rely.inference.sbs.servers import _uncertainty_model_server, _value_model_server
from rely.inference.sbs.clients import ValueClient, UncertaintyClient
from rely.inference.sbs.search import StepBeamSearch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _is_run_valid(summary_path: str) -> bool:
    """
    Checks if an existing summary.json is considered 'valid' by sbs_eval.py criteria.
    Valid means: GT is usable AND (Majority Vote or BoN is applicable).
    """
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gt_usable = bool(data.get('gt_usable', False))
        
        majority_vote = data.get('majority_vote')
        maj_applicable = gt_usable and majority_vote not in (None, "", "TIE", "None") # added string "None" just in case
        
        bon_answer = data.get('best_of_n_answer')
        bon_applicable = gt_usable and bon_answer not in (None, "", "None")
        
        is_valid = gt_usable and (maj_applicable or bon_applicable)
        
        # Debug logging for invalid items that are about to be overwritten (or valid ones being skipped if user is debugging)
        if not is_valid:
            logger.info(f"Invalid item found at {summary_path}: gt_usable={gt_usable}, maj={majority_vote}, bon={bon_answer}")
        
        return is_valid
    except Exception as e:
        logger.warning(f"Failed to check validity of {summary_path}: {e}")
        return False


def _sbs_worker(args: argparse.Namespace, dataset_slice: list, rank: int, strategy: SamplingStrategy, 
                value_client: ValueClient, uncertainty_client: UncertaintyClient | None):
    logger.info(f"[Rank {rank}] Worker started with strategy: {args.strategy}")
    sbs_config = SBSConfig(
        strategy=args.strategy,
        step_beam_width=args.beam_width, n_total_samples=args.n_samples, max_depth=args.max_depth,
        budget=args.budget, temperature=args.temperature, verbose=args.verbose,
        evaluate_n_steps=args.evaluate_n_steps,
        value_method=args.value_method, uncertainty_method=args.uncertainty_method,
        uncertainty_temperature=args.uncertainty_temperature,
        dual_prm_weight=getattr(args, 'dual_prm_weight', 0.5),

        remove_duplicate=not args.keep_duplicates,
        max_tokens=args.max_tokens,
    )
    sbs_instance = StepBeamSearch(
        inference_model_name=args.inference_model, 
        config=sbs_config, 
        strategy=strategy,
        value_client=value_client, 
        worker_rank=rank,
    )

    for item in tqdm(dataset_slice, desc=f"Rank {rank} Processing"):
        output_path = None
        try:
            original_index = item['original_index']
            output_path = os.path.join(args.output_dir, f"q_{original_index:04d}")
            summary_json_path = os.path.join(output_path, "summary.json")
            
            if os.path.exists(summary_json_path):
                if _is_run_valid(summary_json_path):
                    logger.info(f"[Rank {rank}] Skipping item {original_index} as it is already completed and valid.")
                    continue
                else:
                    logger.warning(f"[Rank {rank}] Item {original_index} exists but is invalid/incomplete. Overwriting.")
            
            gt = item.get("solution")
            if not gt:
                logger.warning(f"[Rank {rank}] Missing ground truth for item {original_index} (key 'solution' empty or missing). Item keys: {list(item.keys())}")
            sbs_instance.run(question=item['question'], ground_truth=gt, base_path=output_path)
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"[Rank {rank}] FATAL ERROR processing item {item.get('original_index')}: {e}\n{error_traceback}")
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                # Write summary.json with error info
                error_summary = {
                    "question": item.get('question'),
                    "ground_truth": item.get("solution"),
                    "error": f"{e}\n{error_traceback}",
                    "majority_vote": None,
                    "is_majority_correct": False,
                    "best_of_n_answer": None,
                    "is_best_of_n_correct": False,
                    "accuracy": None,
                    "solutions": [],
                    "total_tokens": 0,
                    "gt_usable": False
                }
                with open(os.path.join(output_path, "summary.json"), 'w') as f:
                    json.dump(error_summary, f, indent=4)
                
                # Also keep error.log for easy grepping
                with open(os.path.join(output_path, "error.log"), 'w') as f:
                    f.write(f"Error on question index {item.get('original_index')}:\n{e}\n\n{error_traceback}")
    logger.info(f"[Rank {rank}] Worker finished.")


def run_sbs_on_dataset(args: argparse.Namespace):
    set_start_method('spawn', force=True)
    
    ds = load_dataset(args.dataset)
    dataset = [{'original_index': i, **item} for i, item in enumerate(ds)]

    if args.idx_start is not None and args.idx_end is not None:
        dataset = dataset[args.idx_start:args.idx_end]
        logger.info(f"Processing dataset slice from {args.idx_start} to {args.idx_end}. Total items: {len(dataset)}")

    if os.path.exists(args.output_dir):
        processed_indices = set()
        for folder_name in os.listdir(args.output_dir):
            if folder_name.startswith("q_"):
                try:
                    idx = int(folder_name.split("_")[1])
                    summary_path = os.path.join(args.output_dir, folder_name, "summary.json")
                    
                    # Only exclude if the run exists AND is valid
                    if os.path.exists(summary_path):
                        if _is_run_valid(summary_path):
                            processed_indices.add(idx)
                        else:
                            logger.warning(f"Run {folder_name} exists but is INVALID. Will be re-run.")
                    else:
                        # Folder exists but no summary - treat as failed/not processed
                        logger.warning(f"Run {folder_name} exists but missing summary.json. Will be re-run.")
                        
                except (IndexError, ValueError):
                    continue
        
        original_count = len(dataset)
        dataset = [item for item in dataset if item['original_index'] not in processed_indices]
        logger.info(f"Excluded {original_count - len(dataset)} already VALID processed questions. Remaining: {len(dataset)}")

    num_workers = args.num_workers
    procs = []

    # --- Value Model Server Setup ---
    value_task_queue = Queue()
    value_result_queues = [Queue() for _ in range(num_workers)]
    logger.info("Starting Value Model Server process...")
    value_server_proc = Process(target=_value_model_server, args=(args, value_task_queue, value_result_queues))
    value_server_proc.start()
    procs.append(value_server_proc)

    # --- Uncertainty Model Server Setup (if needed) ---
    # --- Uncertainty / Second PRM Model Server Setup ---
    uncertainty_task_queue, uncertainty_result_queues = None, None
    if args.strategy in ['pum', 'mean_var']:
        uncertainty_task_queue = Queue()
        uncertainty_result_queues = [Queue() for _ in range(num_workers)]
        
        model_path = args.uncertainty_model_path if args.strategy == 'pum' else args.second_prm_model_path
        gpu_id = args.uncertainty_model_gpu if args.strategy == 'pum' else args.second_prm_model_gpu
        
        # We temporarily patch args to reuse _uncertainty_model_server which expects certain args
        # Or better yet, we pass distinct args to it if possible. 
        # _uncertainty_model_server reads args.uncertainty_model_path. 
        # Let's just create a modified args copy or handle it inside the server function.
        # Actually _uncertainty_model_server is simple, let's look at servers.py later. 
        # For now, let's assume we can repurpose it.
        # NOTE: To reuse _uncertainty_model_server without changing it too much, we will ensure it knows which model to load.
        
        logger.info(f"Starting Second Model Server process (Strategy: {args.strategy})...")
        
        # We need to make sure the server process gets the right path.
        # Since _uncertainty_model_server reads from 'args', we can't easily change 'args' here for just that process 
        # without affecting the main process args? actually multiprocessing forks/spawns.
        # A cleaner way is to create a namespace copy.
        server_args = argparse.Namespace(**vars(args))
        if args.strategy == 'mean_var':
            server_args.uncertainty_model_path = args.second_prm_model_path
            server_args.uncertainty_model_gpu = args.second_prm_model_gpu
            # It might treat it as a regression model if we repurpose it.
            # Usually uncertainty server loads a model. 
            
        uncertainty_server_proc = Process(target=_uncertainty_model_server, args=(server_args, uncertainty_task_queue, uncertainty_result_queues))
        uncertainty_server_proc.start()
        procs.append(uncertainty_server_proc)

    logger.info("Waiting for servers to load models...")
    time.sleep(30)

    # --- Worker and Client Setup ---
    total_samples = len(dataset)
    if total_samples == 0:
        logger.info("No new questions to process. Shutting down.")
        return 
    
    samples_per_worker = (total_samples + num_workers - 1) // num_workers
    logger.info(f"Starting {num_workers} SBS client workers.")
    
    worker_procs = []
    for i in range(num_workers):
        start_idx = i * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        if not (dataset_slice := dataset[start_idx:end_idx]): continue
        
        # Create clients for this worker
        value_client = ValueClient(value_task_queue, value_result_queues[i], i)
        uncertainty_client = None

        # Instantiate strategy for each worker
        # Instantiate strategy for each worker
        if args.strategy == 'pum':
            assert uncertainty_task_queue is not None and uncertainty_result_queues is not None
            uncertainty_client = UncertaintyClient(uncertainty_task_queue, uncertainty_result_queues[i], i)
            strategy = PumStrategy(uncertainty_client)
        
        elif args.strategy == 'mean_var':
            # Reuse the uncertainty client infrastructure for the second PRM
            assert uncertainty_task_queue is not None and uncertainty_result_queues is not None
            # We call it 'uncertainty_client' generically here but it points to the second PRM
            second_prm_client = UncertaintyClient(uncertainty_task_queue, uncertainty_result_queues[i], i)
            # Impor MeanVarStrategy locally or at top level (will update imports next)
            from rely.inference.sbs.strategies import MeanVarStrategy
            strategy = MeanVarStrategy(second_prm_client=second_prm_client, weight=args.dual_prm_weight)

        else: # uniform
            strategy = UniformStrategy()

        proc = Process(target=_sbs_worker, args=(args, dataset_slice, i, strategy, value_client, uncertainty_client))
        proc.start()
        worker_procs.append(proc)
    
    for proc in worker_procs:
        proc.join()
    procs.extend(worker_procs)

    # --- Shutdown ---
    logger.info("All SBS workers finished. Shutting down servers...")
    value_task_queue.put("STOP")
    if uncertainty_task_queue:
        uncertainty_task_queue.put("STOP")
    
    for proc in procs:
        if proc.is_alive():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    logger.info("All processes have completed.")


def main():
    parser = argparse.ArgumentParser(description="Run Step-level Beam Search with various strategies.")
    # Common arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--inference_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model name served by vLLM.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to pretrained value model.")
    parser.add_argument("--value_model_type", type=str, default="classification", choices=["classification", "regression"], help="Type of value model head.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel client worker processes.")
    parser.add_argument("--value_model_gpu", type=int, default=0, help="GPU ID for the value model server.")
    parser.add_argument("--idx_start", type=int, default=None, help="Start index of the dataset split.")
    parser.add_argument("--idx_end", type=int, default=None, help="End index of the dataset split.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")
    parser.add_argument("--keep_duplicates", action='store_true', help="Keep duplicate nodes instead of deduplicating them.")

    # SBS parameters
    parser.add_argument("--beam_width", type=int, default=4, help="Step-level beam width.")
    parser.add_argument("--n_samples", type=int, default=12, help="Total samples per step.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum search depth.")
    parser.add_argument("--budget", type=int, default=None, help="Maximum total generated tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--evaluate_n_steps", type=int, default=1, help="Run value/uncertainty evaluation every n generated steps (default: 1).")
    parser.add_argument("--value_method", type=str, default="last_step", choices=["last_step", "product", "maximum"], help="Method to calculate node value.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens to generate per step.")

    # Strategy selection
    parser.add_argument(
        "--sbs_strategy",
        "--strategy",
        dest="strategy",
        type=str,
        default="uniform",
        choices=["uniform", "pum", "mean_var"],
        help="SBS strategy to use.",
    )
    
    # PUM-specific
    pum_group = parser.add_argument_group('PUM Strategy Arguments')
    pum_group.add_argument("--uncertainty_model_path", type=str, help="Path to pretrained PUM uncertainty model.")
    pum_group.add_argument("--uncertainty_model_type", type=str, default="regression", choices=["classification", "regression"], help="Type of uncertainty model head.")
    pum_group.add_argument("--uncertainty_model_gpu", type=int, help="GPU ID for the uncertainty model server.")
    pum_group.add_argument("--uncertainty_method", type=str, default="last_step", choices=["last_step", "product", "average", "maximum"], help="Method to aggregate PUM uncertainty scores.")
    pum_group.add_argument("--uncertainty_temperature", type=float, default=1.0, help="Temperature for softmax normalization of uncertainty scores. Controls the skewness of the distribution.")

    # Dual-PRM specific
    dual_group = parser.add_argument_group('Dual-PRM Strategy Arguments (mean_var)')
    dual_group.add_argument("--second_prm_model_path", type=str, help="Path to the second PRM model (e.g. normalized/variance model).")
    dual_group.add_argument("--second_prm_model_gpu", type=int, help="GPU ID for the second PRM model server.")
    dual_group.add_argument("--dual_prm_weight", type=float, default=0.5, help="Weight for the second PRM score (score = mean + weight * var).")
    
    args = parser.parse_args()

    if args.strategy == 'pum':
        if not args.uncertainty_model_path or args.uncertainty_model_gpu is None:
            parser.error("--uncertainty_model_path and --uncertainty_model_gpu are required for the selected strategy.")

    if args.strategy == 'mean_var':
        if not args.second_prm_model_path or args.second_prm_model_gpu is None:
            parser.error("--second_prm_model_path and --second_prm_model_gpu are required for 'mean_var' strategy.")

    os.makedirs(args.output_dir, exist_ok=True)
    run_sbs_on_dataset(args)

if __name__ == "__main__":
    main()