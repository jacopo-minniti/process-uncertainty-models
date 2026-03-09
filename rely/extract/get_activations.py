import argparse
import json
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

def parse_args():
    parser = argparse.ArgumentParser(description="Extract model activations for a slice of a dataset.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda:0').")
    parser.add_argument("--start-index", type=int, required=True, help="The starting index of the dataset slice to process.")
    parser.add_argument("--end-index", type=int, required=True, help="The ending index of the dataset slice to process.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output .pt file.")
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit", help="Name of the model to use.")
    parser.add_argument("--input-file", type=str, default="short-completions-mmlu-qwen3-1.7B.jsonl", help="Input JSONL file.")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Configuration ---
    SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about general knowledge. Think step by step and then finish your answer with 'The correct answer is (X)' where X is the correct letter choice.

    EXAMPLE

    Question: The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the other, the detector will detect photons
    Options:
    (A) an average of 10 times, with an rms deviation of about 4
    (B) an average of 10 times, with an rms deviation of about 3
    (C) an average of 10 times, with an rms deviation of about 1
    (D) an average of 10 times, with an rms deviation of about 0.1

    ## Your Example Answer
    [...Brief explanation of the answer...] The correct answer is (B).
    """

    # --- Model Loading ---
    print(f"Loading model: {args.model_name} on device {args.device}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=30_000,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    print("Model loaded successfully.")

    # --- Data Slicing ---
    print(f"Processing indices {args.start_index} to {args.end_index} from {args.input_file}")
    with open(args.input_file, "r") as f:
        lines = f.readlines()
        lines_to_process = lines[args.start_index:args.end_index]

    # --- Main Processing Loop ---
    all_metadata = []
    all_cut_cot_activations = []

    for line in tqdm(lines_to_process, desc=f"Extracting on {args.device}"):
        ex = json.loads(line)
        
        question = ex["question"]
        cut_cot = ex["cut_cot"]
        prompt_base = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{cut_cot}"
        
        if not prompt_base.endswith("\n\n"):
            prompt_base += "\n\n"

        inputs_base = tokenizer(prompt_base, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs_base = model(**inputs_base, output_hidden_states=True)
            last_token_act_base = outputs_base.hidden_states[-2][0, -1].cpu()

        all_cut_cot_activations.append(last_token_act_base)
        all_metadata.append(ex)

    # --- Save Data ---
    data_to_save = []
    for i, meta_item in enumerate(all_metadata):
        recombined_item = meta_item.copy()
        recombined_item["cut_cot_activations"] = all_cut_cot_activations[i]
        data_to_save.append(recombined_item)

    if data_to_save:
        print(f"Saving {len(data_to_save)} items to {args.output_file}...")
        # Note: torch.save creates a binary file, .pt is a conventional extension.
        torch.save(data_to_save, args.output_file)

    print(f"Processing on {args.device} finished.")

if __name__ == "__main__":
    main()
