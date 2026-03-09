import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trainer import RegressionPRMTrainer
from model import RegressionPRMModel
from trl.experimental.prm import PRMTrainer, PRMConfig


if __name__ == "__main__":
    """
    Main script to train a regression-based Process Reward Model (PRM)
    for predicting step-wise uncertainty (variance).
    """

    # --- Model and Dataset Configuration ---
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name = "jacopo-minniti/MMLU-PUM-qwen2.5-1.5B" # "jacopo-minniti/MMLU-PUM-qwen2.5-1.5B"
    dataset_subset = None # "value"

    # Load tokenizer and add a custom separator token for PRM steps.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    step_separator_token = "<extra_0>"
    if step_separator_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [step_separator_token]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine whether the runtime can handle bfloat16. Fall back to float32 otherwise.
    cuda_available = torch.cuda.is_available()
    bf16_supported = False
    if cuda_available:
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported):
            bf16_supported = bool(is_bf16_supported())
        else:
            major, _ = torch.cuda.get_device_capability()
            bf16_supported = major >= 8  # Ampere or newer

    compute_dtype = torch.bfloat16 if bf16_supported else torch.float32

    # Load the custom RegressionPRMModel, which wraps a base transformer model with regression head
    model = RegressionPRMModel.from_base_model(model_name, dtype=compute_dtype)
    model.resize_token_embeddings(len(tokenizer))  # Adjust for the new special token.

    # --- Dataset Loading ---
    train_dataset = load_dataset(dataset_name, dataset_subset, split="train")
    eval_dataset = load_dataset(dataset_name, dataset_subset, split="test")

    # --- Training Configuration ---
    training_args = PRMConfig(
        # --- Core ---
        output_dir=".cache/models/mmlu/qwen2.5-normalized-regression-1.5B",
        hub_model_id="jacopo-minniti/Qwen2.5-MMLU-1.5B-PRM-Normalized",
        num_train_epochs=5,
        
        # --- PRM Specific ---
        max_length=5000,
        train_on_last_step_only=False,  # Train on the value of every step, not just the final one.
        step_separator=step_separator_token,

        # --- Optimizer & Scheduler ---
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.1,
        optim="adamw_torch",

        # --- Batching & Memory ---
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,  # Set to True to save memory at the cost of a minor slowdown.
        bf16=bf16_supported,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,

        # --- Logging & Saving ---
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        push_to_hub=False,
        
        # --- Other ---
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # Can speed up DDP training.
    )

    # --- Trainer Initialization ---
    trainer = RegressionPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        value_baseline="normalized",
    )

    # --- Start Training ---
    trainer.train()

    # --- (Optional) Push to Hub ---
    # After training, you can push the model to the Hugging Face Hub.
    trainer.push_to_hub(commit_message="End of training")
    # print("Script finished successfully.")
