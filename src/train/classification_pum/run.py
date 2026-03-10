import os
import torch
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from trl import PRMConfig, PRMTrainer

# --- 1. Model and Tokenizer Setup ---
# Define the base model identifier from the YAML file
base_model_name = "Qwen/Qwen3-1.7B"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Add the special token used as a step separator
special_tokens_to_add = ["<extra_0>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})

# Check for bfloat16 support to handle `bf16: auto` from the YAML
use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# Load the model for token classification with 2 labels (correct/incorrect)
model = AutoModelForTokenClassification.from_pretrained(
    base_model_name,
    num_labels=2,
    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
    attn_implementation="flash_attention_2" 
)

# Resize token embeddings to accommodate the newly added special token
model.resize_token_embeddings(len(tokenizer))


# --- 2. Dataset Loading ---
# Load the training and testing datasets as specified in the YAML
train_dataset = load_dataset(
    "jacopo-minniti/MMLU-PUM-qwen3-1.7B",
    name="potential",
    split="train",
)
test_dataset = load_dataset(
    "jacopo-minniti/MMLU-PUM-qwen3-1.7B",
    name="potential",
    split="test",
)


# --- 3. Wandb Configuration ---
# Set up Weights & Biases integration from YAML configuration
os.environ["WANDB_PROJECT"] = "pum"
# The `wandb_entity` was empty in your YAML. If you have a specific entity, set it here:
# os.environ["WANDB_ENTITY"] = "your_entity"


# --- 4. Training Configuration (PRMConfig) ---
# Translate the YAML training parameters into a PRMConfig object
# Note: `sample_packing` is an Axolotl-specific feature and is not directly available in TRL.
# This would require a custom data collator for similar functionality.
training_args = PRMConfig(
    # --- File Paths & Data Handling ---
    output_dir=".cache/models/mmlu/potential-classification-1.7B",
    remove_unused_columns=False,
    max_length=8000,
    step_separator="<extra_0>",
    train_on_last_step_only=False,

    # --- Core Training Parameters ---
    num_train_epochs=5,
    per_device_train_batch_size=8,  # Corresponds to `micro_batch_size`
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",  # Corresponds to `lr_scheduler`
    optim="adamw_torch_fused",    # Corresponds to `optimizer`
    weight_decay=0.1,
    warmup_ratio=0.3,
    
    # --- Evaluation ---
    per_device_eval_batch_size=8, # Corresponds to `eval_batch_size`
    eval_strategy="epoch",  # `evals_per_epoch: 1` translates to evaluating each epoch

    # --- Logging & Saving ---
    logging_steps=5,
    save_strategy="epoch",      # `saves_per_epoch: 1` translates to saving each epoch

    # --- Hardware & Performance ---
    bf16=use_bfloat16,
    fp16=not use_bfloat16 and torch.cuda.is_available(),
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=False,

    # --- Hub Upload ---
    push_to_hub=False,
    # hub_model_id="jacopo-minniti/Qwen2.5-Math-1.5B-PUM-cwp_binary",
    # hub_strategy="end",

    # --- W&B Reporting ---
    report_to="wandb",
    run_name="Math-Qwen3-1.7B-potential", # Corresponds to `wandb_name`
)


# --- 5. Initialize and Run Trainer ---
# Create the PRMTrainer instance
trainer = PRMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer, # Pass the tokenizer to handle data processing
)

# Start the training process
print("🚀 Starting PRM training...")
trainer.train()
print("✅ Training finished successfully!")

# The trainer will automatically push to the Hub at the end of training
# because `push_to_hub` is True and `hub_strategy` is 'end'.