import os
import textwrap
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Optional, Union
import numpy as np
import logging

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset, features
from sklearn.metrics import mean_squared_error, r2_score
from transformers import (
    BaseImageProcessor,
    DataCollator,
    DataCollatorForTokenClassification,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from dataclasses import dataclass

from trl.experimental.prm import PRMConfig, PRMTrainer
from trl.trainer.utils import disable_dropout_in_model, generate_model_card

if is_wandb_available():
    import wandb

# Set up a logger for this module
logger = logging.getLogger(__name__)


def compute_regression_metrics(eval_pred: EvalPrediction):
    """
    Computes R2 and MSE scores for regression tasks.
    Filters out predictions where the label is -100.
    """
    predictions, labels = eval_pred
    # Filter out ignored indices
    active_predictions = predictions[labels != -100]
    active_labels = labels[labels != -100]

    if active_labels.size > 0:
        # No extra masking
        pass

    active_labels_count = len(active_labels)
    if active_labels_count == 0:
        return {"r2": 0.0, "mse": 0.0, "active_labels_count": 0}

    r2 = r2_score(active_labels, active_predictions)
    mse = mean_squared_error(active_labels, active_predictions)

    return {"r2": r2, "mse": mse, "active_labels_count": active_labels_count}


@dataclass
class DataCollatorForRegression(DataCollatorForTokenClassification):
    """
    Data collator for regression tasks that corrects the label type.
    Inherits from DataCollatorForTokenClassification but ensures labels are floats.
    """
    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        The default DataCollatorForTokenClassification assumes integer labels.
        We override it to correctly handle float labels for our regression task.
        """
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        final_batch = {}
        for k, v in batch.items():
            if k == label_name:
                # Ensure labels are float32 for MSELoss.
                final_batch[k] = torch.tensor(v, dtype=torch.float32)
            else:
                final_batch[k] = torch.tensor(v, dtype=torch.int64)
        return final_batch


class RegressionPRMTrainer(Trainer):
    """
    Initialize RegressionPRMTrainer.

    This trainer is adapted for PRM-style (Process Reward Model) training for regression tasks.
    It expects continuous, real-valued labels.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably a `RegressionPRMModel` or similar model.
        args (`PRMConfig`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default
            (`DataCollatorForTokenClassification`) will be used.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer ([`~transformers.PreTrainedTokenizerBase`], ...):
            Tokenizer used to process the data.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training.
        compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional*, defaults to `compute_regression_metrics`):
            The metrics to use for evaluation. Defaults to MSE and R2 for regression.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        value_baseline (`str`, *optional*, defaults to `"none"`):
            The baseline strategy for value labels: "none", "cot_mean", or "advantage".
    """

    _tag_names = ["trl", "prm", "regression"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[PRMConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        value_baseline: str = "none",
    ):
        
        tokenizer = processing_class

        if args.disable_dropout:
            disable_dropout_in_model(model)
        
        if compute_metrics is None:
            compute_metrics = compute_regression_metrics

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "A tokenizer must be specified when using the default DataCollatorForTokenClassification"
                )
            data_collator = DataCollatorForRegression(
                tokenizer, 
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt"
            )
        


        # Pre-process the dataset if it hasn't been tokenized yet.
        # This is done only once per process to avoid redundant work.
        if "input_ids" not in train_dataset.column_names:
            with PartialState().main_process_first():
                fn_kwargs = {
                    "tokenizer": tokenizer,
                    "step_separator": args.step_separator,
                    "max_length": args.max_length,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    "train_on_last_step_only": args.train_on_last_step_only,
                    "value_baseline": value_baseline,
                }
                train_fn_kwargs = {**fn_kwargs, "is_eval": False}
                train_dataset = train_dataset.map(
                    self.tokenize_row,
                    fn_kwargs=train_fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=train_dataset.features,
                    desc="Tokenizing train dataset",
                    # We must explicitly define the features to ensure `datasets` maps the
                    # labels as `float32`, which is crucial for our regression task.
                    features=features.Features(
                        {
                            "labels": features.Sequence(features.Value("float32")),
                            "input_ids": features.Sequence(features.Value("int64")),
                        }
                    ),
                )

                eval_fn_kwargs = {**fn_kwargs, "is_eval": True}
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        self.tokenize_row,
                        fn_kwargs=eval_fn_kwargs,
                        num_proc=args.dataset_num_proc,
                        remove_columns=eval_dataset.features,
                        desc="Tokenizing eval dataset",
                        features=features.Features(
                            {
                                "labels": features.Sequence(features.Value("float32")),
                                "input_ids": features.Sequence(features.Value("int64")),
                            }
                        ),
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def training_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], *args, **kwargs) -> torch.Tensor:
        loss = super().training_step(model, inputs)

        # Log training predictions and labels periodically on the main process.
        # if self.is_world_process_zero() and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            # self._log_predictions(model, inputs, "TRAINING STEP")

        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Run the standard evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Log a batch of evaluation predictions for qualitative analysis.
        if self.is_world_process_zero():
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            try:
                inputs = next(iter(eval_dataloader))
                inputs = self._prepare_inputs(inputs)
                # self._log_predictions(self.model, inputs, "EVALUATION STEP")
            except StopIteration:
                logger.warning("Could not get a batch from the evaluation dataloader to log predictions.")

        return metrics

    def _log_predictions(self, model: nn.Module, inputs: dict, context_str: str):
        """Helper function to log predictions and labels for a given batch."""
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["labels"]

            active_logits = logits[labels != -100]
            active_labels = labels[labels != -100]



            if active_labels.numel() > 0:
                active_labels_np = active_labels.detach().cpu().numpy()
                active_logits_np = active_logits.detach().cpu().numpy()

                log_str = f"\n--- {context_str} DEBUG (Step: {self.state.global_step}) ---"
                log_str += f"\nSample labels:      {np.round(active_labels_np[:15], 4)}"
                log_str += f"\nSample predictions: {np.round(active_logits_np[:15], 4)}"
                log_str += f"\nLabel stats:      mean={np.mean(active_labels_np):.4f}, std={np.std(active_labels_np):.4f}"
                log_str += f"\nPrediction stats: mean={np.mean(active_logits_np):.4f}, std={np.std(active_logits_np):.4f}"
                log_str += f"\n---------------------------------------------------"
                logger.info(log_str)
        model.train()

    @staticmethod
    def tokenize_row(
        features,
        tokenizer,
        step_separator,
        max_length,
        max_prompt_length,
        max_completion_length,
        train_on_last_step_only,
        is_eval,
        value_baseline: str = "none",
        evaluate_n_steps: int = 1,
    ):
        """
        Tokenize a row of the dataset for regression.

        This simplified version applies the chat template to the full conversation
        and then aligns labels with separator tokens.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain "prompt", "completions", and "labels".
            tokenizer (`PreTrainedTokenizerBase`):
                Tokenizer used to process the data.
            # ... (other args)

        Returns:
            `dict[str, list]`:
                Tokenized sequences with "input_ids" and "labels".
        """
        eval_every = max(1, int(evaluate_n_steps))

        def _format_steps_with_separator(steps: list[str], labels: list[float]) -> tuple[str, list[float]]:
            """
            Mirror SBS formatting: insert the separator only every `eval_every` steps
            (or at the last step), while keeping double newlines between steps.
            Returns the formatted assistant string and the subset of labels that align
            with the inserted separators.
            """
            formatted_parts: list[str] = []
            separator_labels: list[float] = []
            steps_since_separator = 0

            for idx, (step, label) in enumerate(zip(steps, labels), start=1):
                formatted_parts.append(step.strip())
                steps_since_separator += 1
                is_last = idx == len(steps)

                if steps_since_separator == eval_every or is_last:
                    formatted_parts.append("\n\n" + step_separator)
                    separator_labels.append(float(label))
                    steps_since_separator = 0
                elif not is_last:
                    formatted_parts.append("\n\n")

            return "".join(formatted_parts), separator_labels

        # 1. Construct the full conversation, including the assistant's multi-step response.
        assistant_response, separator_labels = _format_steps_with_separator(
            steps=features["completions"],
            labels=features["labels"],
        )

        # Apply value baseline transformation if needed
        # Apply value baseline transformation if needed
        if value_baseline in ["normalized", "cot_mean"]:
            # Standardize the labels (z-score normalization)
            if len(separator_labels) > 0:
                mean_val = float(np.mean(separator_labels))
                std_val = float(np.std(separator_labels))
                # Avoid division by zero
                if std_val < 1e-8:
                    std_val = 1.0
                separator_labels = [(lbl - mean_val) / std_val for lbl in separator_labels]
        elif value_baseline == "advantage":
            # s_t = s_t - s_{t-1}, assuming s_{-1} = 0.0
            new_labels = []
            prev_val = 0.0
            for lbl in separator_labels:
                new_labels.append(lbl - prev_val)
                prev_val = lbl
            separator_labels = new_labels

        if len(features["completions"]) >= 2:
            print("DEBUG assistant_response around separators:")
            s = assistant_response
            idx = s.find(step_separator)
            print(repr(s[idx-30: idx+30]))


        messages = [
            {"role": "user", "content": features["prompt"]},
            {"role": "assistant", "content": assistant_response},
        ]

        # 2. Use the tokenizer's chat template to get the final `input_ids`.
        #    This is the modern, robust way to handle chat model tokenization.
        input_ids = tokenizer.apply_chat_template(
            messages,
            max_length=max_length,
            truncation=True,
            add_generation_prompt=False,
        )

        # 3. Align the regression labels with the separator token.
        #    Initialize all labels to -100 (the ignore index).
        labels = [-100.0] * len(input_ids)

        separator_token_id = tokenizer.convert_tokens_to_ids(step_separator)

        separator_indices = [i for i, token_id in enumerate(input_ids) if token_id == separator_token_id]

        original_labels = [float(label) for label in separator_labels]

        # The number of separators should match the number of labels.
        # Truncation might cut some off, so we take the minimum.
        num_steps = min(len(separator_indices), len(original_labels))

        # Place the true label at the position of each separator token.
        # This teaches the model to predict the step's value at its conclusion.
        if train_on_last_step_only and not is_eval:
            if num_steps > 0:
                last_label_idx = separator_indices[num_steps - 1]
                labels[last_label_idx] = original_labels[num_steps - 1]
        else:
            for i in range(num_steps):
                label_idx = separator_indices[i]
                labels[label_idx] = original_labels[i]

        return {"input_ids": input_ids, "labels": labels}

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        if "JOB_ID" in os.environ:
            tags.add("hf_jobs")

        tags.update(self._tag_names)

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.args.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            trainer_name="PRM",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
