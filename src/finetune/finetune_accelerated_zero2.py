#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning script for GEMMA3 or other causal language models using Accelerate.
"""
import math
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union, Any, Callable
from tqdm import tqdm
import os
import json
import logging
import torch.nn as nn
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from accelerate import Accelerator
from accelerate.state import DistributedType
from torch.utils.data import DataLoader

from transformers import get_scheduler
from transformers.trainer_utils import seed_worker

# Suppress warnings
warnings.filterwarnings("ignore")

from src.dataset import load_parquet_image_dataset
from src.models import get_collator, configure_model_for_training
from util_finetune import rank0_print, BlueprintGroupedSampler, evaluate

# Environment setup
os.environ["HF_TOKEN"] = "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"
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")

logger = logging.getLogger(__name__)
hf_token = os.environ.get("HF_TOKEN", "")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
            default="google/gemma-3-4b-it",
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    caching_local: bool = field(default=True, metadata={"help": "Whether to cache the model locally."})
    model_class_name: Optional[str] = field(
            default=None,
            metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"}
    )

    mm_tunable_parts: Optional[str] = field(
            default=None,
            metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", etc.'}
    )
    freeze_backbone: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)
    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=2048)
    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
            default=None,
            metadata={"help": "The name of the dataset to use (via the datasets library - or - local function for parquet chexinstruct)."}
    )
    dataset_dir: Optional[str] = field(
            default=None,
            metadata={"help": "Path to a directory containing the dataset files in .parquet format."}
    )
    data_path: str = field(
            default=None,
            metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)
    data_debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode with fewer epochs and smaller batch size."})
    preprocessing_num_workers: Optional[int] = field(
            default=None,
            metadata={"help": "The number of processes to use for the preprocessing."}
    )
    cache_dir: Optional[str] = field(default=CACHE_DIR, metadata={"help": "Path to a directory where the model will be cached."})


@dataclass
class CustomTrainingArguments:
    # Basic training arguments
    output_dir: str = field(default="./results", metadata={"help": "Output directory for model predictions and checkpoints."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed that will be set at the beginning of training."})

    # Model configuration
    model_max_length: int = field(
            default=2048,
            metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    # LoRA/PEFT configuration
    lora_enable: bool = field(default=True, metadata={"help": "Whether to enable LoRA training."})
    lora_r: int = field(default=64, metadata={"help": "Rank for LoRA layers."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_weight_path: str = field(default="", metadata={"help": "Path to LoRA weights."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias."})
    peft_strategy: str = field(default="lora_gaussian", metadata={"help": "PEFT strategy to use."})

    # Training configuration
    max_train_steps: Optional[int] = field(default=None, metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."})
    num_warmup_steps: int = field(
            default=1000,
            metadata={"help": "Number of warmup steps for the learning rate scheduler."}
    )
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use."})

    # Multimodal training configuration
    freeze_multimodal: bool = field(default=True, metadata={"help": "Whether to freeze the multimodal model."})
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Whether to finetune the vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Whether to finetune the language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Whether to finetune the attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Whether to finetune the MLP modules."})

    # Training configuration
    verbose_logging: bool = field(default=False, metadata={"help": "Whether to enable verbose logging."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})

    # Checkpointing and saving
    checkpointing_steps: Optional[str] = field(default='none', metadata={"help": "Checkpointing steps, can be 'epoch', 'steps', or a number of steps."})
    save_every_n_epochs: Optional[int] = field(default=None, metadata={"help": "Save every n epochs."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to checkpoint to resume from."})

    # Evaluation and best model
    load_best_model: bool = field(default=False, metadata={"help": "Whether to load the best model at the end of training."})

    # W&B integration
    report_to: str = field(default="wandb")
    with_tracking: bool = field(default=True, metadata={"help": "Whether to use tracking for the training run."})

    @property
    def train_batch_size(self) -> int:
        """Total effective batch size for training."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def setup_logging(training_args):
    """Setup logging configuration."""
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()


def load_and_prepare_datasets(data_args):
    """Load and prepare the datasets."""
    if data_args.dataset_dir is not None:
        if os.path.exists(data_args.dataset_dir + '_tok'):
            data_args.dataset_dir = data_args.dataset_dir + '_tok'
        else:
            raise ValueError('The dataset directory does not exist in tok version. Please create the format_code.')

        raw_datasets = load_parquet_image_dataset(
                dataset_dir=data_args.dataset_dir,
                split_list=["train", "val"],
                cache_dir=data_args.cache_dir,
                keep_in_memory=True,
                num_proc=data_args.preprocessing_num_workers,
        )

        if data_args.data_debug:
            # Reduce the dataset size for debugging purposes
            for split in raw_datasets.keys():
                raw_datasets[split] = raw_datasets[split].select(range(500))

    elif data_args.dataset_name is None:
        raise ValueError(
                "You need to specify either a dataset name or a dataset directory. "
                "Use --dataset_name for a HF dataset or --dataset_dir to specify the dataset folder in local (parquet)."
        )

    return raw_datasets["train"], raw_datasets["val"]


def setup_model_and_config(model_args, training_args, data_args):
    """Setup model configuration and load the model."""
    assert model_args.model_name_or_path, "You need to specify a model name or path"

    # Attention implementation check
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    # Configuration overrides
    customized_kwargs = dict()
    overwrite_config = {}
    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if overwrite_config:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=data_args.cache_dir,
            **customized_kwargs
    )

    # Configure PEFT/LoRA
    if training_args.lora_enable:
        model = configure_model_for_training(
                model,
                strategy=training_args.peft_strategy,
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                freeze_multimodal=training_args.freeze_multimodal,
                finetune_vision_layers=training_args.finetune_vision_layers,
                finetune_language_layers=training_args.finetune_language_layers,
                finetune_attention_modules=training_args.finetune_attention_modules,
                finetune_mlp_modules=training_args.finetune_mlp_modules,
        )
        for name, param in model.named_parameters():
            if param.requires_grad and (not hasattr(param, "grad") or param.grad is None):
                if hasattr(param, "shape") and 0 in param.shape:
                    print(f"[DISABLE] {name}: shape={param.shape}, grad=None")
                    param.requires_grad = False

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_disable()

    # Set hidden size for compatibility
    try:
        model.config.hidden_size = model.model.language_model.embed_tokens.embedding_dim
    except:
        model.config.hidden_size = 2560

    return model


def create_optimizers(model, training_args):
    """Setup the optimizer and learning rate scheduler."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                    "params"      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": training_args.weight_decay,
            },
            {
                    "params"      : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
            },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    if training_args.max_train_steps is not None:
        lr_scheduler = get_scheduler(
                name=training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=training_args.num_warmup_steps,
                num_training_steps=training_args.max_train_steps,
        )
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler


def training_step(
        model: torch.nn.Module,
        batch: dict[str, Any],
        accelerator: Accelerator,
        gradient_accumulation_steps: int,
        compute_loss_fn: Optional[Callable] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute a single training step."""
    model.train()

    with torch.set_grad_enabled(True):
        if compute_loss_fn is not None:
            loss = compute_loss_fn(model, batch, num_items_in_batch)
        else:
            outputs = model(**batch)
            loss = outputs.loss

    # Normalize loss unless DeepSpeed handles it
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        loss = loss / gradient_accumulation_steps

    accelerator.backward(loss)

    return loss.detach()


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)


    # ----------------------------------- Accelerator Initialization -----------------------------------
    # Initialize accelerator ONCE at the beginning
    accelerator = Accelerator(
            log_with=training_args.report_to if training_args.with_tracking else None,
            project_dir=training_args.output_dir,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    # Initialize tracking if enabled
    if training_args.with_tracking:
        accelerator.init_trackers(
                project_name=f'{model_args.model_name_or_path.split("/")[-1]}-finetuning-{training_args.peft_strategy}',
        )

    # Setup logging
    setup_logging(training_args)

    # Verbose logging
    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n")
        rank0_print(f"data_args = {vars(data_args)}\n")
        rank0_print(f"training_args = {vars(training_args)}\n")

    # Set seed for reproducibility
    if training_args.seed is not None:
        transformers.set_seed(training_args.seed)

    # Load datasets
    train_dataset, eval_dataset = load_and_prepare_datasets(data_args)

    # Setup model
    model = setup_model_and_config(
            model_args=model_args,
            training_args=training_args,
            data_args=data_args
    )

    # Get collator and tokenizer
    collator = get_collator(
            model_id=model_args.model_name_or_path,
            padding_side="left",
            token=hf_token
    )

    # Log sample data
    import random
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]['texts']}.")

    # Create data samplers and loaders
    try:
        sampler_train = BlueprintGroupedSampler(
                batch_size=training_args.per_device_train_batch_size,
                world_size=accelerator.num_processes,
                lengths=train_dataset["length"],
                n_images=train_dataset["n_of_images"],
        )

        sampler_val = BlueprintGroupedSampler(
                batch_size=training_args.per_device_eval_batch_size,
                world_size=accelerator.num_processes,
                lengths=eval_dataset["length"],
                n_images=eval_dataset["n_of_images"],
        )

        # Create partial function for seed worker
        fixed_seed_worker = partial(seed_worker, num_workers=4, rank=accelerator.process_index)

        # DataLoaders creation
        train_dataloader = DataLoader(
                train_dataset,
                sampler=sampler_train,
                collate_fn=collator,
                drop_last=True,
                worker_init_fn=fixed_seed_worker,
                batch_size=training_args.per_device_train_batch_size,
                pin_memory=True,
        )
        eval_dataloader = DataLoader(
                eval_dataset,
                sampler=sampler_val,
                collate_fn=collator,
                drop_last=True,
                batch_size=training_args.per_device_eval_batch_size,
        )
    except Exception as e:
        logger.error(
                "There was an issue creating the dataloaders. Please check your dataset and collator configuration."
        )
        raise


    # Processor and tokenizer setup
    processor = collator.processor

    logger.info(f"Using processor: {processor}")

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizers(model, training_args)

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
    )

    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)

    # Initialize W&B run name
    if training_args.report_to == "wandb":
        run_name = f'{model_args.model_name_or_path.split("/")[-1]}-finetuning-{training_args.peft_strategy}'
        os.environ["WANDB_PROJECT"] = run_name

    # Progress bar setup
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    best_metric = None
    best_metric_checkpoint = None

    # === Training Loop ===
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")

    # Resume handling
    starting_epoch = 0
    resume_step = None
    completed_steps = 0

    if training_args.resume_from_checkpoint:
        accelerator.load_state(training_args.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
        path = os.path.basename(training_args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // num_update_steps_per_epoch
            resume_step -= starting_epoch * num_update_steps_per_epoch
            completed_steps = resume_step

    progress_bar.update(completed_steps)

    # Training loop
    for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0 if training_args.with_tracking else None

        active_dataloader = (
                accelerator.skip_first_batches(train_dataloader, resume_step)
                if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None
                else train_dataloader
        )

        for step, batch in enumerate(active_dataloader):
            loss = training_step(
                    model=model,
                    batch=batch,
                    accelerator=accelerator,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps
            )

            # Gradient accumulation
            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

                if training_args.with_tracking:
                    log_data = {
                            "train/loss": loss.item(),
                    }
                    if lr_scheduler is not None:
                        log_data["train/lr"] = lr_scheduler.get_last_lr()[0]

                    accelerator.log(log_data, step=completed_steps)

            if training_args.with_tracking:
                step_loss = accelerator.reduce(loss.clone()).item()
                total_loss += step_loss * training_args.gradient_accumulation_steps

            # Checkpointing every N steps
            if isinstance(training_args.checkpointing_steps, int):
                if completed_steps % training_args.checkpointing_steps == 0:
                    output_dir = os.path.join(training_args.output_dir, f"step_{completed_steps}")
                    accelerator.save_state(output_dir)

            if completed_steps >= training_args.max_train_steps:
                break

        # Evaluation
        perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator)

        if training_args.with_tracking:
            accelerator.log({
                    "perplexity": perplexity,
                    "eval_loss" : eval_loss,
                    "train_loss": total_loss / len(train_dataloader) if total_loss is not None else 0,
                    "epoch"     : epoch,
                    "step"      : completed_steps,
            }, step=completed_steps)

        # Epoch-level checkpointing
        if training_args.save_every_n_epochs is not None:
            if (epoch + 1) % training_args.save_every_n_epochs == 0:
                save_path = os.path.join(training_args.output_dir, f"epoch_every_{epoch}")
                accelerator.save_state(save_path)
                accelerator.print(f"Saved checkpoint every_n_epochs at: {save_path}")

        if isinstance(training_args.checkpointing_steps, str) and training_args.checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(training_args.output_dir, f"epoch_{epoch}"))

        # Save best model
        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(training_args.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

    # Reload best checkpoint
    if training_args.load_best_model and best_metric_checkpoint:
        accelerator.load_state(best_metric_checkpoint)

    # Final evaluation
    perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator)
    logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")

    if best_metric and perplexity != best_metric:
        logger.warning(
                f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
        )

    # Save final model and tokenizer
    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
                training_args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)

        with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)

    # End tracking
    if training_args.with_tracking:
        accelerator.end_training()

    logger.info("|/| May the force be with you! Training completed successfully.")


if __name__ == "__main__":
    main()