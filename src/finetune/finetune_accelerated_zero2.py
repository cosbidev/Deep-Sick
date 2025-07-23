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
Finetuning script for GEMMA3 or other causal language models using HuggingFace Transformers Trainer.

This script is converted from the accelerate-based version to use the standard Trainer API.
"""
import math
import pathlib
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Union, Any
import os
import json
import logging
import torch.nn as nn
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    HfArgumentParser,
)
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from accelerate.utils import GradientAccumulationPlugin
from torch.utils.data import DataLoader
from transformers import get_scheduler

from transformers.trainer_utils import seed_worker
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, logger, is_accelerate_available
# Suppress warnings
warnings.filterwarnings("ignore")

# W&B setup
#import wandb

#wandb.login(key="8be012e7c0ff0d4431216d5b3f309041178cacd4")

# Import your custom modules
from src.dataset import load_parquet_image_dataset
from src.models import get_collator, configure_model_for_training
from util_finetune import rank0_print, BlueprintGroupedSampler
from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs
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
class CustomTrainingArguments(TrainingArguments):
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
    max_train_steps: int = field(default=None, metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."})
    num_warmup_steps: int = field(
            default=1000,
            metadata={"help": "Number of warmup steps for the learning rate scheduler."}
    )
    # Multimodal training configuration
    freeze_multimodal: bool = field(default=True, metadata={"help": "Whether to freeze the multimodal model."})
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Whether to finetune the vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Whether to finetune the language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Whether to finetune the attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Whether to finetune the MLP modules."})


    # Training configuration
    verbose_logging: bool = field(default=False, metadata={"help": "Whether to enable verbose logging."})



    # Trainer defaults that match your original configuration
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    dataloader_num_workers: int = field(
            default=0,
            metadata={"help": "Number of data loader workers. Set to 0 to disable multiprocessing and avoid pickling errors."}
    )
    # If using pin memory, keep it
    dataloader_pin_memory: bool = field(default=True)
    blueprint_sampler: bool = field(default=True, metadata={"help": "Use BlueprintGroupedSampler for training."})
    # Evaluation and saving
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=10)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=False)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    # W&B integration
    report_to: str = field(default="wandb")



class CustomTrainer(Trainer):
    """Custom Trainer class to handle multimodal training specifics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def create_accelerator_and_postprocess(self):

        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")


        # create accelerator object
        self.accelerator = Accelerator(
           deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, kwargs_handlers=[accelerator_kwargs]
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def is_tp_enabled(self):
        return super().is_tp_enabled


    def compute_loss(self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,):
        """
        Custom loss computation that handles multimodal inputs.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.blueprint_sampler:
            return  BlueprintGroupedSampler(
                    batch_size=self.args.per_device_train_batch_size,
                    world_size=self.accelerator.num_processes,
                    lengths=self.train_dataset["length"],
                    n_images=self.train_dataset["n_of_images"],
            )
        else:

            return self._get_train_sampler()


    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     if (
    #             self.accelerator.state.deepspeed_plugin is None
    #             or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
    #     ):
    #         lr_scheduler = get_scheduler(
    #                 name=self.args.lr_scheduler_type,
    #                 optimizer=self.optimizer,
    #                 num_warmup_steps=self.args.num_warmup_steps,
    #                 num_training_steps=self.args.max_train_steps,
    #         )
    #     else:
    #         lr_scheduler = DummyScheduler(
    #                 optimizer, total_num_steps=self.args.max_train_steps, warmup_num_steps=self.args.num_warmup_steps
    #         )
    #
    #     self.lr_scheduler = lr_scheduler
    #
    #     return self.lr_scheduler





    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override save to handle PEFT models properly."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save the model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            # Fallback for wrapped models
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)

        # Save the tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }


        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


        num_update_steps_per_epoch = math.ceil(len(dataloader) / self.accelerator.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)


        return dataloader


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with perplexity calculation."""
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Calculate perplexity from eval_loss
        import math
        if f"{metric_key_prefix}_loss" in eval_results:
            try:
                perplexity = math.exp(eval_results[f"{metric_key_prefix}_loss"])
                eval_results[f"{metric_key_prefix}_perplexity"] = perplexity
            except OverflowError:
                eval_results[f"{metric_key_prefix}_perplexity"] = float("inf")

        return eval_results




def setup_logging(training_args):
    """Setup logging configuration."""
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    if training_args.local_rank in [-1, 0]:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()


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

    # Load configuration
    # config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    # Attention implementation check
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    # Configuration overrides
    customized_kwargs = dict()
    cfg_pretrained = None
    overwrite_config = {}
    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # Handle rope scaling and other configurations
    # if any([
    #         model_args.rope_scaling_factor is not None,
    #         model_args.rope_scaling_type is not None,
    #         model_args.mm_spatial_pool_stride is not None,
    #         model_args.mm_spatial_pool_out_channels is not None,
    #         model_args.mm_spatial_pool_mode is not None,
    #         model_args.mm_resampler_type is not None,
    # ]):
    #     cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # else:
    #     cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
    #     overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
    #     overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range
    #
    # if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
    #     overwrite_config["rope_scaling"] = {
    #             "factor": model_args.rope_scaling_factor,
    #             "type"  : model_args.rope_scaling_type,
    #     }
    #     if training_args.model_max_length is None:
    #         training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
    #         overwrite_config["max_sequence_length"] = training_args.model_max_length
    #
    # if all([
    #         model_args.mm_spatial_pool_stride is not None,
    #         model_args.mm_spatial_pool_out_channels is not None,
    #         model_args.mm_spatial_pool_mode is not None,
    #         model_args.mm_resampler_type is not None
    # ]):
    #
    #     overwrite_config.update({
    #             "mm_resampler_type"           : model_args.mm_resampler_type,
    #             "mm_spatial_pool_stride"      : model_args.mm_spatial_pool_stride,
    #             "mm_spatial_pool_out_channels": model_args.mm_spatial_pool_out_channels,
    #             "mm_spatial_pool_mode"        : model_args.mm_spatial_pool_mode,
    #     })

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

    # if model_args.freeze_backbone:
    #     model.model.requires_grad_(False)



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

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_disable()
    # Set hidden size for compatibility
    try:
        model.config.hidden_size = model.model.language_model.embed_tokens.embedding_dim
    except:
        model.config.hidden_size = 2560

    return model

def create_optimizers(model, training_args):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """






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

    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimize
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    if getattr(training_args, "adam_epsilon", None) is not None:
        lr_scheduler = get_scheduler(
                name=training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=training_args.num_warmup_steps,
                num_training_steps=training_args.max_train_steps,
        )
    else:
        # If no `adam_epsilon` is specified, we use a DummyScheduler
        lr_scheduler = DummyScheduler(
                optimizer, total_num_steps=training_args.max_train_steps, warmup_num_steps=training_args.num_warmup_steps
        )

    # TODO {if optimizer_cls.__name__ == "Adam8bit": IMPLEMENT bitsandbytes optimization for 8-bit Adam optimizer ? @secondary
    return optimizer, lr_scheduler


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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


    # Setup model and tokenizer/processor
    model = setup_model_and_config(
            model_args=model_args,
            training_args=training_args,
            data_args=data_args)


    # Get collator and tokenizer
    collator = get_collator(
            model_id=model_args.model_name_or_path,
            padding_side="left",
            token=hf_token
    )
    processor = collator.processor
    tokenizer = processor.tokenizer



    logger.info(f"Using processor: {processor} and tokenizer: {tokenizer}")

    # Log sample data
    import random
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]['texts']}.")

    # Initialize W&B run name
    if training_args.report_to == "wandb":
        run_name = f'{model_args.model_name_or_path.split("/")[-1]}-finetuning-{training_args.peft_strategy}'
        os.environ["WANDB_PROJECT"] = run_name

    # Setup callbacks
    callbacks = []
    if training_args.load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)


    # Initialize trainer
    trainer = CustomTrainer(
            optimizers=create_optimizers(model, training_args),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            callbacks=callbacks,
    )


    # Training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")

    # Check if resuming from checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and checkpoint:
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Save model
    trainer.save_model()

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final evaluation
    logger.info("***** Running final evaluation *****")
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    # Save final results
    if training_args.local_rank in [-1, 0]:
        with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
            final_results = {**metrics, **eval_results}
            json.dump(final_results, f, indent=2)

    logger.info("|/| May the force be with you! Training completed successfully.")


if __name__ == "__main__":
    main()