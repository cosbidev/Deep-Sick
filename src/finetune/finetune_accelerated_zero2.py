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
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    HfArgumentParser,
)
from accelerate import Accelerator
from accelerate.state import DistributedType
from accelerate.utils import DummyScheduler, GradientAccumulationPlugin
from torch.utils.data import DataLoader

from transformers import get_scheduler
from transformers.trainer_utils import seed_worker
from transformers.trainer import has_length, logger
# Suppress warnings
warnings.filterwarnings("ignore")

from src.dataset import load_parquet_image_dataset
from src.models import get_collator, configure_model_for_training
from util_finetune import rank0_print, BlueprintGroupedSampler, evaluate

# W&B setup
# import wandb
# wandb.login(key="8be012e7c0ff0d4431216d5b3f309041178cacd4")

# Import your custom modules

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
    output_dir: str = field(default="none", metadata={"help": "Output directory for model predictions and checkpoints."})
    with_tracking: bool = field(default=True, metadata={"help": "Whether to use tracking for the training run."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    checkpointing_steps: Optional[str] = field(default='none', metadata={"help": "Checkpointing steps, can be 'epoch', 'steps', or a number of steps."})
    save_every_n_epochs: Optional[int] = field(default=None, metadata={"help": "Save every n epochs."})
    early_stopping_patience: Optional[int] = field(default=2, metadata={"help": "Number of epochs with no improvement after which training will be stopped."})


class CustomTrainer(Trainer):
    """Custom Trainer class to handle multimodal training specifics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def create_accelerator_and_postprocess(self):

        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)



        # create accelerator object
        self.accelerator = Accelerator(
           deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag

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

    if getattr(training_args, "lr_scheduler_type", None) is not None:
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
    return optimizer, lr_scheduler


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logging(training_args)
    # ----------------------------------- Accelerator Initialization -----------------------------------
    accelerator = (
            Accelerator(
                    log_with=training_args.report_to,
                    project_dir=training_args.output_dir,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            )
            if training_args.with_tracking
            else Accelerator(
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            )
    )
    # New Code #
    accelerator.init_trackers(
            project_name=f'{model_args.model_name_or_path.split("/")}-finetuning-{training_args.peft_strategy}',

    )


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


    import random
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]['texts']}.")
    try:
        # Obtain the world Size:
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

        # Crea una versione parziale della funzione con i parametri fissi
        fixed_seed_worker = partial(seed_worker, num_workers=4, rank=accelerator.process_index)
        # New Code #
        # DataLoaders creation:
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


    tester = next(iter(eval_dataloader))

    # -- Processor and tokenizer setup --
    processor = collator.processor
    tokenizer = collator.tokenizer

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
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    optimizer, lr_scheduler = create_optimizers(model, training_args)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    best_metric = None
    best_metric_checkpoint = None

    def training_step(
        model: torch.nn.Module,
        batch: dict[str, Any],
        accelerator,
        gradient_accumulation_steps: int,
        compute_loss_fn: Optional[Callable] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

    # === Training Loop ===

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")

    # Resume
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
    else:
        starting_epoch = 0
        resume_step = None
        completed_steps = 0

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, training_args.num_train_epochs):
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
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

                if training_args.with_tracking:
                    accelerator.log({
                            "train/loss": loss.item(),
                            "train/lr"  : lr_scheduler.get_last_lr()[0],
                    }, step=completed_steps)

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
                    "train_loss": total_loss / len(train_dataloader),
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
    if training_args.load_best_model:
        accelerator.load_state(best_metric_checkpoint)

    # Final evaluation
    perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator)
    logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")

    if perplexity != best_metric:
        raise AssertionError(
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

    accelerator.end_training()

    print("|/| May the force be with you! Training completed successfully.")

    # Initialize trainer
    # trainer = CustomTrainer(
    #         optimizers=create_optimizers(model, training_args),
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         tokenizer=tokenizer,
    #         data_collator=collator,
    #         callbacks=callbacks,
    # )
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and checkpoint:
    #     train_result = trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    # trainer.save_state()
    #
    # # Save model
    # trainer.save_model()
    #
    # # Save training metrics
    # metrics = train_result.metrics
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()
    #
    # # Final evaluation
    # logger.info("***** Running final evaluation *****")
    # eval_results = trainer.evaluate()
    # trainer.log_metrics("eval", eval_results)
    # trainer.save_metrics("eval", eval_results)
    #
    #
    # # Save final results
    # if training_args.local_rank in [-1, 0]:
    #     with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
    #         final_results = {**metrics, **eval_results}
    #         json.dump(final_results, f, indent=2)

    # Training

    # Check if resuming from checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint



    logger.info("|/| May the force be with you! Training completed successfully.")


if __name__ == "__main__":
    main()