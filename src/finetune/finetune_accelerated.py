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
Finetuning script for GEMMA3 or other causal language models using HuggingFace Transformers and Accelerate.

This script is adapted from the official Transformers language modeling tutorial to support finetuning
GEMMA3 and similar models on a text file or dataset, without using the HuggingFace Trainer.

For more details and checkpoints, see:
https://huggingface.co/models?filter=text-generation
"""
# You can adapt this script for your own causal language modeling tasks. See comments for pointers.
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger, is_accelerate_available, is_datasets_available, GradientAccumulationPlugin


from torch.profiler import profile, record_function, ProfilerActivity
warnings.filterwarnings("ignore")
# import hydra TODO # to use hydra for configuration management
# from omegaconf import DictConfig # TODO # to use hydra for configuration management
import wandb
wandb.login(key="8be012e7c0ff0d4431216d5b3f309041178cacd4")

from transformers.trainer_utils import seed_worker
import argparse
import json
import logging
import math
import os
import random
import datasets
import torch
from torch.utils.data import DataLoader
from functools import partial
import transformers

from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_scheduler,
    MODEL_MAPPING,
    SchedulerType
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from src.dataset import load_parquet_image_dataset
from src.models import get_collator, configure_model_for_training
from util_finetune import evaluate, BlueprintGroupedSampler, rank0_print


os.environ["HF_TOKEN"] = "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"
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["HF_TOKEN"] = "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"
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")
logger = get_logger(__name__)
hf_token = os.environ.get("HF_TOKEN", "")


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs

#
# def parse_args():
#
#     parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
#     parser.add_argument(
#             "--dataset_name",
#             type=str,
#             default=None,
#             help="The name of the dataset to use (via the datasets library).",
#     )
#     parser.add_argument(
#             "--dataset_dir", type=str, default=None, help="Path to a directory containing the dataset files in .parquet format."
#     )
#     parser.add_argument(
#             "--model_name_or_path",
#             type=str,
#             help="Path to pretrained model or model identifier from huggingface.co/models.",
#             required=True,
#             default='google/gemma-3-4b-it',
#     )
#     parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode with fewer epochs and smaller batch size.")
#     parser.add_argument(
#             "--per_device_train_batch_size",
#             type=int,
#             default=8,
#             help="Batch size (per device) for the training dataloader.",
#     )
#     parser.add_argument(
#             "--per_device_eval_batch_size",
#             type=int,
#             default=8,
#             help="Batch size (per device) for the evaluation dataloader.",
#     )
#     parser.add_argument(
#             "--learning_rate",
#             type=float,
#             default=2e-5,
#             help="Initial learning rate (after the potential warmup period) to use.",
#     )
#     parser.add_argument(
#             "--weight_decay",
#             type=float,
#             default=0.001,
#             help="Weight decay to use.")
#
#     parser.add_argument(
#             "--num_train_epochs",
#             type=int,
#             default=3,
#             help="Total number of training epochs to perform.")
#     parser.add_argument(
#             "--max_train_steps",
#             type=int,
#             default=None,
#             help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
#     )
#     parser.add_argument(
#             "--gradient_accumulation_steps",
#             type=int,
#             default=1,
#             help="Number of updates steps to accumulate before performing a backward/update pass.",
#     )
#     parser.add_argument(
#             "--lr_scheduler_type",
#             type=SchedulerType,
#             default="linear",
#             help="The scheduler type to use.",
#             choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
#     )
#     parser.add_argument(
#             "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
#     )
#     parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
#     parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
#     parser.add_argument(
#             "--model_type",
#             type=str,
#             default=None,
#             help="Model type to use if training from scratch.",
#             choices=MODEL_TYPES,
#     )
#     parser.add_argument(
#             "--preprocessing_num_workers",
#             type=int,
#             default=None,
#             help="The number of processes to use for the preprocessing.",
#     )
#
#     parser.add_argument(
#             "--checkpointing_steps",
#             type=str,
#             default=None,
#             help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
#     )
#     parser.add_argument(
#             "--save_every_n_epochs",
#             type=int,
#             default=None,
#             help="Save a model checkpoint every n epochs (in addition to other checkpointing logic)."
#     )
#     parser.add_argument(
#             "--resume_from_checkpoint",
#             type=str,
#             default=None,
#             help="If the training should continue from a checkpoint folder.",
#     )
#     # New Code #
#     # Whether to load the best model at the end of training
#     parser.add_argument(
#             "--load_best_model",
#             action="store_true",
#             help="Whether to load the best model at the end of training",
#     )
#     parser.add_argument(
#             "--with_tracking",
#             action="store_true",
#             help="Whether to enable experiment trackers for logging.",
#     )
#     parser.add_argument("--peft_strategy",
#                         type=str, default="lora_gaussian",)
#     parser.add_argument("--rank", type=int, default=32, help="Rank for LoRA layers.")
#     parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA layers.")
#     parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA layers.")
#     parser.add_argument('--freeze_multimodal', default=True, action='store_true')
#     parser.add_argument('--finetune_vision_layers', default=False, action='store_true')
#     parser.add_argument('--finetune_language_layers', default=True, action='store_true')
#     parser.add_argument('--finetune_attention_modules', default=True, action='store_true')
#     parser.add_argument('--finetune_mlp_modules', default=True, action='store_true')

#     args = parser.parse_args()
#
#     return args
#


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/gemma-3-4b-it", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    caching_local: bool = field(default=True, metadata={"help": "Whether to cache the model locally."})
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)
    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

    lr_rate: Optional[float] = field(default=5e-5, metadata={"help": "Learning rate for the model."})


@dataclass
class DataArguments:

    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library - or - local funcion for parquet chexinstruct)."})
    dataset_dir: Optional[str] = field(default=None, metadata={"help": "Path to a directory containing the dataset files in .parquet format."})
    cache_dir: Optional[str] = field(default=os.path, metadata={"help": "Path to a directory where the dataset will be cached."})
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=CACHE_DIR, metadata={"help": "Path to a directory where the model will be cached."})
    
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    #freeze_mm_mlp_adapter: bool = field(default=False)
    #freeze_mm_vision_resampler: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for the multimodal projector."})
    mm_vision_tower_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for the vision tower."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing."})
    verbose_logging: bool = field(default=False, metadata={"help": "Whether to enable verbose logging."})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    

    mpt_attn_impl: Optional[str] = field(default="triton")
    
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
    #double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    #quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    #bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = field(default=True, metadata={"help": "Whether to enable LoRA training."})
    lora_r: int = field(default=64, metadata={"help": "Rank for LoRA layers."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_weight_path: str = field(default="", metadata={"help": "Path to LoRA weights."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias."})
    peft_strategy: str = field(default="lora_gaussian", metadata={"help": "PEFT strategy to use."})
    freeze_multimodal: bool = field(default=True, metadata={"help": "Whether to freeze the multimodal model."})
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Whether to finetune the vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Whether to finetune the language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Whether to finetune the attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Whether to finetune the MLP modules."}) 
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode with fewer epochs and smaller batch size."})
    split_batches: bool = field(default=True, metadata={"help": "Whether to split batches across multiple devices."})
# @hydra.main(version_base="v1.3", config_path="../../configs/PEFT_runs", config_name="config") # TODO # to use hydra for configuration management
def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")
    #args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    # ----------------------------------- Accelerator Initialization -----------------------------------

    grad_acc_args = {"num_steps": args.gradient_accumulation_steps}
    grad_acc_args["sync_with_dataloader"] = False
    gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_args)
    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
    rank0_print("Setting NCCL timeout to INF to avoid running errors.")

    # create accelerator object
    accelerator = Accelerator(
        log_with=args.report_to,
        deepspeed_plugin=args.deepspeed_plugin,
        project_dir=args.output_dir,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        kwargs_handlers=[accelerator_kwargs]
    )

    accelerator.init_trackers(
            project_name=f'{args.model_name_or_path.split("/")}-finetuning-{args.peft_strategy}',
            config=vars(args)  # or your config dict
    )


    # ----------------------------------- Dataset Loading -----------------------------------
    if args.dataset_dir is not None:
        if os.path.exists(args.dataset_dir + '_tok'):
            args.dataset_dir = args.dataset_dir + '_tok'
        else:
            raise ValueError('The dataset directory does not exist in tok version. Please create the format_code.')

        raw_datasets = load_parquet_image_dataset(
                dataset_dir=args.dataset_dir,
                split_list=["train", "val"],
                cache_dir=cache_dir,# Specify the split you want to use ['test', 'val', 'train']
                keep_in_memory = True,
                num_proc=args.preprocessing_num_workers,
        )
        
        if args.debug:
            # Reduce the dataset size for debugging purposes
            for split in raw_datasets.keys():
                raw_datasets[split] = raw_datasets[split].select(range(500))

    if data_args.dataset_name is None and data_args.dataset_dir is None:
        raise ValueError(
                "You need to specify either a dataset name or a dataset directory. "
                "Use --dataset_name for a HF dataset or --dataset_dir to specify the dataset folder in local (parquet)."
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["val"]

    # ---------------------- CONFIGURATION MODEL -----------------------------------
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    

    # Get the text column names for the training and evaluation datasets
    collator = get_collator(model_id=args.model_name_or_path,
                            padding_side="left",
                            token=hf_token)
    # If the model is a GEMMA3 model, we use the processor and tokenizer from the collator
    processor = collator.processor
    tokenizer = processor.tokenizer
    logger.info(f"Using processor: {processor} and tokenizer: {tokenizer}")

    # --- Load the model ---
    # For everypart of the config file, check the needed implementation

    assert args.model_name_or_path, "You need to specify a model name or path with --model_name_or_path"
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")
    elif training_args.attn_implementation == "flash_attention" and not is_accelerate_available():
        raise ValueError("The 'flash_attention' attention implementation requires the accelerate library to be installed.")

    customized_kwargs = dict()
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained




    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=CACHE_DIR
    )
    # TODO create the config for the peft in the args
    if training_args.enable_lora:
        model = configure_model_for_training(
                model,
                strategy=args.peft_strategy,
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                freeze_multimodal=args.freeze_multimodal,
                finetune_vision_layers=args.finetune_vision_layers,
                finetune_language_layers=args.finetune_language_layers,
                finetune_attention_modules=args.finetune_attention_modules,
                finetune_mlp_modules=args.finetune_mlp_modules,
        )



    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]['texts']}.")
    try:
        # Obtain the world Size:
        sampler_train = BlueprintGroupedSampler(
                batch_size=args.per_device_train_batch_size,
                world_size=accelerator.num_processes,
                lengths=train_dataset["length"],
                n_images=train_dataset["n_of_images"],
        )

        sampler_val = BlueprintGroupedSampler(
                batch_size=args.per_device_eval_batch_size,
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
                sampler= sampler_train,
                collate_fn=collator,
                drop_last=True,
                worker_init_fn=fixed_seed_worker,
                batch_size=args.per_device_train_batch_size,
                pin_memory=True,
        )
        eval_dataloader = DataLoader(
                eval_dataset,
                sampler=sampler_val,
                collate_fn=collator,
                drop_last=True,
                batch_size=args.per_device_eval_batch_size,
        )
    except Exception as e:
        logger.error(
                "There was an issue creating the dataloaders. Please check your dataset and collator configuration."
        )
        raise

    tester = next(iter(eval_dataloader))


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                    "params"      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
            },
            {
                    "params"      : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
            },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
            torch.optim.AdamW
            if accelerator.state.deepspeed_plugin is None
               or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=args.num_warmup_steps,
                num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
                optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )

    try:
        model.config.hidden_size = model.model.language_model.embed_tokens.embedding_dim
    except:
        model.config.hidden_size = 2560



    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = (
            args.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    # Load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        path = os.path.basename(args.resume_from_checkpoint)
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

    # update progress bar if resumed from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        # skip new `skip_first_batches` to skip the batches when resuming from ckpt
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            # RIMUOVI: with accelerator.accumulate(model):

            outputs = model(**batch)
            loss = outputs.loss

            # Scala la loss per gradient accumulation
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)

            # Aggiorna solo quando l'accumulation è completa
            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

                accelerator.log({
                        "train/loss": loss.item(),
                        "train/lr"  : lr_scheduler.get_last_lr()[0]
                }, step=step)
            # We keep track of the loss at each epoch
            if args.with_tracking:
                step_loss = accelerator.reduce(loss.detach().clone()).item()
                total_loss += step_loss * accelerator.gradient_accumulation_steps

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        perplexity, eval_loss = evaluate(args, model, eval_dataloader, accelerator)


        if args.with_tracking:
            accelerator.log(
                    {
                            "perplexity": perplexity,
                            "eval_loss" : eval_loss,
                            "train_loss": total_loss / len(train_dataloader),
                            "epoch"     : epoch,
                            "step"      : completed_steps,
                    },
                    step=completed_steps,
            )

        # Save every N epochs (if specified)
        if args.save_every_n_epochs is not None:
            if (epoch + 1) % args.save_every_n_epochs == 0:
                save_path = os.path.join(args.output_dir, f"epoch_every_{epoch}")
                accelerator.save_state(save_path)
                accelerator.print(f"Saved checkpoint every_n_epochs at: {save_path}")

        if isinstance(checkpointing_steps, str) and checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

        # New Code #
        # Tracks the best checkpoint and best metric
        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(args.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")


    # New Code #
    # Loads the best checkpoint after the training is finished
    if args.load_best_model:
        accelerator.load_state(best_metric_checkpoint)

    # New Code #
    # Evaluates using the best checkpoint
    perplexity, eval_loss = evaluate(args, model, eval_dataloader, accelerator)
    logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")
    if perplexity != best_metric:
        raise AssertionError(
                f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
        )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            # if args.push_to_hub: TODO # Uncomment to push the model to the HuggingFace Hub
            #     api.upload_folder(
            #         repo_id=repo_id,
            #         folder_path=args.output_dir,
            #         commit_message="End of training",
            #     )

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)
    accelerator.end_training()

    print("|/| May the force be with you! Training completed successfully.")

if __name__ == "__main__":
    main()
