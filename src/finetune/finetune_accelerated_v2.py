#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.

"""
Fixed version of finetuning script for GEMMA3 or other causal language models using Accelerate with DeepSpeed Zero-2.
This version addresses the "torch.cat(): expected a non-empty list of Tensors" error.
"""
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable
from tqdm import tqdm
import os
import json
import logging
from accelerate.utils import DummyOptim, DummyScheduler
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
# Suppress warnings
warnings.filterwarnings("ignore")

import sys
sys.path.extend(["./src", './'])
from src.dataset import load_parquet_image_dataset
from src.models import get_collator, configure_model_for_training
from src.distributed import checkpoint_save_with_sync, safe_wait_for_everyone_simple
from util_finetune import rank0_print, BlueprintGroupedSampler, evaluate
from torch.utils.data.distributed import DistributedSampler
# Environment setup
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
    deepspeed_config_file: str = field(default="./configs/deepspeed_zero2.json", metadata={"help": "Path to the DeepSpeed Zero-2 configuration file."})
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
    bf16: bool = field(default=False, metadata={"help": "Whether to use bfloat16 mixed precision."})


    # Scheduler configuration
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use. Options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, or cyclic."})
    warmup_min_lr: float = field(default=0.0, metadata={"help": "Warmup minimum learning rate."})
    num_warmup_steps: int = field(default=None, metadata={"help": "Number of warmup steps for the learning rate scheduler."})
    cycle_min_lr: float = field(default=1e-7, metadata={"help": "Minimum learning rate for cyclical learning rate scheduler."})
    total_num_steps: Optional[int] = field( default=None, metadata={"help": "Total number of steps for the learning rate scheduler."})
    warmup_ratio: float = field(default=0.01, metadata={"help": "Warmup ratio for the learning rate scheduler."})

    # Optimizer configuration
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer."})
    adam_beta2: float = field(default=0.99, metadata={"help": "Beta2 for AdamW optimizer."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})


    # Multimodal training configuration
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Whether to finetune the vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Whether to finetune the language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Whether to finetune the attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Whether to finetune the MLP modules."})

    # Training configuration
    verbose_logging: bool = field(default=False, metadata={"help": "Whether to enable verbose logging."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})

    # Checkpointing and saving
    checkpointing_strategy: Optional[str] = field(default='epoch', metadata={"help": "Checkpointing steps, can be 'epoch', 'steps', or a number of steps."})
    checkpointing_divider: Optional[int] = field(default=1, metadata={"help": "Checkpointing steps, can be 'epoch', 'steps', or a number of steps."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to checkpoint to resume from."})

    # Evaluation and best model
    load_best_model: bool = field(default=True, metadata={"help": "Whether to load the best model at the end of training."})
    eval_steps: int = field(default=None, metadata={"help": "Evaluate every n steps."})
    # W&B integration
    report_to: str = field(default="wandb")
    with_tracking: bool = field(default=True, metadata={"help": "Whether to use tracking for the training run."})
    lr: float = field(default=2e-4, metadata={"help": "Learning rate for the optimizer."})
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode with fewer epochs and smaller batch size."})

    @property
    def train_batch_size(self) -> int:
        """Total effective batch size for training."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


# Replace the accelerator initialization section in your finetune_accelerated_v2.py
# Around line 551 where the error occurs

def initialize_accelerator_safely(training_args):
    """
    Initialize accelerator with proper error handling for SLURM + TorchRun.
    """
    import os
    import time
    import random

    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        print("🔄 Detected TorchRun environment, using simple accelerator initialization...")

        # Simple initialization that works with torchrun
        accelerator = Accelerator(
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                log_with=training_args.report_to if training_args.with_tracking else None,
                project_dir=training_args.output_dir,
                mixed_precision="bf16" if training_args.bf16 else "fp16"

        )
        print("✅ TorchRun accelerator initialized successfully")
        return accelerator

    # Fallback for accelerate launch (with retry logic)
    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        try:
            print(f"🔄 Initializing Accelerator (attempt {attempt + 1}/{max_retries})...")

            # Add a small random delay to prevent simultaneous initialization
            if attempt > 0:
                delay = base_delay + random.uniform(0, 5)
                print(f"⏳ Waiting {delay:.1f}s before retry...")
                time.sleep(delay)

            accelerator = Accelerator(
                    num_processes=4,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                    log_with=training_args.report_to if training_args.with_tracking else None,
                    project_dir=training_args.output_dir,
                    mixed_precision="bf16" if training_args.bf16 else "fp16"
            )

            print("✅ Accelerator initialized successfully")
            return accelerator

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")

            if "EADDRINUSE" in str(e) or "address already in use" in str(e):
                print("🔍 Port conflict detected, will retry with delay...")
                if attempt == max_retries - 1:
                    print("💡 Try running: pkill -f 'python.*finetune' to clean up any hanging processes")
                    raise RuntimeError(
                            "Failed to initialize accelerator after multiple attempts. "
                            "This is likely due to port conflicts from previous runs. "
                            "Please check for hanging processes and try again."
                    )
            else:
                # For non-port related errors, fail immediately
                raise

    raise RuntimeError("Failed to initialize accelerator after all retries")



# Then in your main() function, replace this line:
# accelerator = Accelerator(...)
#
# With:
# accelerator = initialize_accelerator_safely(training_args)
def parse_args_flexible():
    """
    Flexible argument parsing that handles missing arguments gracefully.
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))

    # Check if we're running from command line with arguments
    import sys
    if len(sys.argv) > 1:
        try:
            return parser.parse_args_into_dataclasses(return_remaining_strings=True)
        except Exception as e:
            logger.warning(f"Failed to parse command line arguments: {e}")
            logger.info("Falling back to default arguments")

    # Use defaults if no command line args or parsing failed
    return ModelArguments(), DataArguments(), CustomTrainingArguments(), []


def debug_tensor_shapes(model, prefix=""):
    """Debug function to identify empty tensors that might cause issues."""
    empty_tensors = []
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

        if hasattr(param, 'shape') and any(dim == 0 for dim in param.shape):
            empty_tensors.append((name, param.shape))
            print(f"❌ EMPTY TENSOR: {prefix}{name}: {param.shape}")
        elif param.requires_grad:
            print(f"✅ TRAINABLE: {prefix}{name}: {param.shape}")

    print(f"\n📊 Parameter Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

    return empty_tensors


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
    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=data_args.cache_dir,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
            **customized_kwargs
    )
    # TEST
    training_args.layer_to_unfreeze = ["lm_head", "multi_modal_projector"]
    print("✅ Base model loaded successfully")


    # CRITICAL: Configure PEFT/LoRA BEFORE any other operations
    if training_args.lora_enable:
        print("🔄 Applying LoRA configuration...")

        model = configure_model_for_training(
                model,
                strategy=training_args.peft_strategy,
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                layer_to_unfreeze=training_args.layer_to_unfreeze,
                finetune_vision_layers=training_args.finetune_vision_layers,
                finetune_language_layers=training_args.finetune_language_layers,
                finetune_attention_modules=training_args.finetune_attention_modules,
                finetune_mlp_modules=training_args.finetune_mlp_modules,
        )

        # Debug: Check model state after LoRA
        #print("📊 Model state AFTER LoRA:")
        #debug_tensor_shapes(model, "AFTER_LORA: ")

        # Verify that we have trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("❌ No trainable parameters found after LoRA configuration!")

        print(f"✅ LoRA applied successfully with {len(trainable_params)} trainable parameter groups")

    # # Disable gradient checkpointing to avoid conflicts with DeepSpeed
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_disable()

    # Set hidden size for compatibility
    try:
        model.config.hidden_size = model.model.language_model.embed_tokens.embedding_dim
    except:
        model.config.hidden_size = 2560

    return model



def create_optimizers_with_parameter_groups(model, training_args, accelerator):
    """
    Create optimizer with proper parameter grouping to avoid empty tensor lists.
    This is critical for DeepSpeed Zero-2 compatibility.
    """
    print("🔄 Creating optimizer with parameter groups...")

    # Separate LoRA parameters from base model parameters
    lora_params = []
    base_params = []
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_A' in name or 'lora_B' in name or 'lora_embedding_A' in name or 'lora_embedding_B' in name:
                lora_params.append((name, param))
            else:
                base_params.append((name, param))

    print(f"📊 Parameter breakdown:")
    print(f"   LoRA parameters: {len(lora_params)}")
    print(f"   Base parameters: {len(base_params)}")

    # Create parameter groups with proper filtering
    optimizer_grouped_parameters = []

    # LoRA parameters (typically no weight decay)
    if lora_params:
        lora_params_wd = [p for n, p in lora_params if not any(nd in n for nd in no_decay)]
        lora_params_no_wd = [p for n, p in lora_params if any(nd in n for nd in no_decay)]

        if lora_params_wd:
            optimizer_grouped_parameters.append({
                    "params"      : lora_params_wd,
                    "weight_decay": training_args.weight_decay * 0.1,  # Reduced weight decay for LoRA
                    "lr"          : training_args.learning_rate,
            })

        if lora_params_no_wd:
            optimizer_grouped_parameters.append({
                    "params"      : lora_params_no_wd,
                    "weight_decay": 0.0,
                    "lr"          : training_args.learning_rate,
            })

    # Base model parameters (if any are trainable)
    if base_params:
        base_params_wd = [p for n, p in base_params if not any(nd in n for nd in no_decay)]
        base_params_no_wd = [p for n, p in base_params if any(nd in n for nd in no_decay)]

        if base_params_wd:
            optimizer_grouped_parameters.append({
                    "params"      : base_params_wd,
                    "weight_decay": training_args.weight_decay,
                    "lr"          : training_args.learning_rate * 0.1,  # Lower LR for base params
            })

        if base_params_no_wd:
            optimizer_grouped_parameters.append({
                    "params"      : base_params_no_wd,
                    "weight_decay": 0.0,
                    "lr"          : training_args.learning_rate * 0.1,
            })

    # Verify we have parameter groups
    if not optimizer_grouped_parameters:
        raise RuntimeError("❌ No parameter groups created for optimizer!")

    # Count total parameters in groups
    total_params_in_groups = sum(len(group["params"]) for group in optimizer_grouped_parameters)
    print(f"📊 Created {len(optimizer_grouped_parameters)} parameter groups with {total_params_in_groups} total parameters")


    optimizer_cls = (
            torch.optim.AdamW
            if accelerator.state.deepspeed_plugin is None
               or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters,
                              lr=training_args.learning_rate,
                              eps=training_args.adam_epsilon,
                              betas=(training_args.adam_beta1, training_args.adam_beta2))

    # Create learning rate scheduler

    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
                name=training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=training_args.num_warmup_steps * accelerator.num_processes,
                num_training_steps=training_args.max_train_steps * accelerator.num_processes,
                scheduler_specific_kwargs = dict(num_cycles=training_args.num_train_epochs * 2)
        )
    else:
        lr_scheduler = DummyScheduler(
                optimizer, total_num_steps=training_args.max_train_steps, warmup_num_steps=training_args.num_warmup_steps
        )
    print("✅ Optimizer and scheduler created successfully")
    return optimizer, lr_scheduler


def call_forward_dummy(dummy_inputs, model, device="cuda"):
    model.train()

    # Copia e manda su GPU tutti i tensori
    dummy = {k: (v.to(device) if hasattr(v, "to") else v)
             for k, v in dummy_inputs.items()}

    # Aggiungi labels dummy per calcolare la loss
    dummy["labels"] = dummy["input_ids"].clone()

    # Esegui forward
    out = model(**dummy)
    loss = out.loss
    return loss


def save_complete_checkpoint(
        model, accelerator,
        save_path, tokenizer=None, logger=None,
        additional_info=None
        ):
    """
    Save complete checkpoint: model + optimizer + scheduler + tokenizer
    Fast and distributed-training safe
    """

    if accelerator.is_main_process:
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"💾 Saving complete checkpoint to: {save_path}")


        model.save_pretrained(
                save_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=True  # Use safetensors (faster & safer)
        )

        # 4. Save Tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)

        # 5. Save Training Info (step, epoch, metrics, etc.)
        training_info = {
                "step"         : additional_info.get("step", 0) if additional_info else 0,
                "epoch"        : additional_info.get("epoch", 0) if additional_info else 0,
                "best_metric"  : additional_info.get("best_metric") if additional_info else None,
                "training_args": additional_info.get("training_args") if additional_info else None
        }

        with open(os.path.join(save_path, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2, default=str)

        if logger:
            logger.info(f"✅ Complete checkpoint saved successfully")
            logger.info(f"   📁 Directory: {save_path}")
            logger.info(f"   🏗️  Model: ✓")
            logger.info(f"   🔤 Tokenizer: {'✓' if tokenizer else '✗'}")

    # Synchronize all processes
    accelerator.wait_for_everyone()
    return True
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
    # Parse arguments with flexible handling
    model_args, data_args, training_args, remaining = parse_args_flexible()

    print("🚀 Starting DeepSpeed Zero-[2/3] Training with LoRA")
    print("=" * 60)

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
    print("🔄 Loading datasets...")
    train_dataset, eval_dataset = load_and_prepare_datasets(data_args)
    if training_args.debug:
        # Reduce dataset size for debugging purposes
        train_dataset = train_dataset.select(range(10000))
        eval_dataset = eval_dataset.select(range(1000))


    print("✅ Datasets loaded successfully")

    # CRITICAL: Setup model and LoRA BEFORE accelerator initialization
    print("🔄 Setting up model with LoRA...")
    model = setup_model_and_config(
            model_args=model_args,
            training_args=training_args,
            data_args=data_args
    )
    print("✅ Model and LoRA setup completed")
    # Get collator and tokenizer
    print("🔄 Setting up collator...")
    collator = get_collator(
            model_id=model_args.model_name_or_path,
            padding_side="left",
            token=hf_token
    )

    processor = collator.processor
    tokenizer = collator.tokenizer if hasattr(collator, 'tokenizer') else processor
    print("✅ Collator setup completed")

    # NOW initialize accelerator with DeepSpeed config
    accelerator = initialize_accelerator_safely(training_args=training_args)
    print("✅ Accelerator initialized successfully")

    # Calculate training steps
    if accelerator.num_processes == 0:
        raise RuntimeError("❌ No processes found. Please ensure you are running with multiple GPUs or in a distributed environment.")
    print(f"🔄 Accelerator initialized with {accelerator.num_processes} processes")
    # Create data loaders BEFORE accelerator initialization
    print("🔄 Creating data loaders...")
    try:
        sampler_train = BlueprintGroupedSampler(
                batch_size=training_args.per_device_train_batch_size,
                lengths=train_dataset["length"],
                n_images=train_dataset["n_of_images"],
                seed=42
        )
        train_dataloader = DataLoader(
                train_dataset,
                sampler=sampler_train,
                collate_fn=collator,
                drop_last=True,
                batch_size=training_args.per_device_train_batch_size,
                pin_memory=True
        )
        # # Prima di creare il DataLoader:
        # if accelerator.num_processes > 1:
        #     train_sampler = DistributedSampler(
        #             train_dataset,
        #             num_replicas=accelerator.num_processes,
        #             rank=accelerator.process_index,
        #             shuffle=True
        #     )
        #
        # else:
        #     train_sampler = None
        #     eval_sampler = None
        #     shuffle = True
        #
        # train_dataloader = DataLoader(
        #         train_dataset,
        #         sampler=train_sampler,
        #         collate_fn=collator,
        #         drop_last=True,
        #         batch_size=training_args.per_device_train_batch_size,
        #         pin_memory=True,
        #         shuffle=shuffle
        # )


        eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False
        )
        eval_dataloader = DataLoader(
                eval_dataset,
                sampler=eval_sampler,
                collate_fn=collator,
                drop_last=True,
                batch_size=training_args.per_device_eval_batch_size,
                pin_memory=True,
                shuffle=False
        )
        print("✅ Data loaders created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating data loaders: {e}")
        raise

    #batch
    # _size_real = training_args.per_device_train_batch_size * accelerator.num_processes

    # CORREZIONE: Calcolo corretto degli step
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (training_args.gradient_accumulation_steps * accelerator.num_processes))
    if training_args.max_train_steps is None:
        training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    if training_args.num_warmup_steps is None:
        if training_args.warmup_ratio is not None:

            training_args.num_warmup_steps = int(training_args.warmup_ratio * training_args.max_train_steps)
        else:
            # Default 1% del totale degli step di training (non per epoca!)
            training_args.num_warmup_steps = int(training_args.max_train_steps * 0.02)
    if not training_args.eval_steps:
        training_args.eval_steps = int(num_update_steps_per_epoch / 8)

    # Debug dettagliato dei calcoli
    print(f"   Raw calculation: {len(train_dataloader)} ÷ {training_args.gradient_accumulation_steps} = {len(train_dataloader) / training_args.gradient_accumulation_steps}")
    print(f"   Ceiled result: {math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)}")

    if accelerator.is_main_process:
        print(f"📊 TRAINING PRE-ACCELERATOR PREPARE DEBUG:")
        print(f"   Train dataset size: {len(train_dataloader)}")
        print(f"   Train size computed: ( N_batch * B_size ) {len(train_dataloader) * training_args.per_device_train_batch_size}")
        print(f"   Eval dataset size: {len(eval_dataloader)}")
        print(f"   Train size computed: ( N_batch * B_size ) {len(train_dataloader) * training_args.per_device_eval_batch_size}")
        print(f"   Batch size per device ( Train, Eval ): {training_args.per_device_train_batch_size}, {training_args.per_device_eval_batch_size}")
        print(f"   Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"   Total number of processes: {accelerator.num_processes}")
        print(f"   Total number of GPUs: {torch.cuda.device_count()}")
        print(f"   Total number of training epochs: {training_args.num_train_epochs}")
        print(f"   Total number of warmup steps: {training_args.num_warmup_steps}")
        print(f"   Total number of training steps x Epoch: {num_update_steps_per_epoch}")

    # Create optimizer with proper parameter grouping BEFORE accelerator.prepare()
    optimizer, lr_scheduler = create_optimizers_with_parameter_groups(model, training_args, accelerator)
    print("✅ Optimizer created successfully")

    # Initialize tracking if enabled
    if training_args.with_tracking:
        # Config personalizzato per W&B
        wandb_config = {
                "model"        : model_args.model_name_or_path,
                "batch_size"   : training_args.per_device_train_batch_size,
                "accumulation" : training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "lora_r"       : training_args.lora_r,
                "lora_alpha"   : training_args.lora_alpha,
                "epochs"       : training_args.num_train_epochs,
                "processes"    : accelerator.num_processes
        }


        accelerator.init_trackers(
                project_name=f'{model_args.model_name_or_path.split("/")[-1]}-multinode',
                config=wandb_config,
                init_kwargs={
                        "wandb": {
                                "name": f'gemma3-multinode-{datetime.now().strftime("%m%d-%H%M")}',
                                "tags": ["multinode", "gemma3", "lora", "deepspeed"]
                        }
                }
        )

    print("✅ Accelerator initialized successfully")
    # Final parameter check before prepare
    print("📊 Final parameter check before accelerator.prepare():")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"   Trainable parameters: {len(trainable_params)}")

    # # Verify optimizer has parameters
    print("🔄 Preparing model, optimizer, and data loaders with accelerator...")

    try:
        print("🔄 Enabling gradient checkpointing if supported by the model...")
        hasattr(model, 'gradient_checkpointing_enable')
        model.config.use_cache = False
        model.gradient_checkpointing_enable(dict(use_reentrant=False))
    except AttributeError as e:
        logger.warning(f"❌ Failed to enable gradient checkpointing: {e}")
        logger.warning("This might be due to model not supporting gradient checkpointing or already enabled.")
        # Continue without enabling gradient checkpointing if it fails
    try:

        # Prepare in the correct order: model first, then optimizer, then data loaders
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
                model,
                optimizer,
                lr_scheduler,
                train_dataloader,
                eval_dataloader,
        )
        print("✅ All components prepared successfully with accelerator")

    except Exception as e:
        logger.error(f"❌ Error during accelerator.prepare(): {e}")
        logger.error("This is likely due to empty parameter groups or incompatible configurations")
        raise

    if accelerator.is_main_process:
        print(f"📊 TRAINING CALCULATION VERIFICATION:")
        print("AFTER-ACCELERATOR PREPARE DEBUG:")
        print(f"   Train dataset size: {len(train_dataloader)}")
        print(f"   Train size computed: ( N_batch * B_size ) {len(train_dataloader) * training_args.per_device_train_batch_size}")
        print(f"   Eval dataset size: {len(eval_dataloader)}")
        print(f"   Train size computed: ( N_batch * B_size ) {len(train_dataloader) * training_args.per_device_eval_batch_size}")
        print(f"   Batch size per device ( Train, Eval ): {training_args.per_device_train_batch_size}, {training_args.per_device_eval_batch_size}")
        print(f"   Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"   Total number of processes: {accelerator.num_processes}")
        print(f"   Total number of GPUs: {torch.cuda.device_count()}")
        print(f"   Total number of training epochs: {training_args.num_train_epochs}")
        print(f"   Total number of warmup steps: {training_args.num_warmup_steps}")
        print(f"   Total number of training steps x Epoch: {num_update_steps_per_epoch}")


        try:
            logger.info(f"  Scheduler total steps = {training_args.max_train_steps}")
            logger.info(f"  Scheduler warmup steps = {training_args.num_warmup_steps}")
            if 'lr_scheduler' in locals():
                scheduler_state = getattr(lr_scheduler, 'state_dict', lambda: {})()
                logger.info(f"  Scheduler state keys: {list(scheduler_state.keys()) if scheduler_state else 'No state_dict available'}")
        except Exception as e:
            logger.warning(f"  Could not log scheduler details: {e}")
    # === LoRA + Gradient Checkpointing Safe Enable ===
    if training_args.debug:
        # Debugging: Check if the dataloaders are correctly set up
        for i, b in tqdm(enumerate(train_dataloader), desc="Checking train dataloader batches", total=len(train_dataloader)):
            assert 'pixel_values' in b, f"Batch {i} does not contain 'pixel_values' key. Available keys: {b.keys()}"
            # Debug: Check tensor shapes in the first batch
            if i == 0:
                print("🔄 Checking tensor shapes in the first batch:")
                for key, value in b.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: {value.shape} on device {value.device}")
                    else:
                        print(f"   {key}: {type(value)} (not a tensor)")

        for i, b in tqdm(enumerate(eval_dataloader), desc="Checking eval dataloader batches", total=len(eval_dataloader)):
            assert 'pixel_values' in b, f"Batch {i} does not contain 'pixel_values' key. Available keys: {b.keys()}"
            # Debug: Check tensor shapes in the first batch
            if i == 0:
                print("🔄 Checking tensor shapes in the first eval batch:")
                for key, value in b.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: {value.shape} on device {value.device}")
                    else:
                        print(f"   {key}: {type(value)} (not a tensor)")


    # Initialize W&B run name
    if training_args.report_to == "wandb" and accelerator.is_local_main_process:
        project_name = f'{model_args.model_name_or_path.split("/")[-1]}-FT'
        run_name = f'{model_args.model_name_or_path.split("/")[-1]}-finetuning-{training_args.peft_strategy}-date-{datetime.now().strftime("%Y-%m-%d")}-epoch-{training_args.num_train_epochs}-bs-{training_args.per_device_train_batch_size}'
        os.environ["WANDB_PROJECT"] = project_name  # project in W&B UI
        os.environ["WANDB_RUN_NAME"] = run_name  # run name in W&B UI
        os.environ["WANDB_NAME"] = run_name  # alternative alias for run name



    # Progress bar setup
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    best_metric = None
    best_metric_checkpoint = None

    # === Training Loop (VERSIONE CORRETTA) ===
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
        logger.info(f"  Eval steps = {training_args.eval_steps}")

    # Resume handling
    starting_epoch = 0
    resume_step = None
    completed_steps = 0
    best_model_state = None
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
        logger.info(f"||||---- Epoch {epoch + 1} ----||||")
        model.train()
        total_loss = 0 if training_args.with_tracking else None

        active_dataloader = (
                accelerator.skip_first_batches(train_dataloader, resume_step)
                if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None
                else train_dataloader
        )


        for step, batch in enumerate(active_dataloader):
            if accelerator.is_main_process:
                logger.info(f"||||Step {step + 1} / {len(active_dataloader)} ||||")
            # Training step
            if training_args.debug:
                pixel_values = batch['pixel_values']
                device = pixel_values.device
                logger.info(f"[DEBUG] PIXEL_VALUES shape: {pixel_values.shape} on device: {device} || {completed_steps}")
            # Forward pass -->
            loss = training_step(
                    model=model,
                    batch=batch,
                    accelerator=accelerator,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps
            )

            # Manual gradient accumulation for DeepSpeed compatibility
            if (step + 1) % training_args.gradient_accumulation_steps == 0:

                safe_wait_for_everyone_simple(accelerator=accelerator)
                # Backward pass <--
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

                if training_args.with_tracking:
                    log_data = {
                            "train/loss": loss.item(),
                            "epoch"     : epoch,
                            "step"      : completed_steps,
                    }
                    if lr_scheduler is not None:
                        log_data["train/lr"] = lr_scheduler.get_last_lr()[0]

                    accelerator.log(log_data, step=completed_steps)

            if training_args.with_tracking:
                step_loss = accelerator.reduce(loss.clone()).item()
                if total_loss is not None:
                    total_loss += step_loss

            # === EVALUATION DURANTE I STEPS (VERSIONE SICURA) ===
            if completed_steps % training_args.eval_steps == 0 and eval_dataloader is not None and completed_steps > 0:
                if accelerator.is_main_process:
                    logger.info(f"Starting evaluation at step {completed_steps}")
                # CRITICAL: Sincronizza PRIMA della valutazione
                try:
                    # Valutazione con gestione errori robusta
                    perplexity, eval_loss = None, None
                    try:
                        perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator, logger)

                        if accelerator.is_main_process:
                            logger.info(f"Evaluation completed: perplexity={perplexity}, eval_loss={eval_loss}")
                    except Exception as eval_error:
                        logger.error(f"Evaluation failed at step {completed_steps}: {eval_error}")
                        perplexity, eval_loss = float('inf'), torch.tensor(float('inf'))

                    # Log metrics solo se la valutazione è riuscita
                    if perplexity is not None and training_args.with_tracking:
                        accelerator.log({
                                "perplexity": perplexity,
                                "eval/loss" : eval_loss if isinstance(eval_loss, (int, float)) else eval_loss.item(),
                                "epoch"     : epoch,
                                "step"      : completed_steps,
                        }, step=completed_steps)

                    if perplexity is not None and perplexity != float('inf'):
                        if best_metric is None or best_metric > perplexity:
                            best_metric = perplexity
                            best_metric_checkpoint = os.path.join(training_args.output_dir, "best_checkpoint")

                            # Clear previous best model state to free memory
                            if best_model_state is not None:
                                del best_model_state

                            # Get state dict and move to CPU
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model_state = {k: v.cpu() for k, v in unwrapped_model.state_dict().items()}

                            checkpoint_info = {
                                    "step"       : completed_steps,
                                    "epoch"      : epoch,
                                    "best_metric": best_metric,
                                    "perplexity" : perplexity,
                                    "eval_loss"  : eval_loss if isinstance(eval_loss, (int, float)) else eval_loss.item()
                            }

                    safe_wait_for_everyone_simple(accelerator=accelerator)
                    model.train()  # Assicurati che il modello sia in modalità train dopo la valutazione

                except Exception as sync_error:
                    logger.error(f"Synchronization error during evaluation: {sync_error}")

        # === EVALUATION FINE EPOCA (VERSIONE SICURA) ===
        logger.info(f"Starting end-of-epoch evaluation for epoch {epoch}")
        # For saving later, create a temporary model:
        if best_model_state is not None:
            # Create temporary model for saving
            temp_model = accelerator.unwrap_model(model)
            temp_model.load_state_dict({k: v.cuda() for k, v in best_model_state.items()})

            save_complete_checkpoint(
                    model=temp_model,
                    accelerator=accelerator,
                    save_path=best_metric_checkpoint,
                    tokenizer=tokenizer,
                    logger=logger,
                    additional_info=checkpoint_info
            )
            del temp_model  # Clean up


        try:
            # Sincronizza prima della valutazione
            safe_wait_for_everyone_simple(accelerator=accelerator)
            # Valutazione
            perplexity, eval_loss = None, None
            try:
                perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator, logger)
                logger.info(f"End-of-epoch evaluation: perplexity={perplexity}, eval_loss={eval_loss}")
            except Exception as eval_error:
                logger.error(f"End-of-epoch evaluation failed: {eval_error}")
                perplexity, eval_loss = float('inf'), torch.tensor(float('inf'))

            # Log metrics
            if training_args.with_tracking and perplexity is not None:
                accelerator.log({
                        "perplexity": perplexity,
                        "eval/loss" : eval_loss if isinstance(eval_loss, (int, float)) else eval_loss.item(),
                        "train_loss_total": total_loss / len(train_dataloader) if total_loss is not None and len(train_dataloader) > 0 else 0,
                        "epoch"     : epoch,
                        "step"      : completed_steps,
                }, step=completed_steps)

            # === CHECKPOINTING EPOCA (SOLO MAIN PROCESS) ===

            # Checkpoint periodici per epoca
            if (training_args.checkpointing_strategy == 'epoch' and
                    isinstance(training_args.checkpointing_divider, int)):
                if (epoch + 1) % training_args.checkpointing_divider == 0:
                    save_path = os.path.join(training_args.output_dir, f"epoch_every_{epoch + 1}")
                    # Assicurati che la directory esista

                    checkpoint_save_with_sync(accelerator=accelerator,
                                              save_path=save_path)

                    logger.info(f"Saved periodic epoch checkpoint: {save_path} -- EPOCH {epoch + 1}")


            # Sincronizza dopo tutti i salvataggi
            safe_wait_for_everyone_simple(accelerator=accelerator)

        except Exception as epoch_error:
            logger.error(f"Error during end-of-epoch processing: {epoch_error}")
            # Continua con l'epoca successiva
            continue


    logger.info("Epoch Training loop completed successfully!")
    # === EVALUATION FINALE ===
    # logger.info("Starting final evaluation...")

    # # Carica il miglior checkpoint se richiesto
    # if training_args.load_best_model and best_metric_checkpoint and os.path.exists(best_metric_checkpoint):
    #     try:
    #         logger.info(f"Loading best checkpoint: {best_metric_checkpoint}")
    #         accelerator.load_state(best_metric_checkpoint)
    #         logger.info("Best checkpoint loaded successfully")
    #     except Exception as load_error:
    #         logger.error(f"Failed to load best checkpoint: {load_error}")

    # Valutazione finale
    try:
        safe_wait_for_everyone_simple(accelerator=accelerator)

        perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator, logger)
        logger.info(f"Final model metrics: perplexity={perplexity}, eval_loss={eval_loss}")

        if best_metric and perplexity != best_metric and perplexity != float('inf'):
            logger.warning(
                    f"Best metric {best_metric} does not match final metric {perplexity}."
            )
        if training_args.with_tracking and perplexity is not None:
            accelerator.log({
                    "perplexity": perplexity,
                    "eval/loss" : eval_loss if isinstance(eval_loss, (int, float)) else eval_loss.item(),
                    "train_loss_total": total_loss / len(train_dataloader) if total_loss is not None and len(train_dataloader) > 0 else 0,
                    "epoch"     : epoch,
                    "step"      : completed_steps,
            }, step=completed_steps)

        safe_wait_for_everyone_simple(accelerator=accelerator)

    except Exception as final_eval_error:
        logger.error(f"Final evaluation failed: {final_eval_error}")
        perplexity, eval_loss = float('inf'), torch.tensor(float('inf'))
    # === SALVATAGGIO FINALE ===
    if training_args.output_dir is not None:
        try:
            safe_wait_for_everyone_simple(accelerator=accelerator)

            # Salva modello finale
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                    training_args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
            )

            # Salva tokenizer (solo main process)
            if accelerator.is_main_process and tokenizer is not None:
                tokenizer.save_pretrained(training_args.output_dir)

                # Salva risultati finali
                results = {
                        "perplexity" : perplexity if perplexity != float('inf') else "inf",
                        "eval/loss"  : eval_loss.item() if hasattr(eval_loss, 'item') else eval_loss,
                        "best_metric": best_metric if best_metric is not None else "not_computed"
                }
                results_path = os.path.join(training_args.output_dir, "all_results.json")
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

                logger.info(f"Results saved to: {results_path}")

            logger.info("✅ Model and results saved successfully")

        except Exception as save_error:
            logger.error(f"Error saving final model: {save_error}")

    # End tracking
    if training_args.with_tracking:
        try:
            accelerator.end_training()
        except Exception as tracking_error:
            logger.error(f"Error ending tracking: {tracking_error}")

    logger.info("🎉 Training completed successfully!")
    logger.info("|/| May the force be with you! Training completed successfully.")

if __name__ == "__main__":
    main()