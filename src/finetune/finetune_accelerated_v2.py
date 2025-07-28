#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.

"""
Fixed version of finetuning script for GEMMA3 or other causal language models using Accelerate with DeepSpeed Zero-2.
This version addresses the "torch.cat(): expected a non-empty list of Tensors" error.
"""
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from tqdm import tqdm
import os
import json
import logging
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
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
    lr: float = field(default=2e-4, metadata={"help": "Learning rate for the optimizer."})

    @property
    def train_batch_size(self) -> int:
        """Total effective batch size for training."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


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

        # Debug: Check model state before LoRA
        print("📊 Model state BEFORE LoRA:")
        debug_tensor_shapes(model, "BEFORE_LORA: ")

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
        print("📊 Model state AFTER LoRA:")
        debug_tensor_shapes(model, "AFTER_LORA: ")

        # Verify that we have trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("❌ No trainable parameters found after LoRA configuration!")

        print(f"✅ LoRA applied successfully with {len(trainable_params)} trainable parameter groups")

    # Disable gradient checkpointing to avoid conflicts with DeepSpeed
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_disable()

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
                              eps=1e-8,
                              betas=(0.9, 0.999))

    # Create learning rate scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
                name=training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=training_args.num_warmup_steps,
                num_training_steps=training_args.max_train_steps,
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

    print("🚀 Starting DeepSpeed Zero-2 Training with LoRA")
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
    print("🔄 Initializing Accelerator with DeepSpeed...")


    # Initialize accelerator with proper DeepSpeed configuration
    accelerator = Accelerator(
            log_with=training_args.report_to if training_args.with_tracking else None,
            project_dir=training_args.output_dir,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,

    )

    # Calculate training steps
    batch_size_real = training_args.per_device_train_batch_size * accelerator.num_processes
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / (batch_size_real * training_args.gradient_accumulation_steps))
    if training_args.max_train_steps is None:
        training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)
    # Create data loaders BEFORE accelerator initialization
    print("🔄 Creating data loaders...")
    try:
        sampler_train = BlueprintGroupedSampler(
                batch_size=training_args.per_device_train_batch_size,
                world_size= accelerator.num_processes,
                lengths=train_dataset["length"],
                n_images=train_dataset["n_of_images"],
        )

        sampler_val = BlueprintGroupedSampler(
                batch_size=training_args.per_device_eval_batch_size,
                world_size= accelerator.num_processes,
                lengths=eval_dataset["length"],
                n_images=eval_dataset["n_of_images"],
        )

        train_dataloader = DataLoader(
                train_dataset,
                sampler=sampler_train,
                collate_fn=collator,
                drop_last=True,
                batch_size=training_args.per_device_train_batch_size,
                pin_memory=True,
                num_workers=0,  # Disable multiprocessing to avoid issues
        )
        eval_dataloader = DataLoader(
                eval_dataset,
                sampler=sampler_val,
                collate_fn=collator,
                drop_last=True,
                batch_size=training_args.per_device_eval_batch_size,
                num_workers=0,
        )
        print("✅ Data loaders created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating data loaders: {e}")
        raise



    # Create optimizer with proper parameter grouping BEFORE accelerator.prepare()
    print("🔄 Creating optimizer...")
    optimizer, lr_scheduler = create_optimizers_with_parameter_groups(model, training_args, accelerator)
    print("✅ Optimizer created successfully")


    # Initialize tracking if enabled
    if training_args.with_tracking:
        accelerator.init_trackers(
                project_name=f'{model_args.model_name_or_path.split("/")[-1]}-finetuning-{training_args.peft_strategy}',
        )

    print("✅ Accelerator initialized successfully")

    # Final parameter check before prepare
    print("📊 Final parameter check before accelerator.prepare():")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"   Trainable parameters: {len(trainable_params)}")

    # # Verify optimizer has parameters
    # total_optimizer_params = sum(len(group['params']) for group in optimizer.param_groups)
    # print(f"   Optimizer parameter count: {total_optimizer_params}")
    #
    # if total_optimizer_params == 0:
    #     raise RuntimeError("❌ Optimizer has no parameters!")
    # CRITICAL: Prepare everything with accelerator in the correct order
    print("🔄 Preparing model, optimizer, and data loaders with accelerator...")

    try:
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


        # Uso:
        loss = call_forward_dummy(model.module.dummy_inputs, model)
        print("LOSS GRADIENT:", loss.requires_grad)

        print("✅ All components prepared successfully with accelerator")
    except Exception as e:
        logger.error(f"❌ Error during accelerator.prepare(): {e}")
        logger.error("This is likely due to empty parameter groups or incompatible configurations")
        raise
    # === LoRA + Gradient Checkpointing Safe Enable ===

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
            # For DeepSpeed Zero-2, we DON'T use accelerator.accumulate()
            pixel_values = batch['pixel_values']  # shape: (B, C, H, W)
            device = pixel_values.device

            logger.info(f"[DEBUG] PIXEL_VALUES shape: {pixel_values.shape} on device: {device} || {completed_steps}")
            # as it conflicts with DeepSpeed's gradient partitioning
            loss = training_step(
                    model=model,
                    batch=batch,
                    accelerator=accelerator,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps
            )
            # Manual gradient accumulation for DeepSpeed compatibility
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
                if total_loss is not None:
                    total_loss += step_loss

            # Checkpointing every N steps
            if isinstance(training_args.checkpointing_steps, int):
                if completed_steps % training_args.checkpointing_steps == 0:
                    output_dir = os.path.join(training_args.output_dir, f"step_{completed_steps}")
                    accelerator.save_state(output_dir)

            if completed_steps >= training_args.max_train_steps:
                break

        # Evaluation
        try:
            perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            perplexity, eval_loss = float('inf'), torch.tensor(float('inf'))

        if training_args.with_tracking:
            accelerator.log({
                    "perplexity": perplexity,
                    "eval_loss" : eval_loss if isinstance(eval_loss, (int, float)) else eval_loss.item(),
                    "train_loss": total_loss / len(train_dataloader) if total_loss is not None and len(train_dataloader) > 0 else 0,
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
        if best_metric is None or (perplexity != float('inf') and best_metric > perplexity):
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(training_args.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")
    # Reload best checkpoint
    if training_args.load_best_model and best_metric_checkpoint:
        accelerator.load_state(best_metric_checkpoint)
    # Final evaluation
    try:
        perplexity, eval_loss = evaluate(training_args, model, eval_dataloader, accelerator)
        logger.info(f"Final model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")

        if best_metric and perplexity != best_metric and perplexity != float('inf'):
            logger.warning(
                    f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
            )
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        perplexity, eval_loss = float('inf'), torch.tensor(float('inf'))

    # Save final model and tokenizer
    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        try:
            unwrapped_model.save_pretrained(
                    training_args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
            )
            if accelerator.is_main_process and tokenizer is not None:
                tokenizer.save_pretrained(training_args.output_dir)

            # Save results
            results = {
                    "perplexity": perplexity if perplexity != float('inf') else "inf",
                    "eval_loss" : eval_loss.item() if hasattr(eval_loss, 'item') else eval_loss
            }
            with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
                json.dump(results, f)

            print("✅ Model and results saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    # End tracking
    if training_args.with_tracking:
        accelerator.end_training()

    print("🎉 Training completed successfully!")
    logger.info("|/| May the force be with you! Training completed successfully.")


if __name__ == "__main__":
    main()