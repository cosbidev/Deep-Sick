#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.

"""
Fixed version of finetuning script for GEMMA3 or other causal language models using Accelerate with DeepSpeed Zero-2.
This version addresses the "torch.cat(): expected a non-empty list of Tensors" error.
"""
import ast
import gc
import math
import warnings
from datetime import datetime
from typing import Optional, Any, Callable

from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
import os
import json
import logging
from accelerate.utils import DummyOptim, DummyScheduler
import torch
import transformers
from transformers import (
    BitsAndBytesConfig, Gemma3ForConditionalGeneration
)

from accelerate.state import DistributedType
from torch.utils.data import DataLoader, DistributedSampler

from transformers import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled

# Suppress warnings
warnings.filterwarnings("ignore")
import sys
sys.path.extend(["./src", './'])

from src.finetune.monkey_patch_forward import replace_gemma3_forward
from src.params import ModelArguments, DataArguments, CustomTrainingArguments

from src.dataset import load_parquet_image_dataset
from src.models import get_collator, configure_model_for_training
from src.distributed import checkpoint_save_with_sync, safe_wait_for_everyone_simple, initialize_accelerator_safely
from util_finetune import evaluate, rank0_print, compute_SFT, compute_DFT, FakeVisionDataset

from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma3_text
# Environment setup
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")

logger = logging.getLogger(__name__)
hf_token = os.environ.get("HF_TOKEN", "")

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

        # if data_args.data_debug:
        #     # Reduce the dataset size for debugging purposes
        #     for split in raw_datasets.keys():
        #         raw_datasets[split] = raw_datasets[split].select(range(0, len(raw_datasets[split]), len(raw_datasets[split]) // 1787))

    elif data_args.dataset_name is None:
        raise ValueError(
                "You need to specify either a dataset name or a dataset directory. "
                "Use --dataset_name for a HF dataset or --dataset_dir to specify the dataset folder in local (parquet)."
        )

    return raw_datasets["train"], raw_datasets["val"]


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.vision_tower
    #vision_tower.to(dtype=compute_dtype, device=device)

    img_projection_params = model.multi_modal_projector.parameters()
    set_requires_grad(img_projection_params, not training_args.freeze_projector)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # if training_args.bits in [4, 8]:
    #     model.model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=device)


def configure_llm(model, training_args):
    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def setup_model_and_config(model_args, training_args, device, bnb_model_from_pretrained_args={}, compute_dtype=torch.float16):
    """Setup model configuration and load the model."""
    assert model_args.model_name_or_path, "You need to specify a model name or path"

    # Configuration overrides
    customized_kwargs = dict()
    overwrite_config = {}
    # cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
    #
    # if overwrite_config:
    #     for k, v in overwrite_config.items():
    #         setattr(cfg_pretrained, k, v)
    #     customized_kwargs["config"] = cfg_pretrained


    # Load model
    print("🔄 Loading base model...")
    if 'gemma' in model_args.model_name_or_path.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    #torch_dtype=compute_dtype,
                    cache_dir=CACHE_DIR,
                    attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
                    **bnb_model_from_pretrained_args
            )
    # TEST

    print("✅ Base model loaded successfully")

    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, device)
    model.config.use_cache = False


    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        # This is a workaround for a bug in the current implementation of gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})

    # CRITICAL: Configure PEFT/LoRA BEFORE any other operations
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        print("🔄 Applying LoRA configuration...")

        model = configure_model_for_training(
                model,
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                lora_namespan_exclude=lora_namespan_exclude,
                training_args=training_args,

        )

        # Verify that we have trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("❌ No trainable parameters found after LoRA configuration!")

        print(f"✅ LoRA applied successfully with {len(trainable_params)} trainable parameter groups")

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
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    lr_mapper = {}
    if training_args.projector_lr is not None:
        lr_mapper["multi_modal_projector"] = training_args.projector_lr
    if training_args.vision_lr is not None:
        lr_mapper["vision_tower"] = training_args.vision_lr


    if len(lr_mapper) > 0:
        special_lr_parameters = [name for name, _ in model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
        optimizer_grouped_parameters = [
                {
                        "params"      : [p for n, p in model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": training_args.weight_decay,
                },
                {
                        "params"      : [p for n, p in model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                },
        ]
        for module_keyword, lr in lr_mapper.items():
            module_parameters = [name for name, _ in model.named_parameters() if module_keyword in name]
            optimizer_grouped_parameters.extend(
                    [
                            {
                                    "params"      : [p for n, p in model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                    "weight_decay": training_args.weight_decay,
                                    "lr"          : lr,
                            },
                            {
                                    "params"      : [p for n, p in model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                    "weight_decay": 0.0,
                                    "lr"          : lr,
                            },
                    ]
            )
    else:
        optimizer_grouped_parameters = [
                {
                        "params"      : [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": training_args.weight_decay,
                },
                {
                        "params"      : [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                },
        ]



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
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in model.modules():
            import torch.nn as nn
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped / 2 ** 20}M params")

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


# def save_complete_checkpoint(
#         model, accelerator,
#         save_path, tokenizer=None, logger=None,
#         additional_info=None
#         ):
#     """
#     Save complete checkpoint: model + optimizer + scheduler + tokenizer
#     Fast and distributed-training safe
#     """
#
#     if accelerator.is_main_process:
#         # Create save directory
#         os.makedirs(save_path, exist_ok=True)
#         logger.info(f"💾 Saving complete checkpoint to: {save_path}")
#
#
#         model.save_pretrained(
#                 save_path,
#                 is_main_process=accelerator.is_main_process,
#                 save_function=accelerator.save,
#                 state_dict=accelerator.get_state_dict(model),
#                 safe_serialization=True  # Use safetensors (faster & safer)
#         )
#
#         # 4. Save Tokenizer
#         if tokenizer is not None:
#             tokenizer.save_pretrained(save_path)
#
#         # 5. Save Training Info (step, epoch, metrics, etc.)
#         training_info = {
#                 "step"         : additional_info.get("step", 0) if additional_info else 0,
#                 "epoch"        : additional_info.get("epoch", 0) if additional_info else 0,
#                 "best_metric"  : additional_info.get("best_metric") if additional_info else None,
#                 "training_args": additional_info.get("training_args") if additional_info else None
#         }
#
#         with open(os.path.join(save_path, "training_info.json"), "w") as f:
#             json.dump(training_info, f, indent=2, default=str)
#
#         if logger:
#             logger.info(f"✅ Complete checkpoint saved successfully")
#             logger.info(f"   📁 Directory: {save_path}")
#             logger.info(f"   🏗️  Model: ✓")
#             logger.info(f"   🔤 Tokenizer: {'✓' if tokenizer else '✗'}")
#
#     # Synchronize all processes
#     accelerator.wait_for_everyone()
#     return True


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


def train():
    # Parse arguments with flexible handling
    parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, CustomTrainingArguments))

    model_args, data_args, training_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    rank0_print(f"model_args: {model_args}")
    rank0_print(f"data_args: {data_args}")
    rank0_print(f"training_args: {training_args}")


    if training_args.use_liger:
        apply_liger_kernel_to_gemma3_text(
            rope=True, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=True, geglu=True
        )
    # Setup logging
    setup_logging(training_args)
    rank0_print("✅ Logging setup completed")


    # Replace GEMMA3 forward method if using Liger
    #replace_gemma3_forward(use_liger=training_args.use_liger)

    training_args.output_dir += "lora" + str(training_args.lora_r) + "_alpha" + str(training_args.lora_alpha) if training_args.lora_enable else ""
    training_args.output_dir += f"_{training_args.loss_function}" if training_args.loss_function != "default" else "_vanilla"

    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")


    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.lora_namespan_exclude is None:
        training_args.lora_namespan_exclude = ["multi_modal_projector"]

    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["vision_tower", "multi_modal_projector"]

    # Compute dtype
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


    # NOW initialize accelerator with DeepSpeed config
    accelerator = initialize_accelerator_safely(training_args=training_args)
    rank0_print("Accelerator initialized successfully")
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
                quantization_config=BitsAndBytesConfig(
                        device=accelerator.device,
                        load_in_4bit=training_args.bits == 4,
                        load_in_8bit=training_args.bits == 8,
                        llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=training_args.double_quant,
                        bnb_4bit_quant_type=training_args.quant_type,
                )
        ))


    # Set seed for reproducibility
    if training_args.seed is not None:
        transformers.set_seed(training_args.seed)

    # Load datasets
    rank0_print("🔄 Loading datasets...")
    train_dataset, eval_dataset = load_and_prepare_datasets(data_args)
    rank0_print("✅ Datasets loaded successfully")


    # CRITICAL: Setup model and LoRA BEFORE accelerator initialization
    rank0_print("Setting up model with LoRA...")


    # Ensure model_args is not None
    accelerator.wait_for_everyone()
    model = setup_model_and_config(
            model_args=model_args,
            training_args=training_args,
            bnb_model_from_pretrained_args=bnb_model_from_pretrained_args,
            device=accelerator.device,
    )


    accelerator.wait_for_everyone()
    print("✅ Model and LoRA setup completed")
    # Get collator and tokenizer
    print("🔄 Setting up collator...")
    collator = get_collator(
            model_id=model_args.model_name_or_path,
            padding_side="left",
            max_length=training_args.model_max_length,
            token=hf_token
    )

    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    processor = collator.processor
    tokenizer = collator.tokenizer if hasattr(collator, 'tokenizer') else processor
    print("✅ Collator setup completed")


    # Calculate training steps
    if accelerator.num_processes == 0:
        raise RuntimeError("❌ No processes found. Please ensure you are running with multiple GPUs or in a distributed environment.")
    print(f"🔄 Accelerator initialized with {accelerator.num_processes} processes")
    # Create data loaders BEFORE accelerator initialization
    print("🔄 Creating data loaders...")
    try:
        sampler_train = DistributedSampler(
                train_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                drop_last=True
        )


        # sampler_train = DistributedSamplerWithBluprint(
        #         train_dataset,
        #         num_replicas=accelerator.num_processes,
        #         rank=accelerator.process_index,
        #         batch_size=training_args.per_device_train_batch_size,
        #         lengths=train_dataset["length"],
        #         n_images=train_dataset["n_of_images"],
        #         shuffle=True,
        #         drop_last=True
        #
        # )

        train_dataloader = DataLoader(
                train_dataset,
                sampler=sampler_train,
                collate_fn=collator,
                batch_size=training_args.per_device_train_batch_size,
                pin_memory=True
        )


        print(f"✅ Train dataloader created with {len(train_dataloader)} batches")

        # eval_sampler = DistributedSamplerWithBluprint(
        #         eval_dataset,
        #         num_replicas=accelerator.num_processes,
        #         rank=accelerator.process_index,
        #         batch_size=training_args.per_device_eval_batch_size,
        #         lengths=eval_dataset["length"],
        #         n_images=eval_dataset["n_of_images"],
        #         shuffle=False,
        #         drop_last=True
        # )

        eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
                drop_last=True
        )
        print(f"✅ Eval sampler created with {len(eval_dataset)} samples")


        eval_dataloader = DataLoader(
                eval_dataset,
                sampler=eval_sampler,
                collate_fn=collator,
                drop_last=True,
                batch_size=training_args.per_device_eval_batch_size,
                pin_memory=True,
                shuffle=False
        )

    except Exception as e:
        logger.error(f"❌ Error creating data loaders: {e}")
        raise

    print("✅ Data loaders created successfully")

    # Test eval dataloader
    _ = next(iter(eval_dataloader))
    print(f"✅ Eval dataloader created with {len(eval_dataloader)} batches")
    # _size_real = training_args.per_device_train_batch_size * accelerator.num_processes
    len_total_dataloader = len(train_dataloader)
    # CORREZIONE: Calcolo corretto degl i step
    num_update_steps_per_epoch = math.ceil( len_total_dataloader / training_args.gradient_accumulation_steps)
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

    if training_args.loss_function == "SFT":
        compute_loss_fn = compute_SFT
    elif training_args.loss_function == "DFT":
        compute_loss_fn = compute_DFT
    else:
        compute_loss_fn = None  # Default loss function in the model.loss attribute (NLLLoss)

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
                "processes"    : accelerator.num_processes,
                "loss"         : training_args.loss_function
        }


        accelerator.init_trackers(
                project_name=f'{model_args.model_name_or_path.split("/")[-1]}-multinode',
                config=wandb_config,
                init_kwargs={
                        "wandb": {
                                "name": f'gemma3-multinode-{datetime.now().strftime("%m%d-%H%M")}-JOB-{os.environ.get("SLURM_JOB_ID", "")}',
                                "tags": ["multinode", f"{model_args.model_name_or_path}", "lora", "deepspeed", f"{training_args.loss_function}"]
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

        if training_args.gradient_checkpointing:

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
        model, optimizer, lr_scheduler = accelerator.prepare(
                model,
                optimizer,
                lr_scheduler,

        )
        print("✅ All components prepared successfully with accelerator")
        if accelerator.is_main_process:
            print(model)
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



    # Initialize W&B run name
    if training_args.report_to == "wandb" and accelerator.is_local_main_process:
        project_name = f'{model_args.model_name_or_path.split("/")[-1]}-FT'
        run_name = f'{model_args.model_name_or_path.split("/")[-1]}-finetuning-{training_args.peft_strategy}-date-{datetime.now().strftime("%Y-%m-%d")}-epoch-{training_args.num_train_epochs}-bs-{training_args.per_device_train_batch_size}'
        os.environ["WANDB_PROJECT"] = project_name  # project in W&B UI
        os.environ["WANDB_RUN_NAME"] = run_name  # run name in W&B UI
        os.environ["WANDB_NAME"] = run_name  # alternative alias for run name        [0, 0, 0,  ..., 0, 0, 0]]), 'pixel_values': tensor([[[[-1., -1., -1.,  ..., -1., -1., -1.],

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
        path = os.path.basename(training_args.resume_from_checkpoint)
        training_difference = path.split("_")[-1]  # Extract the last part after the last hyphen

        starting_epoch = int(training_difference)
        resume_step = None
        completed_steps = (starting_epoch * num_update_steps_per_epoch) + 1
        logger.info(f"  Resume from checkpoint: {training_args.resume_from_checkpoint}")
        logger.info(f"  Starting epoch: {starting_epoch}, Completed steps: {completed_steps}")

    progress_bar.update(completed_steps)
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
        logger.info(f"||||---- Epoch {epoch + 1} ----||||")
        total_loss = 0 if training_args.with_tracking else None

        active_dataloader = (
                accelerator.skip_first_batches(train_dataloader, resume_step)
                if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None
                else train_dataloader
        )

        for step, batch in enumerate(active_dataloader):
            pixel_values = batch['pixel_values']


            if accelerator.is_main_process:
                logger.info(f"||||Step {step + 1} / {len(active_dataloader)} ||||  ")
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
                    compute_loss_fn=compute_loss_fn,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps
            )
            # Manual gradient accumulation for DeepSpeed compatibility
            if (step + 1) % training_args.gradient_accumulation_steps == 0:

                #safe_wait_for_everyone_simple(accelerator=accelerator)
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
                            unwrapped_model_best = accelerator.unwrap_model(model)
                            best_model_state = {k: v.cpu() for k, v in unwrapped_model_best.state_dict().items()}
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
        if hasattr(train_dataloader.sampler, 'refresh_sampling'):
            print(f"🔄 Refreshing sampler after epoch {epoch + 1}...")
            train_dataloader.sampler.refresh_sampling()
            print("✅ Sampler refreshed successfully")
        elif hasattr(train_dataloader.sampler, 'sampler') and hasattr(train_dataloader.sampler.sampler, 'refresh_sampling'):
            # Handle case where sampler is wrapped by DistributedSamplerWrapper
            print(f"🔄 Refreshing wrapped sampler after epoch {epoch + 1}...")
            train_dataloader.sampler.sampler.refresh_sampling()
            print("✅ Wrapped sampler refreshed successfully")
        else:
            print(f"⚠️  No refresh_sampling method found on sampler after epoch {epoch + 1}")
        logger.info(f"Starting end-of-epoch evaluation for epoch {epoch}")
        # For saving later, create a temporary model:
        if best_model_state is not None:
            # Create temporary model for saving
            safe_wait_for_everyone_simple(accelerator=accelerator)

            #
            # non_lora_weights = get_peft_state_non_lora_maybe_zero_3(model.named_parameters(), require_grad_only=False)
            # torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            unwrapped_model_best.save_pretrained(
                    training_args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=best_model_state,  # full parameters
            )
            # Salva tokenizer (solo main process)
            if accelerator.is_main_process and tokenizer is not None:
                tokenizer.save_pretrained(training_args.output_dir)

                results_path = os.path.join(training_args.output_dir, "all_results.json")
                with open(results_path, "w") as f:
                    json.dump(checkpoint_info, f, indent=2)

                logger.info(f"Results saved to: {results_path}")

            logger.info("✅ Model and results saved successfully")

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
                    save_path = os.path.join(training_args.output_dir, f"epoch_{epoch + 1}")
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

    # End tracking
    if training_args.with_tracking:
        try:
            accelerator.end_training()
        except Exception as tracking_error:
            logger.error(f"Error ending tracking: {tracking_error}")

    logger.info("🎉 Training completed successfully!")
    logger.info("|/| May the force be with you! Training completed successfully.")

if __name__ == "__main__":
    train()
