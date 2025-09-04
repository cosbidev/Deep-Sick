#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.

"""
Fixed version of finetuning script for GEMMA3 or other causal language models using Accelerate with DeepSpeed Zero-2.
This version addresses the "torch.cat(): expected a non-empty list of Tensors" error.
"""
import ast
import gc
import math
import pathlib
import warnings
from datetime import datetime
from typing import Optional, Any, Callable

from accelerate import Accelerator
from torch.xpu import device
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

# Suppress warnings
warnings.filterwarnings("ignore")
import sys
sys.path.extend(["./src", './'])

from src.finetune.monkey_patch_forward import replace_gemma3_forward
from src.params import ModelArguments, DataArguments, TrainingArguments

from src.dataset import load_parquet_image_dataset
from src.models import get_collator, configure_model_for_training, GemmaSFTTrainer
from src.distributed import checkpoint_save_with_sync, safe_wait_for_everyone_simple, initialize_accelerator_safely
from util_finetune import evaluate, rank0_print, compute_SFT, compute_DFT, FakeVisionDataset, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer

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


def setup_model_and_config(model_args, training_args, device, compute_dtype=torch.float16, **kwargs):
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
                    torch_dtype=compute_dtype,
                    cache_dir=CACHE_DIR,
                    attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
                    **kwargs
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




def train():
    # Parse arguments with flexible handling
    parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    local_rank = training_args.local_rank
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
    rank0_print("Accelerator initialized successfully")
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
                quantization_config=BitsAndBytesConfig(
                        device_map={"": training_args.device},
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
    model = setup_model_and_config(
            model_args=model_args,
            training_args=training_args,
            device=torch.device("cuda:{}".format(local_rank) if torch.cuda.is_available() else "cpu"),
            compute_dtype=compute_dtype,
            **bnb_model_from_pretrained_args
    )

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
    print("✅ Collator setup completed")


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)

            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    trainer = GemmaSFTTrainer(
            model=model,
            processing_class=processor,
            args=training_args,
            **dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator)
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

    print("✅ Data loaders created successfully")

if __name__ == "__main__":
    train()
