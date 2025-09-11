from dataclasses import dataclass, field
from typing import Optional, List, Union
import os

from transformers import TrainingArguments
from transformers.trainer_utils import SaveStrategy, IntervalStrategy

cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
            default="google/gemma-3-4b-it",
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )


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
    # Optimizer configuration

    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use. Options: adamw_hf, adamw_torch, adamw_torch_fused, adafactor."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer."})
    adam_beta2: float = field(default=0.99, metadata={"help": "Beta2 for AdamW optimizer."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_projector: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    # Basic training arguments
    deepspeed_config_file: str = field(default="deepspeed/zero3.json", metadata={"help": "Path to the DeepSpeed Zero-2 configuration file."})
    output_dir: str = field(default="./results", metadata={"help": "Output directory for model predictions and checkpoints."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed that will be set at the beginning of training."})

    # Model configuration
    model_max_length: int = field(
            default=131072,
            metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    # LoRA/PEFT configuration
    lora_enable: bool = field(default=False, metadata={"help": "Whether to enable LoRA training."})
    vision_lora: bool = field(default=False, metadata={"help": "Whether to enable LoRA training."})

    lora_r: int = field(default=64, metadata={"help": "Rank for LoRA layers."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})

    lora_weight_path: str = field(default="", metadata={"help": "Path to LoRA weights."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias."})
    vision_lr: Optional[float] = field(default=None, metadata={"help": "LoRA learning rate."})
    projector_lr: Optional[float] = field(default=None, metadata={"help": "LoRA learning rate."})
    peft_strategy: str = field(default="lora_gaussian", metadata={"help": "PEFT strategy to use."})
    num_lora_modules: int = -1
    regex_module: bool = field(default=False, metadata={"help": "Whether to use regex to match module names for LoRA."})


    lora_namespan_exclude: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Layers to exclude"}
    )
    use_liger: bool = field(default=False, metadata={"help": "Whether to use liger kernel."})


    # Training configuration
    max_train_steps: Optional[int] = field(default=None, metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."})

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    bits: int = field(default=16, metadata={"help": "Quantization bits. Choose from 16, 8, 4, or 0 (no quantization)."})
    loss_function: str = field(default="default", metadata={"help": "Loss function to use. Options: default, SFT or DFT."})



    # Scheduler configuration
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use. Options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, or cyclic."})
    warmup_min_lr: float = field(default=0.0, metadata={"help": "Warmup minimum learning rate."})
    num_warmup_steps: int = field(default=None, metadata={"help": "Number of warmup steps for the learning rate scheduler."})
    cycle_min_lr: float = field(default=1e-7, metadata={"help": "Minimum learning rate for cyclical learning rate scheduler."})
    total_num_steps: Optional[int] = field( default=None, metadata={"help": "Total number of steps for the learning rate scheduler."})
    warmup_ratio: float = field(default=0.01, metadata={"help": "Warmup ratio for the learning rate scheduler."})



    # Multimodal training configuration
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Whether to finetune the vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Whether to finetune the language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Whether to finetune the attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Whether to finetune the MLP modules."})

    # Training configuration
    verbose_logging: bool = field(default=False, metadata={"help": "Whether to enable verbose logging."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to use gradient checkpointing to save memory at the expense of slower backward pass."})
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


    # liger-kernel


@dataclass
class TrainingArguments(TrainingArguments):
    # Optimizer configuration
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed."})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use. Options: adamw_hf, adamw_torch, adamw_torch_fused, adafactor."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer."})
    adam_beta2: float = field(default=0.99, metadata={"help": "Beta2 for AdamW optimizer."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_projector: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    # Basic training arguments
    deepspeed_config_file: str = field(default="./configs/deepspeed_zero2.json", metadata={"help": "Path to the DeepSpeed Zero-2 configuration file."})
    output_dir: str = field(default="./results", metadata={"help": "Output directory for model predictions and checkpoints."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})

    # Model configuration
    model_max_length: int = field(
            default=2048,
            metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    remove_unused_columns: bool = field(default=False, metadata={"help": "Whether to remove unused columns."})
    attn_implementation: str = field(default="eager", metadata={"help": "Use transformers attention implementation."})

    # LoRA/PEFT configuration
    lora_enable: bool = field(default=False, metadata={"help": "Whether to enable LoRA training."})
    vision_lora: bool = field(default=False, metadata={"help": "Whether to enable LoRA training."})

    lora_r: int = field(default=64, metadata={"help": "Rank for LoRA layers."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})

    lora_weight_path: str = field(default="", metadata={"help": "Path to LoRA weights."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias."})
    vision_lr: Optional[float] = field(default=None, metadata={"help": "LoRA learning rate."})
    projector_lr: Optional[float] = field(default=None, metadata={"help": "LoRA learning rate."})
    peft_strategy: str = field(default="lora_gaussian", metadata={"help": "PEFT strategy to use."})
    num_lora_modules: int = -1
    regex_module: bool = field(default=False, metadata={"help": "Whether to use regex to match module names for LoRA."})


    lora_namespan_exclude: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Layers to exclude"}
    )
    use_liger: bool = field(default=False, metadata={"help": "Whether to use liger kernel."})


    # Training configuration
    max_train_steps: Optional[int] = field(default=None, metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."})

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    bits: int = field(default=16, metadata={"help": "Quantization bits. Choose from 16, 8, 4, or 0 (no quantization)."})
    loss_function: str = field(default="default", metadata={"help": "Loss function to use. Options: default, SFT or DFT."})



    # Scheduler configuration
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use. Options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, or cyclic."})
    warmup_min_lr: float = field(default=0.0, metadata={"help": "Warmup minimum learning rate."})
    num_warmup_steps: int = field(default=None, metadata={"help": "Number of warmup steps for the learning rate scheduler."})
    cycle_min_lr: float = field(default=1e-7, metadata={"help": "Minimum learning rate for cyclical learning rate scheduler."})
    total_num_steps: Optional[int] = field( default=None, metadata={"help": "Total number of steps for the learning rate scheduler."})
    warmup_ratio: float = field(default=0.01, metadata={"help": "Warmup ratio for the learning rate scheduler."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."})
    # Multimodal training configuration
    finetune_vision_layers: bool = field(default=False, metadata={"help": "Whether to finetune the vision layers."})
    finetune_language_layers: bool = field(default=True, metadata={"help": "Whether to finetune the language layers."})
    finetune_attention_modules: bool = field(default=True, metadata={"help": "Whether to finetune the attention modules."})
    finetune_mlp_modules: bool = field(default=True, metadata={"help": "Whether to finetune the MLP modules."})

    # Training configuration
    verbose_logging: bool = field(default=False, metadata={"help": "Whether to enable verbose logging."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to use gradient checkpointing to save memory at the expense of slower backward pass."})
    # Checkpointing and saving
    checkpointing_strategy: Optional[str] = field(default='epoch', metadata={"help": "Checkpointing steps, can be 'epoch', 'steps', or a number of steps."})
    checkpointing_divider: Optional[int] = field(default=1, metadata={"help": "Checkpointing steps, can be 'epoch', 'steps', or a number of steps."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to checkpoint to resume from."})

    # Evaluation and best model
    load_best_model: bool = field(default=True, metadata={"help": "Whether to load the best model at the end of training."})

    # W&B integration
    report_to: str = field(default="wandb")
    with_tracking: bool = field(default=True, metadata={"help": "Whether to use tracking for the training run."})
    lr: float = field(default=2e-4, metadata={"help": "Learning rate for the optimizer."})

    save_strategy: Union[SaveStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(
            default=None,
            metadata={
                    "help": (
                            "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                            "If smaller than 1, will be interpreted as ratio of total training steps."
                    )
            },
    )

    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    greater_is_better : bool = field(
        default=False, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    # liger-kernel
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow comet and swanlab logging."
        },
    )

    label_names: Optional[list[str]] = field(
        default_factory=list,
        metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
