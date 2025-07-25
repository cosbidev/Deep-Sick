# finetune.py
import os
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from src.dataset import load_parquet_image_dataset
from trl import SFTTrainer, SFTConfig
import torch


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default="google/gemma-3-4b-it")  # Model name or path to the pretrained model
    dataset_name: str = field(default='chexinstruct')
    dataset_path: str = field(default='/data_chexinstruct/hf_parquet_gemma_format/gemma_findings')  # Path to the dataset if not using a predefined dataset
    output_dir: str = field(default="gemma-3-4b-it-chexinstruct-trl-sft")
    num_train_epochs: int = field(default=40)
    learning_rate: float = field(default=2e-5)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    save_strategy: str = field(default="epoch")  # Save checkpoints at the end of each epoch
    save_every_epochs: int = field(default=5)
    bf16: bool = field(default=True)  # Use bfloat16 precision for training
    fp16: bool = field(default=True)
    load_in_8bit: bool = field(default=False)
    deepspeed_config: str = field(default="deepspeed/ds_zero3_config.yaml")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    for field_ in ScriptArguments.__dataclass_fields__.values():
        arg_name = f"--{field_.name}"
        kwargs = {"default": field_.default, "type": type(field_.default) if field_.default is not None else str}
        if isinstance(field_.default, bool):
            kwargs["action"] = "store_true" if not field_.default else "store_false"
        parser.add_argument(arg_name, **kwargs)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the Accelerator with DeepSpeed
    accel = Accelerator(
        deepspeed_plugin={"config_file": args.deepspeed_config} if args.deepspeed_config else None,
        mixed_precision="bf16" if args.bf16 else ("fp16" if args.fp16 else "no"),
    )

    # Load model+tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=args.load_in_8bit,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    # Load dataset
    dataset = load_parquet_image_dataset(
        dataset_path=args.dataset_path,
        split_list=['train', 'val'],  # Specify the split you want to use ['test', 'val', 'train']
    )

    training_args = SFTConfig(
            output_dir=args.output_dir,  # Directory to save the model and push to the Hub. Use a specific repository id (e.g., gemma-3-4b-it-trl-sft-MMIU-Benchmark for multi-image datasets).
            num_train_epochs=args.num_train_epochs,  # Set the number of epochs to train the model.
            per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size for each device (e.g., GPU) during training. multi-image -> per_device_train_batch_size=1
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of steps before performing a backward/update pass to accumulate gradients. multi-image -> gradient_accumulation_steps=1
            gradient_checkpointing=args.gradient_checkpointing,  # Enable gradient checkpointing to reduce memory usage during training.
            optim=args.optim,  # Use the fused AdamW optimizer for better performance.
            save_strategy=args.save_strategy,  # Save checkpoints at the end of each epoch.
            save_every_epochs=args.save_every_epochs,  # Save the model every specified number of epochs.
            learning_rate=args.learning_rate,  # Learning rate for training.
            bf16=True,  # Enable bfloat16 precision for training to save memory and speed up computations.
            push_to_hub=True,  # Automatically push the fine-tuned model to Hugging Face Hub after training.
            report_to="tensorboard",  # Automatically report metrics to tensorboard.
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Set gradient checkpointing to non-reentrant to avoid issues.
            dataset_kwargs={"skip_prepare_dataset": True},  # Skip dataset preparation to handle preprocessing manually.
            remove_unused_columns=False,  # Ensure unused columns are not removed in the collator (important for batch processing).
    )


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

if __name__ == "__main__":
    main()
