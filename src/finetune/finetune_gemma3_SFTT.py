
# -*- coding: utf-8 -*-
"""
SFTT fine‑tuning of Qwen2.5‑VL‑3B, Gemma3b, MedGemma3B or Similar VLMs on CheXinstruct
-----------------------------------------------------------------------
This script continues from the user‑suppli"""

import os, sys
import argparse

sys.path.append('./')
# Set Hugging Face cache to current working directory
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

from datasets import load_dataset, config
print(f"Cache directory: {config.HF_DATASETS_CACHE}")

# ---------------------------------------------------------------------
# 0.  GLOBAL HYPERPARAMETERS & CONFIGURATIONS  ─────────────────────────
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = "unsloth/gemma-3-4b-it"
DEFAULT_N_SAMPLES_TRAIN = 5000  # Number of samples for quick testing
DEFAULT_MAX_STEPS = 5000
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1  # 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TENSORBOARD_DIR = "tensorboard"
DEFAULT_SAVE_STEPS = 50
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")
# ---------------------------------------------------------------------


import io
import json
from typing import Any, cast
import requests
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from trl import SFTTrainer, SFTConfig
from peft.tuners.lora.config import LoraConfig
from datasets import IterableDataset, Features
import datasets
from PIL import Image
from src.dataset import load_parquet_image_dataset
from src.models import data_format_gemma_conversation

HF_TOKEN = "..."

def load_image(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return image

def image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a VLM on CheXpert with GRPO.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the pre-trained model to use.")
    parser.add_argument("--n_samples_train", type=int, default=DEFAULT_N_SAMPLES_TRAIN, help="Number of training samples to use (for quick testing). Set to -1 to use full training set.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum training steps.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--num_generations", type=int, default=DEFAULT_NUM_GENERATIONS, help="Number of generations per prompt in GRPO.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE, help="Training batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps.")
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R, help="LoRA r dimension.")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save all outputs (checkpoints, logs, final model).")
    parser.add_argument("--tensorboard_dir", type=str, default=DEFAULT_TENSORBOARD_DIR, help="Tensorboard logs directory (relative to output_dir).")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS, help="Save checkpoint every N steps.")

    return parser.parse_args()



# ---------------------------------------------------------------------
# MAIN SCRIPT LOGIC  ──────────────────────────────────────────────────
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    tensorboard_path = os.path.join(args.output_dir, args.tensorboard_dir)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Tensorboard logs: {tensorboard_path}")

    # ---------------------------------------------------------------------
    # 1.  LOAD MODEL ─────────────────────────────────────────────────────
    # ---------------------------------------------------------------------

    def main():

        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", token=HF_TOKEN
        )
        model.config.use_cache = False  # Disable caching for training

        processor = AutoProcessor.from_pretrained(model_id, padding_side="right", token=HF_TOKEN)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Use eos token as pad token
        processor.tokenizer.padding_side = "right"

        # ---------------------------------------------------------------------
        # 2.  LOAD + PREP DATA  ────────────────────────────────────────────────
        # ---------------------------------------------------------------------
        print("Loading and preparing dataset...")
        chexinstruct_data = load_parquet_image_dataset('data_chexinstruct/hf_parquet')

        train_data = chexinstruct_data['train']
        val_data = chexinstruct_data['val']
        print(f"Train samples: {len(train_data)}")


        train_data = train_data.map(data_format_gemma_conversation)
        val_data = val_data.map(data_format_gemma_conversation)
        print(f"Train samples after formatting: {len(train_data)}")

        # Display a processed data sample
        print(train_data["train"][0])

        # ---------------------------------------------------------------------
        # 5.  SAVE FINAL MODEL  ──────────────────────────────────────────────
        # ---------------------------------------------------------------------
        final_model_path = os.path.join(args.output_dir, "final_model")
        print(f"Saving final LoRA adapter to {final_model_path}")
        # model.save_lora(final_model_path)
        print(f"Final LoRA adapter saved to {final_model_path}")


        USE_ITERABLE_DATASET = False

        messages_obj = [
                {
                        "role"   : "user",
                        "content": [
                                {"type": "image", },
                                {"type": "image", }
                        ]
                },
                {
                        "role"   : "assistant",
                        "content": [{"type": "text", "text": "duck"}]
                }
        ]

        if USE_ITERABLE_DATASET:
            def train_iterable_gen():
                yield {
                        "messages": json.dumps(messages_obj)
                }

            train_ds = IterableDataset.from_generator(
                    train_iterable_gen,
                    features=Features({
                            'messages': datasets.Value(dtype='string', id=None)
                    })
            )
        else:
            train_ds = [
                    {
                            "messages": json.dumps(messages_obj)
                    }
            ]

        image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg").resize((896, 896))

        def collate_fn(examples):
            print("collate_fn examples", examples)
            # Get the texts and images, and apply the chat template
            texts = [processor.apply_chat_template(json.loads(example["messages"]), tokenize=False, add_generation_prompt=False) for example in examples]
            images = [[image.convert("RGB"), image.convert("RGB")]]

            print("collate_fn texts", texts)
            print("collate_fn images", images)

            # Tokenize the texts and process the images
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

            print("collate_fn pixel_values", batch["pixel_values"].shape)
            print("collate_fn input_ids", batch["input_ids"].shape)

            # The labels are the input_ids, and we mask the padding tokens in the loss computation
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[labels == processor.image_token_id] = -100
            batch["labels"] = labels

            return batch

        # Set up LoRA configuration for causal language modeling
        lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM"
        )

        # Define training arguments
        training_args = SFTConfig(
                output_dir="./results",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                learning_rate=2e-4,
                logging_steps=1,
                save_steps=25,
                report_to="tensorboard",
                group_by_length=False,
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                gradient_checkpointing_kwargs=dict(use_reentrant=False),
                max_steps=1
        )

        # Create the SFTTrainer with LoRA parameters
        trainer = SFTTrainer(
                model=model,
                train_dataset=cast(Any, train_ds),

                peft_config=lora_config,
                args=training_args,
                data_collator=collate_fn,
                processing_class=processor.tokenizer,
        )

        print("Training model...")
        trainer.train()
        print("Training complete.")

if __name__ == "__main__":
    main()
#



# if __name__ == "__main__":
#     main()