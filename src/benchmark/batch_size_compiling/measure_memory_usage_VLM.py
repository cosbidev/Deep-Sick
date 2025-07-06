import sys
sys.path.append('./')
import time
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
from src.models import UnslothVisionDataCollator


class InterruptTraining(Exception):
    pass
import os
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["HF_TOKEN"] = "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"
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")

deepspeed_cfg = {
    "zero_optimization": {
        "stage": 3
    },
    "fp16": {
        "enabled": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": 1e-8,
        "weight_decay": "auto"
        }
    },
    "train_micro_batch_size_per_gpu": "auto"
    }

## Set the token for Hugging Face authentication
hf_token = os.getenv("HF_TOKEN")
login(hf_token)
# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


class InterruptableTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_time = 0
        self.end_time = 0
    END_ON_ITERATION = 30

    def training_step(self, model, inputs):
        step = self.state.global_step
        if step == 2:
            self.start_time = time.time()
        if step == self.END_ON_ITERATION:
            self.end_time = time.time()
            raise InterruptTraining()
        return super().training_step(model, inputs)


    def average_iterations_per_second(self):
        run_time = self.end_time - self.start_time
        return (self.END_ON_ITERATION - 1) / run_time



def process_vision_language_data(examples, processor, max_length=2048):
    """
    Process data for vision-language models like Qwen2.5-VL.
    This handles both image and text inputs.
    """
    images = examples["messages"]
    texts = examples["text"]

    # Create conversation format for VL models
    conversations = []
    for i, text in enumerate(texts):
        conversation = [
                {
                        "role"   : "user",
                        "content": [
                                {"type": "image"},
                                {"type": "text", "text": "What is the LaTeX formula shown in this image?"}
                        ]
                },
                {
                        "role"   : "assistant",
                        "content": [{"type": "text", "text": text}]
                }
        ]
        conversations.append(conversation)

    # Process with the processor (handles both images and text)
    processed = processor(
            conversations,
            images,
            truncation=True,
            padding="max_length",
            max_length=max_length
    )
    processed["labels"] = processed["input_ids"][:]
    return processed



def process_latex_ocr_data(sample):
    """
        Process LaTeX OCR dataset examples.
        The dataset typically contains 'image' and 'text' (LaTeX formula) fields.
        """
    # For text-only models like Gemma, we'll use the LaTeX text
    # For vision-language models like Qwen2.5-VL, we'd process both image and text
    instruction = "Write the LaTeX representation for this image."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return conversation




def main_gemma_3b(batch_size):
    """Main function for Gemma-3B-IT model"""
    print(f"Testing Gemma-3B-IT with batch size: {batch_size}")

    # Load the LaTeX OCR dataset
    dataset = load_dataset("unsloth/LaTeX_OCR", split="train")

    # Use a smaller subset for testing if dataset is large
    if len(dataset) > 10000:
        dataset = dataset.select(range(10000))

    # Load Gemma-3B-IT model and tokenizer
    model_name = "google/gemma-3-4b-it"

    processor = AutoProcessor.from_pretrained(model_name, padding_side="right", token=hf_token)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Use eos token as pad token
    processor.tokenizer.padding_side = "right"

    dataset = dataset.map(process_latex_ocr_data, remove_columns=dataset.column_names, num_proc=6, desc="Processing LaTeX OCR data")

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",

    )

    model = AutoModelForImageTextToText.from_pretrained(model_name, cache_dir=CACHE_DIR, **model_kwargs)

    model.gradient_checkpointing_enable()


    dataset_tok = dataset.map(UnslothVisionDataCollator(model, processor),batched = True, batch_size = 128, num_proc=1, desc="Tokenizing dataset",)


    # Training arguments
    args = TrainingArguments(
            'outputs_gemma',
            learning_rate=2e-5,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            fp16=True,
            eval_strategy="no",  # Disable eval for memory testing
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            weight_decay=0.01,
            deepspeed=deepspeed_cfg,
            report_to='none',
            save_strategy="no",  # Disable saving for memory testing
            logging_steps=10,
            dataloader_num_workers=0,  # Reduce CPU overhead
    )


    trainer = InterruptableTrainer(
            model,
            args,
            train_dataset=dataset,
            data_collator=UnslothVisionDataCollator(model, processor)

    )

    try:
        trainer.train()
    except InterruptTraining:
        pass
    except torch.cuda.OutOfMemoryError:
        with open("./results_gemma.csv", "a") as f:
            f.write(f"{batch_size}, OOM\n")
            return

    # Collect memory usage statistics
    memory_usages = []
    num_gpus = torch.cuda.device_count()

    for gpu in range(num_gpus):
        if torch.cuda.is_available():
            stats = torch.cuda.memory_stats(device=gpu)
            memory_usages.append((
                    int(stats["active_bytes.all.peak"] / (1024 * 1024)),
                    int(stats["reserved_bytes.all.peak"] / (1024 * 1024))
            ))

    with open("./results_gemma.csv", "a") as f:
        f.write(f"{batch_size}, ")
        for memory_usage in memory_usages:
            active_peak_mib, reserved_peak_mib = memory_usage
            f.write(f"{active_peak_mib}, {reserved_peak_mib}, ")
        f.write(f"{trainer.average_iterations_per_second()}\n")

# Uncomment the following code block to test Qwen2.5-VL model (TODO)
#
# def main_qwen_vl(batch_size):
#     """Main function for Qwen2.5-VL model"""
#     print(f"Testing Qwen2.5-VL with batch size: {batch_size}")
#
#     # Load the LaTeX OCR dataset
#     dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
#
#     # Use a smaller subset for testing
#     if len(dataset) > 5000:  # Smaller subset for VL models (more memory intensive)
#         dataset = dataset.select(range(5000))
#
#     # Load Qwen2.5-VL model and processor
#     model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # or "Qwen/Qwen2.5-VL-3B-Instruct"
#
#     from transformers import Qwen2VLForConditionalGeneration
#     processor = AutoProcessor.from_pretrained(model_name)
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             cache_dir=CACHE_DIR,
#     )
#     model.gradient_checkpointing_enable()
#
#     # Training arguments for VL model
#     args = TrainingArguments(
#             'outputs_qwen_vl',
#             learning_rate=1e-5,  # Lower learning rate for VL models
#             warmup_ratio=0.1,
#             lr_scheduler_type='cosine',
#             fp16=True,
#             evaluation_strategy="no",
#             per_device_train_batch_size=batch_size,
#             gradient_accumulation_steps=2,  # Use gradient accumulation
#             num_train_epochs=1,
#             weight_decay=0.01,
#             deepspeed="ds_config.json",
#             report_to='none',
#             save_strategy="no",
#             logging_steps=10,
#             dataloader_num_workers=0,
#     )
#
#     # Process the dataset for VL model
#     tokenized_dataset = dataset.map(
#             lambda examples: process_vision_language_data(examples, processor),
#             batched=True,
#             remove_columns=dataset.column_names,
#             num_proc=2  # Fewer processes for VL processing
#     )
#
#     trainer = InterruptableTrainer(
#             model,
#             args,
#             train_dataset=tokenized_dataset,
#             tokenizer=processor.tokenizer,
#     )
#
#     try:
#         trainer.train()
#     except InterruptTraining:
#         pass
#     except torch.cuda.OutOfMemoryError:
#         with open("./results_qwen_vl.csv", "a") as f:
#             f.write(f"{batch_size}, OOM\n")
#             return
#
#     # Collect memory usage statistics
#     memory_usages = []
#     num_gpus = torch.cuda.device_count()
#
#     for gpu in range(num_gpus):
#         if torch.cuda.is_available():
#             stats = torch.cuda.memory_stats(device=gpu)
#             memory_usages.append((
#                     int(stats["active_bytes.all.peak"] / (1024 * 1024)),
#                     int(stats["reserved_bytes.all.peak"] / (1024 * 1024))
#             ))
#
#     with open("./results_qwen_vl.csv", "a") as f:
#         f.write(f"{batch_size}, ")
#         for memory_usage in memory_usages:
#             active_peak_mib, reserved_peak_mib = memory_usage
#             f.write(f"{active_peak_mib}, {reserved_peak_mib}, ")
#         f.write(f"{trainer.average_iterations_per_second()}\n")

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if True: # model_type == "gemma": # Uncomment this line to switch between models TODO
        main_gemma_3b(batch_size)
    elif model_type == "qwen_vl":
        main_qwen_vl(batch_size)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}. Use 'gemma' or 'qwen_vl'")


