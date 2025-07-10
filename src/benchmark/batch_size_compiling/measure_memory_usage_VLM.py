import io
import sys
sys.path.append('./')
import time
import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
global processor
import zipfile
from datasets import DatasetDict
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
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

def format_data(samples: dict[str, any]) -> dict[str, list]:
    formatted_samples = {"messages": []}
    for cont in range(len(samples["question"])):
        images = []
        for img_path in samples["input_image_path"][cont]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        formatted_samples["messages"].append(
            [
                {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
                {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
            ]
        )
    return formatted_samples
def prepare_dataset(dataset: DatasetDict, dataset_name: str, dataset_train_split: str) -> DatasetDict:
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]


    for zip_filename in zip_files:
        zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset", cache_dir=CACHE_DIR)
        extract_folder = zip_filename.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
    return dataset








dataset_name = "FanqingM/MMIU-Benchmark"
# Load Dataset
dataset = load_dataset(dataset_name)

dataset_train_split = "test"


#dataset = prepare_dataset(dataset, dataset_name, dataset_train_split)
dataset = dataset['test'].select(range(1000))
dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)


## Set the token for Hugging Face authentication
hf_token = os.getenv("HF_TOKEN")
login(hf_token)
# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def tokenize_function(processor, examples):
    tokenized = processor(examples["text"], truncation=True, padding="max_length", max_length=2048)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

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
    return {"messages": conversation}



from PIL import Image

# For multi-image cases
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


def collate_fn(examples, processor):


    texts = [processor.apply_chat_template(example, tokenize=False, add_generation_prompt=False).strip() for example in examples["messages"]]
    if "image" in examples:  # single-image
        images = [
            img.convert("RGB") for img in examples["image"]
        ]
    else:  # multi-image
        images = [process_vision_info(example["messages"]) for example in examples]


    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=images, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch  # Return the prepared batch


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



    converted_dataset = dataset.map(process_latex_ocr_data, num_proc=4, desc="Processing LaTeX OCR data")

    converted_dataset = converted_dataset.map(
        lambda examples: collate_fn(examples, processor=processor), num_proc=4, batch_size=128, batched=True
    )

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,

    )

    model = AutoModelForImageTextToText.from_pretrained(model_name, cache_dir=CACHE_DIR, **model_kwargs)

    model.gradient_checkpointing_enable()

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
            train_dataset=converted_dataset,


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
# def process_vision_language_data(examples, processor, max_length=2048):
#     """
#     Process data for vision-language models like Qwen2.5-VL.
#     This handles both image and text inputs.
#     """
#     images = examples["messages"]
#     texts = examples["text"]
#
#     # Create conversation format for VL models
#     conversations = []
#     for i, text in enumerate(texts):
#         conversation = [
#                 {
#                         "role"   : "user",
#                         "content": [
#                                 {"type": "image"},
#                                 {"type": "text", "text": "What is the LaTeX formula shown in this image?"}
#                         ]
#                 },
#                 {
#                         "role"   : "assistant",
#                         "content": [{"type": "text", "text": text}]
#                 }
#         ]
#         conversations.append(conversation)
#
#     # Process with the processor (handles both images and text)
#     processed = processor(
#             conversations,
#             images,
#             truncation=True,
#             padding="max_length",
#             max_length=max_length
#     )
#     processed["labels"] = processed["input_ids"][:]
#     return processed

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


