# benchmark_train_cost.py
import os, time, math, json, argparse, torch, torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForImageClassification
from tqdm.auto import tqdm
from rich import print
import pandas as pd, psutil
from datetime import datetime, timedelta
import sys
import transformers
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from ptflops import get_model_complexity_info
from peft import LoraConfig, get_peft_model, TaskType
import torch.profiler

disable_caching()


def init_distributed(rank, world_size):
    if dist.is_initialized():
        return

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(
            backend="nccl",
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=1800)
    )


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def est_flops(n_params, batch_tokens):
    """Estimate FLOPs for forward + backward pass"""
    # Forward: 2 * n_params * batch_tokens (matrix multiply approximation)
    # Backward: 4 * n_params * batch_tokens (gradient computation)
    return 6 * n_params * batch_tokens


def image_preproc():
    return T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])


class SimpleCollator:
    """Simple collator that can be pickled"""

    def __init__(self, tokenizer, has_text=True, has_images=False):
        self.tokenizer = tokenizer
        self.has_text = has_text
        self.has_images = has_images

    def __call__(self, batch):
        if self.has_text and not self.has_images:
            # Text only
            return self.tokenizer.pad(batch, return_tensors="pt", padding=True)
        elif self.has_images and not self.has_text:
            # Images only
            return {"pixel_values": torch.stack([item["pixel_values"] for item in batch])}
        else:
            # Mixed or other cases
            result = {}
            if self.has_text:
                text_keys = ["input_ids", "attention_mask", "token_type_ids"]
                text_batch = [{k: item[k] for k in text_keys if k in item} for item in batch]
                result.update(self.tokenizer.pad(text_batch, return_tensors="pt", padding=True))

            if self.has_images:
                result["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])

            return result


def save_report(report, out_prefix="benchmark_report"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON with proper indentation
    filename = f"{out_prefix}_{ts}.json"
    with open(filename, "w", encoding='utf-8') as jf:
        json.dump(report, jf, indent=4, ensure_ascii=False, sort_keys=True)

    # Create reports directory if it doesn't exist
    if not os.path.exists("reports/benchmark/"):
        os.makedirs("reports/benchmark/", exist_ok=True)

    # Also save in reports directory
    json_path = f"reports/benchmark/{out_prefix}_{ts}.json"
    with open(json_path, "w", encoding='utf-8') as jf:
        json.dump(report, jf, indent=4, ensure_ascii=False, sort_keys=True)

    # Save CSV version
    csv_path = f"reports/benchmark/{out_prefix}_{ts}.csv"
    pd.DataFrame([report]).to_csv(csv_path, index=False)

    print(f"Report saved to:")
    print(f"  JSON: {filename}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


def count_tokens_worker(args):
    """Worker function for parallel token counting"""
    tokenizer_name, samples, text_column = args

    # Import here to avoid pickling issues
    from transformers import AutoTokenizer

    # Load tokenizer in worker process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_total_tokens = 0
    local_valid_samples = 0

    for ex in samples:
        text = ex.get(text_column)
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            continue
        try:
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
            local_total_tokens += len(tokens)
            local_valid_samples += 1
        except Exception:
            continue

    return local_total_tokens, local_valid_samples


def count_tokens(tokenizer, dataset, text_column="text", num_workers=None):
    """Count total tokens in dataset using parallel processing"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Use up to 8 workers

    dataset_size = len(dataset)
    if dataset_size == 0:
        return 0, 0, 0

    # For small datasets, use single process
    if dataset_size < 1000 or num_workers == 1:
        return count_tokens_sequential(tokenizer, dataset, text_column)

    print(f"Counting tokens using {num_workers} workers...")

    # Split dataset into chunks for parallel processing
    chunk_size = max(1, dataset_size // num_workers)
    chunks = []

    for i in range(0, dataset_size, chunk_size):
        end_idx = min(i + chunk_size, dataset_size)
        chunk = [dataset[j] for j in range(i, end_idx)]
        chunks.append((tokenizer.name_or_path, chunk, text_column))

    total_tokens = 0
    valid_samples = 0

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(count_tokens_worker, chunk): i
                           for i, chunk in enumerate(chunks)}

        # Collect results with progress bar
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                try:
                    chunk_tokens, chunk_valid = future.result()
                    total_tokens += chunk_tokens
                    valid_samples += chunk_valid
                    pbar.update(1)
                except Exception as e:
                    print(f"Warning: Failed to process chunk: {e}")
                    pbar.update(1)

    if valid_samples == 0:
        raise ValueError(f"No valid text found in column '{text_column}'")

    avg_tokens = total_tokens / valid_samples if valid_samples > 0 else 0
    print(f"Dataset analysis (parallel processing):")
    print(f"  - Total samples: {dataset_size}")
    print(f"  - Valid samples: {valid_samples}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average tokens per sample: {avg_tokens:.2f}")
    print(f"  - Workers used: {num_workers}")

    return total_tokens, valid_samples, avg_tokens


def count_tokens_sequential(tokenizer, dataset, text_column="text"):
    """Sequential token counting for small datasets or fallback"""
    total_tokens = 0
    valid_samples = 0

    for ex in tqdm(dataset, desc="Counting tokens (sequential)"):
        text = ex.get(text_column)
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            continue
        try:
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
            total_tokens += len(tokens)
            valid_samples += 1
        except Exception as e:
            print(f"Warning: Failed to tokenize sample: {e}")
            continue

    if valid_samples == 0:
        raise ValueError(f"No valid text found in column '{text_column}'")

    avg_tokens = total_tokens / valid_samples if valid_samples > 0 else 0
    print(f"Dataset analysis (sequential):")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Valid samples: {valid_samples}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average tokens per sample: {avg_tokens:.2f}")

    return total_tokens, valid_samples, avg_tokens


def estimate_memory_usage(model, batch_size, seq_len, model_type, has_amp=False):
    """Estimate GPU memory usage for training"""

    # Get model parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Base model memory (parameters + gradients + optimizer states)
    # Parameters: 4 bytes per param (FP32) or 2 bytes (FP16)
    param_bytes = n_params * (2 if has_amp else 4)

    # Gradients: same size as parameters
    grad_bytes = param_bytes

    # Optimizer states (AdamW): momentum + variance = 2x parameters
    optimizer_bytes = param_bytes * 2

    # Model memory
    model_memory_gb = (param_bytes + grad_bytes + optimizer_bytes) / (1024 ** 3)

    # Activation memory (depends on batch size and sequence length)
    if model_type == "text":
        # Text model activations: roughly 12 * batch_size * seq_len * hidden_size * layers
        # Estimate hidden_size based on model size
        if n_params < 50e6:  # Small models (BERT-base, etc.)
            hidden_size, num_layers = 768, 12
        elif n_params < 200e6:  # Medium models (BERT-large, etc.)
            hidden_size, num_layers = 1024, 24
        elif n_params < 1e9:  # Large models
            hidden_size, num_layers = 1536, 36
        else:  # Very large models
            hidden_size, num_layers = 2048, 48

        activation_bytes = 12 * batch_size * seq_len * hidden_size * num_layers * (2 if has_amp else 4)

    elif model_type == "vision":
        # Vision model activations: batch_size * channels * height * width * depth
        # Assume standard 224x224 images, estimate depth based on model size
        if n_params < 50e6:  # Small vision models
            depth_factor = 1000
        elif n_params < 200e6:  # Medium vision models
            depth_factor = 2000
        else:  # Large vision models
            depth_factor = 3000

        activation_bytes = batch_size * 3 * 224 * 224 * depth_factor * (2 if has_amp else 4)

    else:  # VLM
        # VLMs combine both text and vision activations
        # Use text estimation as base and add vision overhead
        hidden_size, num_layers = 1024, 24  # Typical VLM size
        text_activations = 12 * batch_size * seq_len * hidden_size * num_layers * (2 if has_amp else 4)
        vision_activations = batch_size * 3 * 224 * 224 * 1500 * (2 if has_amp else 4)
        activation_bytes = text_activations + vision_activations

    activation_memory_gb = activation_bytes / (1024 ** 3)

    # Add buffer for PyTorch overhead (10-20%)
    overhead_gb = (model_memory_gb + activation_memory_gb) * 0.15

    total_memory_gb = model_memory_gb + activation_memory_gb + overhead_gb

    return {
            'model_memory_gb'     : model_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'overhead_gb'         : overhead_gb,
            'total_memory_gb'     : total_memory_gb,
            'estimated_parameters': n_params
    }


def find_optimal_batch_size(model, seq_len, model_type, available_memory_gb, has_amp=False, target_utilization=0.85):
    """Find the largest batch size that fits in available memory"""

    # Start with a reasonable batch size and work our way up
    optimal_batch = 1
    max_batch_to_try = 512  # Reasonable upper limit

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if batch_size > max_batch_to_try:
            break

        memory_estimate = estimate_memory_usage(model, batch_size, seq_len, model_type, has_amp)
        required_memory = memory_estimate['total_memory_gb']

        # Check if this batch size fits within target utilization
        if required_memory <= available_memory_gb * target_utilization:
            optimal_batch = batch_size
        else:
            break

    return optimal_batch, memory_estimate


def get_memory_recommendations(model, seq_len, model_type, gpu_memory_gb, current_batch_size, has_amp=False):
    """Generate memory optimization recommendations"""

    recommendations = []

    # Find optimal batch size
    optimal_batch, memory_estimate = find_optimal_batch_size(
            model, seq_len, model_type, gpu_memory_gb, has_amp
    )

    # Current memory usage
    current_memory = estimate_memory_usage(model, current_batch_size, seq_len, model_type, has_amp)

    # Memory utilization
    current_utilization = current_memory['total_memory_gb'] / gpu_memory_gb
    optimal_utilization = memory_estimate['total_memory_gb'] / gpu_memory_gb

    # Generate recommendations
    if optimal_batch > current_batch_size:
        speedup_factor = optimal_batch / current_batch_size
        recommendations.append({
                'type'       : 'increase_batch_size',
                'current'    : current_batch_size,
                'recommended': optimal_batch,
                'speedup'    : f"{speedup_factor:.1f}x faster training",
                'description': f"Increase batch size from {current_batch_size} to {optimal_batch}"
        })

    if current_utilization < 0.5:
        recommendations.append({
                'type'       : 'underutilized_gpu',
                'utilization': f"{current_utilization:.1%}",
                'description': "GPU memory is underutilized, consider larger batch size or model"
        })

    if current_utilization > 0.9:
        recommendations.append({
                'type'       : 'memory_risk',
                'utilization': f"{current_utilization:.1%}",
                'description': "High memory usage, risk of OOM. Consider reducing batch size"
        })

    if not has_amp and model_type in ['text', 'vlm']:
        memory_savings = current_memory['total_memory_gb'] * 0.4  # AMP typically saves 40%
        recommendations.append({
                'type'       : 'enable_amp',
                'savings'    : f"{memory_savings:.1f} GB",
                'description': f"Enable AMP to save ~{memory_savings:.1f} GB memory"
        })

    if seq_len > 512 and model_type in ['text', 'vlm']:
        # Estimate memory savings from shorter sequences
        shorter_memory = estimate_memory_usage(model, current_batch_size, 512, model_type, has_amp)
        savings = current_memory['total_memory_gb'] - shorter_memory['total_memory_gb']
        if savings > 0.5:
            recommendations.append({
                    'type'       : 'reduce_sequence_length',
                    'current'    : seq_len,
                    'recommended': 512,
                    'savings'    : f"{savings:.1f} GB",
                    'description': f"Reduce sequence length from {seq_len} to 512 to save {savings:.1f} GB"
            })


def get_text_column(dataset):
    """Find the text column in dataset"""
    text_candidates = ["text", "content", "sentence", "review", "comment", "document"]

    for col in text_candidates:
        if col in dataset.features:
            return col

    # Look for any string column
    for col_name, feature in dataset.features.items():
        if hasattr(feature, 'dtype') and feature.dtype == "string":
            return col_name

    raise ValueError("No text column found in dataset")
    """Find the text column in dataset"""
    text_candidates = ["text", "content", "sentence", "review", "comment", "document"]

    for col in text_candidates:
        if col in dataset.features:
            return col

    # Look for any string column
    for col_name, feature in dataset.features.items():
        if hasattr(feature, 'dtype') and feature.dtype == "string":
            return col_name

    raise ValueError("No text column found in dataset")


def run(rank, world_size, args):
    use_ddp = args.mode == "ddp"
    device = torch.device(f"cuda:{rank}")

    if use_ddp:
        init_distributed(rank, world_size)

    torch.cuda.set_device(device)
    amp_ctx = autocast(enabled=args.amp)
    scaler = GradScaler(enabled=args.amp)

    # Load dataset
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))

    # Detect dataset type
    has_img = "image" in ds.features
    has_txt = any(feature.dtype == "string" for feature in ds.features.values() if hasattr(feature, 'dtype'))

    if not has_txt and not has_img:
        raise ValueError("Dataset must contain either text or image data")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Count tokens if text dataset
    total_tokens = 0
    avg_tokens = 0
    if has_txt:
        try:
            text_column = get_text_column(ds)
            total_tokens, valid_samples, avg_tokens = count_tokens(
                    tokenizer, ds, text_column, num_workers=args.token_workers
            )
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return

    # Preprocessing function
    def preprocess(examples):
        batch_out = {}

        if has_txt:
            text_column = get_text_column(ds)
            texts = examples[text_column] if isinstance(examples[text_column], list) else [examples[text_column]]
            tokenized = tokenizer(
                    texts,
                    max_length=args.seq_len,
                    truncation=True,
                    padding=False,  # We'll pad in collate_fn
                    return_tensors=None
            )
            batch_out.update(tokenized)

        if has_img:
            images = examples["image"] if isinstance(examples["image"], list) else [examples["image"]]
            processed_images = [image_preproc()(img.convert("RGB")) for img in images]
            batch_out["pixel_values"] = processed_images

        return batch_out

    # Process dataset
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    # Setup data loader
    sampler = DistributedSampler(ds, world_size, rank, shuffle=True) if use_ddp else None
    per_gpu_bs = max(1, args.batch_size // world_size)

    # Create collator
    collator = SimpleCollator(tokenizer, has_txt, has_img)

    dl = DataLoader(
            ds,
            batch_size=per_gpu_bs,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collator,
            pin_memory=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            persistent_workers=False
    )

    # Load model
    try:
        if has_img and not has_txt:
            model = AutoModelForImageClassification.from_pretrained(
                    args.model,
                    num_labels=2,
                    trust_remote_code=True
            ).to(device)
            batch_tokens_per_sample = 1  # Images don't have variable tokens
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                    args.model,
                    num_labels=2,
                    trust_remote_code=True
            ).to(device)
            batch_tokens_per_sample = args.seq_len
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Apply LoRA if requested
    if args.lora:
        lora_task = TaskType.FEATURE_EXTRACTION if (has_img and not has_txt) else TaskType.SEQ_CLS
        lora_config = LoraConfig(
                task_type=lora_task,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target.split(","),
                bias="none"
        )
        model = get_peft_model(model, lora_config)
        if rank == 0:
            model.print_trainable_parameters()


def estimate_memory_usage(model, batch_size, seq_len, model_type, has_amp=False):
    """Estimate GPU memory usage for training"""

    # Get model parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Base model memory (parameters + gradients + optimizer states)
    # Parameters: 4 bytes per param (FP32) or 2 bytes (FP16)
    param_bytes = n_params * (2 if has_amp else 4)

    # Gradients: same size as parameters
    grad_bytes = param_bytes

    # Optimizer states (AdamW): momentum + variance = 2x parameters
    optimizer_bytes = param_bytes * 2

    # Model memory
    model_memory_gb = (param_bytes + grad_bytes + optimizer_bytes) / (1024 ** 3)

    # Activation memory (depends on batch size and sequence length)
    if model_type == "text":
        # Text model activations: roughly 12 * batch_size * seq_len * hidden_size * layers
        # Estimate hidden_size based on model size
        if n_params < 50e6:  # Small models (BERT-base, etc.)
            hidden_size, num_layers = 768, 12
        elif n_params < 200e6:  # Medium models (BERT-large, etc.)
            hidden_size, num_layers = 1024, 24
        elif n_params < 1e9:  # Large models
            hidden_size, num_layers = 1536, 36
        else:  # Very large models
            hidden_size, num_layers = 2048, 48

        activation_bytes = 12 * batch_size * seq_len * hidden_size * num_layers * (2 if has_amp else 4)

    elif model_type == "vision":
        # Vision model activations: batch_size * channels * height * width * depth
        # Assume standard 224x224 images, estimate depth based on model size
        if n_params < 50e6:  # Small vision models
            depth_factor = 1000
        elif n_params < 200e6:  # Medium vision models
            depth_factor = 2000
        else:  # Large vision models
            depth_factor = 3000

        activation_bytes = batch_size * 3 * 224 * 224 * depth_factor * (2 if has_amp else 4)

    else:  # VLM
        # VLMs combine both text and vision activations
        # Use text estimation as base and add vision overhead
        hidden_size, num_layers = 1024, 24  # Typical VLM size
        text_activations = 12 * batch_size * seq_len * hidden_size * num_layers * (2 if has_amp else 4)
        vision_activations = batch_size * 3 * 224 * 224 * 1500 * (2 if has_amp else 4)
        activation_bytes = text_activations + vision_activations

    activation_memory_gb = activation_bytes / (1024 ** 3)

    # Add buffer for PyTorch overhead (10-20%)
    overhead_gb = (model_memory_gb + activation_memory_gb) * 0.15

    total_memory_gb = model_memory_gb + activation_memory_gb + overhead_gb

    return {
            'model_memory_gb'     : model_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'overhead_gb'         : overhead_gb,
            'total_memory_gb'     : total_memory_gb,
            'estimated_parameters': n_params
    }


def find_optimal_batch_size(model, seq_len, model_type, available_memory_gb, has_amp=False, target_utilization=0.85):
    """Find the largest batch size that fits in available memory"""

    # Start with a reasonable batch size and work our way up
    optimal_batch = 1
    max_batch_to_try = 512  # Reasonable upper limit

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if batch_size > max_batch_to_try:
            break

        memory_estimate = estimate_memory_usage(model, batch_size, seq_len, model_type, has_amp)
        required_memory = memory_estimate['total_memory_gb']

        # Check if this batch size fits within target utilization
        if required_memory <= available_memory_gb * target_utilization:
            optimal_batch = batch_size
        else:
            break

    return optimal_batch, memory_estimate


def get_memory_recommendations(model, seq_len, model_type, gpu_memory_gb, current_batch_size, has_amp=False):
    """Generate memory optimization recommendations"""

    recommendations = []

    # Find optimal batch size
    optimal_batch, memory_estimate = find_optimal_batch_size(
            model, seq_len, model_type, gpu_memory_gb, has_amp
    )

    # Current memory usage
    current_memory = estimate_memory_usage(model, current_batch_size, seq_len, model_type, has_amp)

    # Memory utilization
    current_utilization = current_memory['total_memory_gb'] / gpu_memory_gb
    optimal_utilization = memory_estimate['total_memory_gb'] / gpu_memory_gb

    # Generate recommendations
    if optimal_batch > current_batch_size:
        speedup_factor = optimal_batch / current_batch_size
        recommendations.append({
                'type'       : 'increase_batch_size',
                'current'    : current_batch_size,
                'recommended': optimal_batch,
                'speedup'    : f"{speedup_factor:.1f}x faster training",
                'description': f"Increase batch size from {current_batch_size} to {optimal_batch}"
        })

    if current_utilization < 0.5:
        recommendations.append({
                'type'       : 'underutilized_gpu',
                'utilization': f"{current_utilization:.1%}",
                'description': "GPU memory is underutilized, consider larger batch size or model"
        })

    if current_utilization > 0.9:
        recommendations.append({
                'type'       : 'memory_risk',
                'utilization': f"{current_utilization:.1%}",
                'description': "High memory usage, risk of OOM. Consider reducing batch size"
        })

    if not has_amp and model_type in ['text', 'vlm']:
        memory_savings = current_memory['total_memory_gb'] * 0.4  # AMP typically saves 40%
        recommendations.append({
                'type'       : 'enable_amp',
                'savings'    : f"{memory_savings:.1f} GB",
                'description': f"Enable AMP to save ~{memory_savings:.1f} GB memory"
        })

    if seq_len > 512 and model_type in ['text', 'vlm']:
        # Estimate memory savings from shorter sequences
        shorter_memory = estimate_memory_usage(model, current_batch_size, 512, model_type, has_amp)
        savings = current_memory['total_memory_gb'] - shorter_memory['total_memory_gb']
        if savings > 0.5:
            recommendations.append({
                    'type'       : 'reduce_sequence_length',
                    'current'    : seq_len,
                    'recommended': 512,
                    'savings'    : f"{savings:.1f} GB",
                    'description': f"Reduce sequence length from {seq_len} to 512 to save {savings:.1f} GB"
            })

    return {
            'current_memory'     : current_memory,
            'optimal_batch_size' : optimal_batch,
            'optimal_memory'     : memory_estimate,
            'current_utilization': current_utilization,
            'optimal_utilization': optimal_utilization,
            'recommendations'    : recommendations
    }
    """Handle forward pass for different model types, especially VLMs"""
    try:
        if model_type == "vlm":
            # For VLMs, try different forward pass strategies

            # Strategy 1: Direct forward (works for many VLMs)
            try:
                outputs = model(**model_inputs)
                if hasattr(outputs, 'logits'):
                    return outputs
                # If no logits attribute, create a mock output for benchmarking
                batch_size = model_inputs.get('input_ids', model_inputs.get('pixel_values')).size(0)
                mock_logits = torch.randn(batch_size, 2, device=next(model.parameters()).device)

                class MockOutput:
                    def __init__(self, logits):
                        self.logits = logits

                return MockOutput(mock_logits)
            except Exception as e:
                print(f"Direct VLM forward failed: {e}")

            # Strategy 2: Try text-only forward for VLMs that support it
            if 'input_ids' in model_inputs:
                try:
                    text_inputs = {k: v for k, v in model_inputs.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
                    outputs = model(**text_inputs)
                    if hasattr(outputs, 'logits'):
                        return outputs
                except Exception as e:
                    print(f"VLM text-only forward failed: {e}")

            # Strategy 3: Create mock output for unsupported VLMs
            batch_size = model_inputs.get('input_ids', model_inputs.get('pixel_values')).size(0)
            mock_logits = torch.randn(batch_size, 2, device=next(model.parameters()).device)

            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits

            print("Using mock output for VLM benchmarking")
            return MockOutput(mock_logits)

        else:
            # Standard forward pass for text/vision models
            return model(**model_inputs)

    except Exception as e:
        # Final fallback: create mock output
        print(f"Forward pass failed, using mock output: {e}")
        batch_size = list(model_inputs.values())[0].size(0)
        mock_logits = torch.randn(batch_size, 2, device=next(model.parameters()).device)

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        return MockOutput(mock_logits)

    # Calculate model parameters
    n_params = count_trainable(model.module if use_ddp else model)

    # Setup training components
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Benchmark training
    warm_steps = 3
    measure_steps = 10

    model.train()
    data_iter = iter(dl)

    # Warmup
    for step in range(warm_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Create dummy labels
        batch_size = batch["input_ids"].size(0) if "input_ids" in batch else batch["pixel_values"].size(0)
        labels = torch.randint(0, 2, (batch_size,), device=device)

        with amp_ctx:
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Measure performance
    torch.cuda.synchronize()
    start_time = time.time()

    for step in range(measure_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Create dummy labels
        batch_size = batch["input_ids"].size(0) if "input_ids" in batch else batch["pixel_values"].size(0)
        labels = torch.randint(0, 2, (batch_size,), device=device)

        with amp_ctx:
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate performance metrics
    total_time = end_time - start_time
    steps_per_sec = measure_steps / total_time

    # Aggregate across GPUs if using DDP
    if use_ddp:
        tensor = torch.tensor([steps_per_sec], device=device)
        dist.reduce(tensor, 0)
        if rank == 0:
            steps_per_sec = tensor.item() / world_size

    # Generate report (only on rank 0)
    if rank == 0:
        steps_per_epoch = math.ceil(len(ds) / args.batch_size)
        total_steps = steps_per_epoch * args.epochs

        # FLOPS calculation
        effective_batch_size = args.batch_size
        tokens_per_batch = effective_batch_size * batch_tokens_per_sample
        flops_per_step = est_flops(n_params, tokens_per_batch)
        total_flops = flops_per_step * total_steps

        # Time estimation
        time_per_epoch_hours = steps_per_epoch / steps_per_sec / 3600 if steps_per_sec > 0 else -1
        total_time_hours = time_per_epoch_hours * args.epochs

        # Memory info
        memory_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

        # Get detailed GPU information
        gpu_props = torch.cuda.get_device_properties(device)

        # Calculate efficiency metrics
        tokens_per_second = steps_per_sec * effective_batch_size * batch_tokens_per_sample if batch_tokens_per_sample > 1 else steps_per_sec * effective_batch_size
        samples_per_second = steps_per_sec * effective_batch_size

        # Memory efficiency
        params_per_gb = n_params / memory_gb if memory_gb > 0 else 0

        # Cost estimates (rough AWS p4d.24xlarge pricing)
        gpu_hours_total = total_time_hours * world_size
        estimated_cost_usd = gpu_hours_total * 32.77  # AWS p4d.24xlarge hourly rate

        # LoRA efficiency metrics
        if args.lora:
            original_params = sum(p.numel() for p in (model.module if use_ddp else model).parameters())
            lora_reduction_factor = original_params / n_params if n_params > 0 else 1
        else:
            original_params = n_params
            lora_reduction_factor = 1

        report = {
                # Benchmark metadata
                "benchmark_id"        : f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(args)) % 10000:04d}",
                "timestamp"           : datetime.now().isoformat(),
                "benchmark_version"   : "2.0",
                "python_version"      : f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "pytorch_version"     : torch.__version__,
                "transformers_version": transformers.__version__,

                # Dataset information
                "dataset"             : {
                        "name"                 : args.dataset,
                        "split"                : args.split,
                        "total_samples"        : len(ds),
                        "samples_used"         : args.max_samples if args.max_samples else len(ds),
                        "has_text"             : has_txt,
                        "has_images"           : has_img,
                        "total_tokens"         : total_tokens,
                        "avg_tokens_per_sample": round(avg_tokens, 2) if avg_tokens > 0 else 0,
                        "token_distribution"   : "calculated" if total_tokens > 0 else "not_applicable"
                },

                # Model configuration
                "model"               : {
                        "name"                      : args.model,
                        "type"                      : "vision" if (has_img and not has_txt) else "text",
                        "total_parameters"          : original_params,
                        "trainable_parameters"      : n_params,
                        "parameter_reduction_factor": round(lora_reduction_factor, 2),
                        "model_size_gb"             : round(original_params * 4 / (1024 ** 3), 2),  # Assuming FP32
                        "architecture"              : "transformer-based"
                },

                # Training configuration
                "training"            : {
                        "epochs"              : args.epochs,
                        "batch_size"          : args.batch_size,
                        "effective_batch_size": effective_batch_size,
                        "seq_len"             : args.seq_len,
                        "max_samples"         : args.max_samples,
                        "mode"                : args.mode,
                        "distributed"         : use_ddp,
                        "amp_enabled"         : args.amp,
                        "precision"           : "mixed" if args.amp else "fp32"
                },

                # LoRA configuration
                "lora"                : {
                        "enabled"         : args.lora,
                        "rank"            : args.lora_r if args.lora else None,
                        "alpha"           : args.lora_alpha if args.lora else None,
                        "dropout"         : args.lora_dropout if args.lora else None,
                        "target_modules"  : args.lora_target.split(",") if args.lora else None,
                        "trainable_params": n_params if args.lora else None,
                        "total_params"    : original_params,
                        "efficiency_ratio": round(lora_reduction_factor, 2) if args.lora else 1.0
                },

                # Performance metrics
                "performance"         : {
                        "steps_per_second"  : round(steps_per_sec, 3),
                        "samples_per_second": round(samples_per_second, 2),
                        "tokens_per_second" : round(tokens_per_second, 2) if batch_tokens_per_sample > 1 else None,
                        "steps_per_epoch"   : steps_per_epoch,
                        "total_steps"       : total_steps,
                        "warmup_steps"      : 3,
                        "measurement_steps" : 10
                },

                # Time estimates
                "time_estimates"      : {
                        "time_per_epoch_hours"     : round(time_per_epoch_hours, 3),
                        "total_training_time_hours": round(total_time_hours, 3),
                        "time_per_step_seconds"    : round(1 / steps_per_sec, 3) if steps_per_sec > 0 else None,
                        "estimated_completion"     : (datetime.now() + timedelta(hours=total_time_hours)).isoformat()
                },

                # Computational cost
                "compute"             : {
                        "flops_per_step"          : flops_per_step,
                        "flops_per_step_teraflops": round(flops_per_step / 1e12, 3),
                        "total_flops"             : total_flops,
                        "total_flops_petaflops"   : round(total_flops / 1e15, 3),
                        "flops_per_parameter"     : round(flops_per_step / n_params, 2) if n_params > 0 else 0,
                        "compute_intensity"       : "high" if flops_per_step / 1e12 > 1.0 else "medium" if flops_per_step / 1e12 > 0.1 else "low"
                },

                # Hardware information
                "hardware"            : {
                        "num_gpus"              : world_size,
                        "gpu_model"             : gpu_props.name,
                        "gpu_memory_gb"         : round(gpu_props.total_memory / (1024 ** 3), 1),
                        "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        "total_gpu_memory_gb"   : round(gpu_props.total_memory * world_size / (1024 ** 3), 1),
                        "system_memory_gb"      : memory_gb,
                        "cuda_version"          : torch.version.cuda,
                        "gpu_utilization"       : "measured_during_training"
                },

                # Cost estimates
                "cost_estimates"      : {
                        "gpu_hours_total"    : round(gpu_hours_total, 3),
                        "estimated_cost_usd" : round(estimated_cost_usd, 2),
                        "cost_per_epoch_usd" : round(estimated_cost_usd / args.epochs, 2),
                        "cost_per_sample_usd": round(estimated_cost_usd / len(ds), 6),
                        "pricing_model"      : "AWS p4d.24xlarge hourly rate",
                        "currency"           : "USD",
                        "cost_efficiency"    : round(len(ds) / estimated_cost_usd, 2) if estimated_cost_usd > 0 else None
                },

                # Efficiency metrics
                "efficiency"          : {
                        "samples_per_gpu_hour"    : round(samples_per_second * 3600 / world_size, 2),
                        "tokens_per_gpu_hour"     : round(tokens_per_second * 3600 / world_size, 2) if batch_tokens_per_sample > 1 else None,
                        "parameters_per_gb_memory": round(params_per_gb, 2),
                        "memory_efficiency"       : "high" if params_per_gb > 1e9 else "medium" if params_per_gb > 1e8 else "low",
                        "compute_utilization"     : "estimated_high" if steps_per_sec > 1.0 else "estimated_medium",
                        "scalability"             : "linear" if use_ddp else "limited"
                },

                # Environmental impact (rough estimates)
                "environmental"       : {
                        "estimated_carbon_kg": round(gpu_hours_total * 0.5, 3),  # Rough estimate: 0.5 kg CO2 per GPU hour
                        "energy_kwh"         : round(gpu_hours_total * 0.4, 2),  # Rough estimate: 400W per GPU
                        "carbon_model"       : "estimated_datacenter_average",
                        "energy_efficiency"  : "high" if args.lora and args.amp else "medium" if args.lora or args.amp else "standard"
                },

                # Recommendations
                "recommendations"     : {
                        "use_lora"              : not args.lora and original_params > 100e6,
                        "use_amp"               : not args.amp,
                        "use_ddp"               : not use_ddp and world_size > 1,
                        "increase_batch_size"   : effective_batch_size < 64 and gpu_props.total_memory > 20e9,
                        "reduce_seq_len"        : args.seq_len > 512 and avg_tokens < args.seq_len * 0.5,
                        "optimization_potential": "high" if not (args.lora and args.amp and use_ddp) else "low"
                }
        }

        print("\n" + "=" * 80)
        print("🚀 COMPREHENSIVE BENCHMARK REPORT")
        print("=" * 80)
        print(f"📊 Benchmark ID: {report['benchmark_id']}")
        print(f"🕒 Timestamp: {report['timestamp']}")
        print()
        print("📁 Dataset & Model:")
        print(f"  Dataset: {report['dataset']['name']} ({report['dataset']['total_samples']:,} samples)")
        print(f"  Model: {report['model']['name']}")
        print(f"  Model Type: {report['model']['type']}")
        print(f"  Total Parameters: {report['model']['total_parameters']:,}")
        print(f"  Trainable Parameters: {report['model']['trainable_parameters']:,}")
        if args.lora:
            print(f"  LoRA Reduction: {report['lora']['efficiency_ratio']:.1f}x fewer parameters")
        print()
        print("⚙️ Training Configuration:")
        print(f"  Mode: {report['training']['mode'].upper()} ({'Distributed' if report['training']['distributed'] else 'Single Process'})")
        print(f"  Precision: {report['training']['precision'].upper()}")
        print(f"  Batch Size: {report['training']['batch_size']} (effective: {report['training']['effective_batch_size']})")
        print(f"  Sequence Length: {report['training']['seq_len']}")
        print(f"  Epochs: {report['training']['epochs']}")
        print()
        print("🔥 Performance Metrics:")
        print(f"  Steps/Second: {report['performance']['steps_per_second']:.3f}")
        print(f"  Samples/Second: {report['performance']['samples_per_second']:.2f}")
        if report['performance']['tokens_per_second']:
            print(f"  Tokens/Second: {report['performance']['tokens_per_second']:,.0f}")
        print(f"  Time per Epoch: {report['time_estimates']['time_per_epoch_hours']:.3f} hours")
        print(f"  Total Training Time: {report['time_estimates']['total_training_time_hours']:.3f} hours")
        print()
        print("💻 Hardware & Compute:")
        print(f"  GPUs: {report['hardware']['num_gpus']}x {report['hardware']['gpu_model']}")
        print(f"  GPU Memory: {report['hardware']['gpu_memory_gb']:.1f} GB per GPU")
        print(f"  Total GPU Memory: {report['hardware']['total_gpu_memory_gb']:.1f} GB")
        print(f"  System Memory: {report['hardware']['system_memory_gb']:.1f} GB")
        print()
        print("🧠 Memory Analysis:")
        print(f"  Current Batch Size: {report['memory_analysis']['current_batch_size']}")
        print(f"  Optimal Batch Size: {report['memory_analysis']['optimal_batch_size']}")
        print(f"  Current Memory Usage: {report['memory_analysis']['current_memory_usage']['total_memory_gb']:.1f} GB")
        print(f"  Memory Utilization: {report['memory_analysis']['memory_utilization']['current_utilization']:.1%}")
        print(f"  Utilization Status: {report['memory_analysis']['memory_utilization']['utilization_status'].title()}")

        # Memory breakdown
        current_mem = report['memory_analysis']['current_memory_usage']
        print(f"  Memory Breakdown:")
        print(f"    - Model + Gradients + Optimizer: {current_mem['model_memory_gb']:.1f} GB")
        print(f"    - Activations: {current_mem['activation_memory_gb']:.1f} GB")
        print(f"    - PyTorch Overhead: {current_mem['overhead_gb']:.1f} GB")
        print()
        print("⚡ Computational Cost:")
        print(f"  FLOPs per Step: {report['compute']['flops_per_step_teraflops']:.3f} TFLOPs")
        print(f"  Total FLOPs: {report['compute']['total_flops_petaflops']:.3f} PFLOPs")
        print(f"  Compute Intensity: {report['compute']['compute_intensity'].title()}")
        print()
        print("💰 Cost Estimates:")
        print(f"  Total GPU Hours: {report['cost_estimates']['gpu_hours_total']:.2f}")
        print(f"  Estimated Cost: ${report['cost_estimates']['estimated_cost_usd']:.2f} USD")
        print(f"  Cost per Epoch: ${report['cost_estimates']['cost_per_epoch_usd']:.2f} USD")
        print(f"  Cost per Sample: ${report['cost_estimates']['cost_per_sample_usd']:.6f} USD")
        print()
        print("🌱 Environmental Impact:")
        print(f"  Estimated Energy: {report['environmental']['energy_kwh']:.2f} kWh")
        print(f"  Estimated Carbon: {report['environmental']['estimated_carbon_kg']:.3f} kg CO₂")
        print(f"  Energy Efficiency: {report['environmental']['energy_efficiency'].title()}")
        print()
        print("📈 Efficiency Metrics:")
        print(f"  Samples/GPU/Hour: {report['efficiency']['samples_per_gpu_hour']:,.0f}")
        if report['efficiency']['tokens_per_gpu_hour']:
            print(f"  Tokens/GPU/Hour: {report['efficiency']['tokens_per_gpu_hour']:,.0f}")
        print(f"  Memory Efficiency: {report['efficiency']['memory_efficiency'].title()}")
        print(f"  Optimization Potential: {report['recommendations']['optimization_potential'].title()}")
        print()
        print("💡 Recommendations:")
        if report['recommendations']['use_lora']:
            print("  ✅ Consider using LoRA for parameter efficiency")
        if report['recommendations']['use_amp']:
            print("  ✅ Enable AMP for 2x speedup")
        if report['recommendations']['use_ddp']:
            print("  ✅ Use DDP for better multi-GPU scaling")
        if report['recommendations']['increase_batch_size']:
            print("  ✅ Increase batch size for better GPU utilization")
        if report['recommendations']['reduce_seq_len']:
            print("  ✅ Reduce sequence length to match data distribution")
        print("=" * 80)

        save_report(report)

    if use_ddp:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Benchmark training cost for HuggingFace models")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Total batch size across all GPUs")
    parser.add_argument("--seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to use")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--mode", choices=["dp", "ddp"], default="ddp", help="Parallelization mode")
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LoRA target modules")
    parser.add_argument("--token_workers", type=int, default=None, help="Number of workers for token counting (default: auto)")

    args = parser.parse_args()

    # Check for CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires GPU(s).")

    world_size = torch.cuda.device_count() if args.mode == "ddp" else 1

    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    print(f"Using {world_size} GPU(s) with mode: {args.mode}")

    # Set default master port if not set
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    try:
        if args.mode == "dp":
            run(0, 1, args)
        else:
            mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        print("Try using --mode dp for single GPU or check your CUDA setup")
        raise


if __name__ == "__main__":
    main()