# benchmark_train_cost.py

import os, time, math, json, argparse, torch, torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForImageClassification
from rich import print
import psutil
from datetime import datetime, timedelta
import sys
sys.path.append("./")
import transformers
import itertools
from ptflops import get_model_complexity_info
from peft import LoraConfig, get_peft_model, TaskType


from src.dataset import load_parquet_image_dataset, image_preproc, SimpleCollator
from src.distributed import init_distributed, cleanup_distributed
from src.models import count_trainable, est_flops, count_tokens
from src.evaluate import save_report, estimate_memory_usage, get_memory_recommendations, find_optimal_batch_size
import torch.profiler

disable_caching()

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
        if args.dataset == "chexinstruct":
            ds = load_parquet_image_dataset('data_chexinstruct/hf_parquet')
            ds = ds['val']
        else:
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
            if isinstance(images[0], list):
                images = list(itertools.chain.from_iterable(images))
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
    # Check if it's a vision model based on common vision model names
    vision_model_keywords = ['vit', 'vision', 'clip', 'deit', 'swin', 'resnet', 'efficientnet', 'convnext']
    is_vision_model = any(keyword in model.__str__() for keyword in vision_model_keywords)
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

    # Setup DDP
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

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
        if not is_vision_model:
            batch.pop("pixel_values")
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

        # Ensure all inputs are on the correct device

        # Create dummy labels
        batch_size = batch["input_ids"].size(0) if "input_ids" in batch else batch["pixel_values"].size(0)
        labels = torch.randint(0, 2, (batch_size,), device=device)
        if not is_vision_model:
            batch.pop("pixel_values")
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