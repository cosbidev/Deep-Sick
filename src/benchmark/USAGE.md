# ⚡ Benchmarking Fine-Tuning Cost (Multimodal + LoRA Ready)

This benchmark measures training speed, FLOPs, and costs on GPU clusters. It supports:

* **Multimodal datasets** (text, image, or both)
* **Mixed precision training** (AMP)
* **Efficient fine-tuning** with LoRA
* **Distributed training** with DDP vs DP comparison
* **Comprehensive metrics**: tokens/sec, FLOPs, memory usage, time estimation

| Scenario | Precision | Parallel Mode | Launch Command |
| -------- | --------- | ------------- | -------------- |
| Basic | FP32 | DDP | `torchrun --nproc_per_node=4 benchmark_train_cost.py --mode ddp` |
| Optimized | AMP + LoRA | DDP | `torchrun --nproc_per_node=4 benchmark_train_cost.py --mode ddp --amp --lora` |
| Comparison | FP32/AMP | DP vs DDP | `python benchmark_train_cost.py --mode dp` / `torchrun ... --mode ddp` |

---

## 🔧 Installation

```bash
# Clone and setup
git clone <your-repo>
cd <your-repo>

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets transformers accelerate peft
pip install tqdm rich psutil pandas ptflops
```

---

## ▶️ 1. Basic Text Classification (4 GPUs, FP32)

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29500 \
  benchmark_train_cost.py \
    --dataset="imdb" \
    --model="bert-base-uncased" \
    --batch_size=64 \
    --seq_len=512 \
    --epochs=3 \
    --mode=ddp
```

**Expected Output:**
- Steps per second: ~2.5
- Time per epoch: ~1.2 hours
- Total training time: ~3.6 hours
- FLOPs per step: ~0.8 TFLOPs

---

## ▶️ 2. Optimized Training (AMP + LoRA)

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29501 \
  benchmark_train_cost.py \
    --dataset="imdb" \
    --model="bert-base-uncased" \
    --batch_size=128 \
    --seq_len=512 \
    --epochs=3 \
    --mode=ddp \
    --amp \
    --lora \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --lora_target="q_proj,v_proj,k_proj,out_proj"
```

**Benefits:**
- **2x faster** with AMP (mixed precision)
- **10x fewer parameters** with LoRA
- **50% less memory** usage
- **Same accuracy** as full fine-tuning

---

## ▶️ 3. Large Language Models

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port=29502 \
  benchmark_train_cost.py \
    --dataset="squad" \
    --model="microsoft/DialoGPT-large" \
    --batch_size=32 \
    --seq_len=1024 \
    --epochs=1 \
    --mode=ddp \
    --amp \
    --lora \
    --lora_r=64 \
    --max_samples=50000
```

---

## ▶️ 4. Vision Models

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29503 \
  benchmark_train_cost.py \
    --dataset="cifar10" \
    --model="google/vit-base-patch16-224" \
    --batch_size=256 \
    --epochs=10 \
    --mode=ddp \
    --amp
```

---

## ▶️ 5. Performance Comparison: DP vs DDP

### DataParallel (Single Process)
```bash
python benchmark_train_cost.py \
  --dataset="imdb" \
  --model="distilbert-base-uncased" \
  --batch_size=64 \
  --seq_len=256 \
  --epochs=1 \
  --mode=dp \
  --amp
```

### DistributedDataParallel (Multi Process)
```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29504 \
  benchmark_train_cost.py \
    --dataset="imdb" \
    --model="distilbert-base-uncased" \
    --batch_size=64 \
    --seq_len=256 \
    --epochs=1 \
    --mode=ddp \
    --amp
```

**Expected Results:**
- DDP: ~3x faster than DP
- DDP: Better memory distribution
- DDP: More stable training

---

## 🚀 Multi-Node Distributed Training

### Node 0 (Master):
```bash
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr="192.168.1.100" \
  --master_port=29505 \
  benchmark_train_cost.py \
    --dataset="wikitext" \
    --model="gpt2-medium" \
    --batch_size=128 \
    --seq_len=1024 \
    --epochs=3 \
    --mode=ddp \
    --amp \
    --lora
```

### Node 1 (Worker):
```bash
torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=8 \
  --master_addr="192.168.1.100" \
  --master_port=29505 \
  benchmark_train_cost.py \
    --dataset="wikitext" \
    --model="gpt2-medium" \
    --batch_size=128 \
    --seq_len=1024 \
    --epochs=3 \
    --mode=ddp \
    --amp \
    --lora
```

---

## 🔄 Elastic Training (Fault Tolerance)

```bash
torchrun \
  --nnodes=1:4 \
  --nproc_per_node=8 \
  --max_restarts=3 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="auto" \
  benchmark_train_cost.py \
    --dataset="openwebtext" \
    --model="gpt2-large" \
    --batch_size=64 \
    --seq_len=2048 \
    --epochs=1 \
    --mode=ddp \
    --amp \
    --lora
```

**Features:**
- **Auto-scaling**: 1-4 nodes dynamically
- **Fault tolerance**: Restart failed workers
- **Checkpointing**: Resume from interruptions

---

## 🎯 SLURM Integration

### SLURM Job Script:
```bash
#!/bin/bash
#SBATCH --job-name=benchmark_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Get SLURM variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29506

# Launch training
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --node_rank=$SLURM_PROCID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  benchmark_train_cost.py \
    --dataset="c4" \
    --model="t5-large" \
    --batch_size=64 \
    --seq_len=512 \
    --epochs=3 \
    --mode=ddp \
    --amp \
    --lora \
    --max_samples=1000000
```

---

## 📊 Understanding the Output

### Console Output:
```
======================================================================
BENCHMARK REPORT
======================================================================
Model: bert-base-uncased
Dataset: imdb (25000 samples)
Trainable Parameters: 2,345,678
Steps per second: 2.456
Time per epoch: 1.234 hours
Total training time: 3.702 hours
FLOPs per step: 0.856 TFLOPs
Total FLOPs: 2.341 PFLOPs
======================================================================
```

### Generated Files:
- `benchmark_report_20241201_143022.json` - Detailed metrics
- `reports/benchmark/benchmark_report_20241201_143022.csv` - Spreadsheet format

### Key Metrics:
- **steps_per_second**: Training throughput
- **time_per_epoch_hours**: Time estimation per epoch
- **flops_per_step_teraflops**: Computational intensity
- **trainable_params**: Model complexity
- **total_tokens**: Dataset size in tokens

---

## 🛠️ Advanced Usage

### Custom Datasets:
```bash
# Local dataset
python benchmark_train_cost.py \
  --dataset="json" \
  --dataset_path="./data/custom_dataset.json" \
  --model="roberta-base"

# Streaming dataset (large)
python benchmark_train_cost.py \
  --dataset="streaming/openwebtext" \
  --model="gpt2" \
  --max_samples=100000
```

### Memory Optimization:
```bash
# Gradient checkpointing + small batch
torchrun --nproc_per_node=4 benchmark_train_cost.py \
  --dataset="squad" \
  --model="t5-3b" \
  --batch_size=8 \
  --seq_len=512 \
  --amp \
  --lora \
  --lora_r=32
```

### Profiling Mode:
```bash
# Generate detailed profiling data
torchrun --nproc_per_node=1 benchmark_train_cost.py \
  --dataset="imdb" \
  --model="bert-base-uncased" \
  --batch_size=16 \
  --epochs=1 \
  --max_samples=1000 \
  --mode=ddp
```

---

## ⚠️ Important Notes

### Common Issues:
1. **Port conflicts**: Use different `--master_port` for concurrent jobs
2. **Memory errors**: Reduce `--batch_size` or `--seq_len`
3. **Tokenizer issues**: Some models need `trust_remote_code=True`
4. **CUDA OOM**: Enable `--amp` and reduce batch size

### Best Practices:
- **Use DDP** instead of DP for multi-GPU
- **Enable AMP** for 2x speedup on modern GPUs
- **Use LoRA** for large models to reduce memory
- **Set appropriate** `--max_samples` for quick testing
- **Monitor GPU memory** usage during benchmarking

### Troubleshooting:
```bash
# Check GPU availability
nvidia-smi

# Test single GPU first
python benchmark_train_cost.py --dataset="imdb" --model="bert-base-uncased" --mode=dp --max_samples=100

# Debug distributed setup
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --nproc_per_node=2 benchmark_train_cost.py --dataset="imdb" --model="bert-base-uncased" --mode=ddp --max_samples=100
```

---

## 📈 Expected Performance

### BERT-base on IMDB (4x A100):
- **Full fine-tuning**: 2.5 steps/sec, 3.7 hours total
- **LoRA fine-tuning**: 4.2 steps/sec, 2.1 hours total
- **Memory usage**: 18GB → 8GB with LoRA

### GPT-2 on WikiText (8x A100):
- **Full fine-tuning**: 1.2 steps/sec, 12.3 hours total
- **LoRA fine-tuning**: 2.8 steps/sec, 5.1 hours total

### Vision Transformer on CIFAR-10 (4x A100):
- **Full fine-tuning**: 5.6 steps/sec, 0.8 hours total
- **With AMP**: 8.9 steps/sec, 0.5 hours total


# VLM Benchmark Usage Guide

Complete guide for benchmarking Vision-Language Models (Qwen-2.5VL and PaliGemma) with optimized fine-tuning configurations.

## 🚀 Quick Start

### Basic Installation
```bash
# Core dependencies
pip install torch torchvision transformers accelerate peft
pip install datasets tqdm rich pandas psutil bitsandbytes
pip install pillow numpy

# Optional: Flash Attention 2 (for PaliGemma)
pip install flash-attn --no-build-isolation
```

### Simple Benchmark
```bash
# Quick test with Qwen-2.5VL 3B
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 3b \
  --batch_size 4

# Quick test with PaliGemma 3B
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 8
```

## 📋 Complete Command Reference

### Required Arguments
- `--model_family`: Choose `qwen25vl` or `paligemma`
- `--model_size`: Model variant (see [Model Variants](#model-variants))

### Optimization Arguments (Recommended: Enable All)
- `--use_lora`: Enable LoRA fine-tuning (default: True)
- `--use_amp`: Enable Automatic Mixed Precision (default: True)
- `--use_flash_attn`: Enable Flash Attention 2 (default: True)
- `--vision_lora`: Apply LoRA to vision components (default: True)

### Advanced Configuration
- `--use_quantization`: Enable 4-bit quantization (QLoRA)
- `--lora_rank`: LoRA rank, 8-128 (default: 16)
- `--lora_alpha`: LoRA alpha, typically 2x rank (default: 32)
- `--batch_size`: Batch size per GPU (default: 4)
- `--learning_rate`: Base learning rate (default: 1e-4)

### Benchmark Settings
- `--warmup_steps`: Warmup iterations (default: 10)
- `--benchmark_steps`: Measurement iterations (default: 50)
- `--total_steps`: Steps for cost estimation (default: 1000)

### Control Options
- `--disable_ddp`: Force single GPU mode
- `--force_cpu`: CPU-only benchmarking (not recommended)

## 🎯 Model Variants

### Qwen-2.5VL Models
| Size | Model ID | Parameters | Memory (FP16) | Memory (4-bit) |
|------|----------|------------|---------------|----------------|
| 3b   | Qwen/Qwen2.5-VL-3B-Instruct | 3.4B | 8 GB | 3 GB |
| 7b   | Qwen/Qwen2.5-VL-7B-Instruct | 7.6B | 16 GB | 5 GB |
| 32b  | Qwen/Qwen2.5-VL-32B-Instruct | 32.8B | 66 GB | 18 GB |
| 72b  | Qwen/Qwen2.5-VL-72B-Instruct | 72.7B | 145 GB | 36 GB |

### PaliGemma Models
| Size | Model ID | Parameters | Memory (FP16) | Memory (4-bit) |
|------|----------|------------|---------------|----------------|
| 3b   | google/paligemma2-3b-pt-224 | 3B | 8 GB | 3 GB |
| 3b_mix | google/paligemma2-3b-mix-224 | 3B | 8 GB | 3 GB |
| 10b  | google/paligemma2-10b-pt-224 | 10B | 24 GB | 8 GB |
| 28b  | google/paligemma2-28b-pt-224 | 28B | 60 GB | 18 GB |

## 🧪 Experimental Scenarios

### 1. Performance Optimization Experiments

#### Maximum Performance (High-End GPUs)
```bash
# 8x A100/H100 setup - Maximum throughput
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 16 \
  --use_lora \
  --vision_lora \
  --use_amp \
  --use_flash_attn \
  --lora_rank 32 \
  --lora_alpha 64
```

#### Memory-Optimized (Consumer GPUs)
```bash
# Single RTX 4090/3090 - Maximum efficiency
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 4 \
  --use_quantization \
  --use_lora \
  --use_amp \
  --lora_rank 8 \
  --lora_alpha 16
```

#### Ultra-Low Memory (RTX 3060/4060)
```bash
# Extreme memory constraints
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 3b \
  --batch_size 1 \
  --use_quantization \
  --use_lora \
  --use_amp \
  --lora_rank 4 \
  --lora_alpha 8
```

### 2. LoRA Configuration Experiments

#### Minimal LoRA (Fastest Training)
```bash
# Minimum parameter adaptation
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 8 \
  --use_lora \
  --lora_rank 4 \
  --lora_alpha 8 \
  --vision_lora
```

#### Balanced LoRA (Recommended)
```bash
# Balance between speed and adaptation
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 6 \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --vision_lora \
  --use_amp
```

#### High-Rank LoRA (Maximum Adaptation)
```bash
# Maximum parameter adaptation
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 4 \
  --use_lora \
  --lora_rank 128 \
  --lora_alpha 256 \
  --vision_lora \
  --use_amp
```

#### Vision-Only LoRA
```bash
# LoRA only on vision components
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 8 \
  --use_lora \
  --vision_lora \
  --lora_rank 32 \
  # Language model frozen, vision adapted
```

### 3. Quantization Experiments

#### QLoRA vs Standard LoRA
```bash
# Standard LoRA (higher memory, faster)
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 4 \
  --use_lora \
  --use_amp

# QLoRA (lower memory, slower)
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 8 \
  --use_quantization \
  --use_lora \
  --use_amp
```

#### Large Model Quantization
```bash
# 32B model with aggressive quantization
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 32b \
  --batch_size 2 \
  --use_quantization \
  --use_lora \
  --use_amp \
  --lora_rank 64
```

### 4. Scaling Experiments

#### Single GPU vs Multi-GPU
```bash
# Force single GPU
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 10b \
  --batch_size 4 \
  --disable_ddp \
  --use_lora \
  --use_amp

# Multi-GPU (automatic DDP)
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 10b \
  --batch_size 16 \
  --use_lora \
  --use_amp
```

#### Batch Size Scaling Study
```bash
# Small batch (more updates)
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 2 \
  --use_lora

# Large batch (fewer updates)
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 32 \
  --use_lora
```

### 5. Model Comparison Experiments

#### Qwen vs PaliGemma (Same Memory Budget)
```bash
# Qwen-2.5VL 3B
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 3b \
  --batch_size 8 \
  --use_lora \
  --use_amp

# PaliGemma 3B
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 8 \
  --use_lora \
  --use_amp \
  --use_flash_attn
```

#### Size vs Performance Trade-off
```bash
# Small model, large batch
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 16 \
  --use_lora \
  --use_amp

# Large model, small batch
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 4 \
  --use_lora \
  --use_amp
```

### 6. Learning Rate Experiments

#### Conservative Learning Rates
```bash
# Low learning rate for stable training
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --use_lora \
  --use_amp
```

#### Aggressive Learning Rates
```bash
# High learning rate for fast adaptation
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --use_lora \
  --use_amp
```

## 🏆 Recommended Configurations

### For Development/Testing
```bash
# Fast iteration, minimal resources
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 4 \
  --use_lora \
  --use_amp \
  --use_flash_attn \
  --warmup_steps 5 \
  --benchmark_steps 20
```

### For Production Benchmarking
```bash
# Comprehensive evaluation
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 8 \
  --use_lora \
  --vision_lora \
  --use_amp \
  --lora_rank 32 \
  --warmup_steps 20 \
  --benchmark_steps 100
```

### For Memory-Constrained Environments
```bash
# Maximum efficiency
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 3b \
  --batch_size 2 \
  --use_quantization \
  --use_lora \
  --use_amp \
  --lora_rank 8
```

### For High-Performance Computing
```bash
# Maximum throughput
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 32b \
  --batch_size 16 \
  --use_lora \
  --vision_lora \
  --use_amp \
  --lora_rank 64 \
  --lora_alpha 128
```

## 📊 Interpreting Results

### Performance Metrics
- **Steps/Second**: Higher is better (target: >1.0)
- **GPU Efficiency**: Percentage of successful training steps
- **Memory Utilization**: Optimal range 70-90%
- **Cost Estimates**: Based on AWS p4d.24xlarge pricing

### Performance Tiers
| Tier | Steps/Sec | Status | Recommendations |
|------|-----------|--------|-----------------|
| 🏆 Excellent | ≥2.0 | Optimal | Consider larger models or batches |
| ✅ Good | 1.0-2.0 | Solid | Fine-tune hyperparameters |
| ⚠️ Fair | 0.5-1.0 | Acceptable | Enable more optimizations |
| ❌ Poor | <0.5 | Needs work | Reduce model size or batch |

### Key Optimization Indicators
- **LoRA Reduction Factor**: 50-200x is typical
- **Memory Breakdown**: Model vs Activations vs Overhead
- **Optimization Score**: 80-100 indicates well-configured setup

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Solutions in order of preference:
# 1. Enable quantization
--use_quantization

# 2. Reduce batch size
--batch_size 1

# 3. Reduce LoRA rank
--lora_rank 4 --lora_alpha 8

# 4. Use smaller model
--model_size 3b
```

#### Slow Performance
```bash
# Enable all optimizations:
--use_amp --use_lora --use_flash_attn --vision_lora

# Increase batch size if memory allows:
--batch_size 16

# Use DDP on multiple GPUs (automatic)
# Ensure CUDA_VISIBLE_DEVICES includes all GPUs
```

#### Model Loading Failures
```bash
# Try basic configuration:
python vlm_benchmark_optimized.py \
  --model_family paligemma \
  --model_size 3b \
  --batch_size 2

# Check available models:
python -c "from transformers import AutoModel; print('Transformers version:', transformers.__version__)"
```

### Hardware-Specific Recommendations

#### RTX 3060 (12GB)
```bash
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 3b \
  --batch_size 1 \
  --use_quantization \
  --use_lora \
  --lora_rank 8
```

#### RTX 4090 (24GB)
```bash
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 4 \
  --use_lora \
  --use_amp \
  --lora_rank 16
```

#### A100 (40GB)
```bash
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 7b \
  --batch_size 12 \
  --use_lora \
  --vision_lora \
  --use_amp \
  --lora_rank 32
```

#### A100 (80GB)
```bash
python vlm_benchmark_optimized.py \
  --model_family qwen25vl \
  --model_size 32b \
  --batch_size 8 \
  --use_lora \
  --vision_lora \
  --use_amp \
  --lora_rank 64
```

## 📈 Advanced Experiment Design

### Ablation Studies

#### Optimization Feature Ablation
```bash
# Baseline (no optimizations)
python vlm_benchmark_optimized.py --model_family paligemma --model_size 3b --batch_size 2

# +LoRA
python vlm_benchmark_optimized.py --model_family paligemma --model_size 3b --batch_size 2 --use_lora

# +LoRA +AMP
python vlm_benchmark_optimized.py --model_family paligemma --model_size 3b --batch_size 4 --use_lora --use_amp

# +LoRA +AMP +Flash Attention
python vlm_benchmark_optimized.py --model_family paligemma --model_size 3b --batch_size 4 --use_lora --use_amp --use_flash_attn

# +LoRA +AMP +Flash Attention +Vision LoRA
python vlm_benchmark_optimized.py --model_family paligemma --model_size 3b --batch_size 4 --use_lora --use_amp --use_flash_attn --vision_lora
```

#### LoRA Rank Sweep
```bash
for rank in 4 8 16 32 64 128; do
  echo "Testing LoRA rank: $rank"
  python vlm_benchmark_optimized.py \
    --model_family qwen25vl \
    --model_size 7b \
    --batch_size 4 \
    --use_lora \
    --lora_rank $rank \
    --lora_alpha $((rank * 2))
done
```

#### Batch Size Scaling
```bash
for bs in 1 2 4 8 16; do
  echo "Testing batch size: $bs"
  python vlm_benchmark_optimized.py \
    --model_family paligemma \
    --model_size 3b \
    --batch_size $bs \
    --use_lora \
    --use_amp
done
```

### Comparative Analysis

#### Cross-Architecture Comparison
```bash
# Script to compare all architectures
#!/bin/bash
models=("qwen25vl:3b" "qwen25vl:7b" "paligemma:3b" "paligemma:10b")

for model in "${models[@]}"; do
  IFS=':' read -r family size <<< "$model"
  echo "Benchmarking $family-$size"
  
  python vlm_benchmark_optimized.py \
    --model_family $family \
    --model_size $size \
    --batch_size 4 \
    --use_lora \
    --use_amp \
    --vision_lora
done
```

## 📁 Output Files

### Generated Reports
```
reports/vlm_benchmark/
├── vlm_benchmark_20250102_143022.json  # Detailed metrics
├── vlm_benchmark_20250102_143022.csv   # Summary table
└── ...
```

### JSON Report Structure
```json
{
  "benchmark_id": "vlm_bench_20250102_143022",
  "model": {
    "family": "qwen25vl",
    "size": "7b",
    "trainable_parameters": 67108864,
    "lora_enabled": true
  },
  "performance": {
    "steps_per_second": 1.847,
    "samples_per_second": 7.39,
    "gpu_efficiency": 0.96
  },
  "memory_analysis": {
    "current_memory_usage_gb": 15.2,
    "optimal_batch_size": 8,
    "memory_utilization": 0.76
  },
  "recommendations": {
    "optimization_score": 85,
    "memory_optimizations": [...]
  }
}
```

## 🎯 Best Practices

1. **Start Small**: Begin with 3B models and small batches
2. **Enable Core Optimizations**: Always use `--use_lora --use_amp`
3. **Monitor Memory**: Check utilization before scaling up
4. **Use Quantization**: For memory-constrained setups
5. **Benchmark Systematically**: Run multiple configurations
6. **Document Results**: Save and compare benchmark outputs
7. **Consider Total Cost**: Balance performance vs resource usage

## 🆘 Support & Debugging

### Enable Debug Mode
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

python vlm_benchmark_optimized.py [your_args] --warmup_steps 2 --benchmark_steps 5
```

### Check System Compatibility
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory/1024**3:.1f} GB)')
"
```

For additional support, check the benchmark output logs and error messages for specific guidance.