Here’s the modified and **enhanced** version of your benchmark markdown file, integrating LoRA PEFT, multimodal support, and precision/scaling configurations:

---

# ⚡ Benchmarking Fine-Tuning Cost (Multimodal + LoRA Ready)

This guide benchmarks the training speed and FLOPs of Hugging Face models using **4 A100 GPUs**, under three scenarios. It supports:

* Multimodal datasets (image + text)
* Mixed precision (AMP)
* PEFT with **LoRA**
* DDP vs DP comparison

| Scenario                    | Precision | Parallel Mode | Launch Command                                               |
| --------------------------- | --------- | ------------- | ------------------------------------------------------------ |
| 1. **4-GPUs FP32**          | FP32      | DDP           | `torchrun --nproc_per_node=4 ... --mode ddp`                 |
| 2. **4-GPUs AMP + LoRA**    | AMP       | DDP           | `torchrun --nproc_per_node=4 ... --mode ddp --amp --lora`    |
| 3. **DP vs DDP comparison** | FP32/AMP  | DP & DDP      | DP: `python ... --mode dp`<br>DDP: `torchrun ... --mode ddp` |

---

## 🔧 Installation

```bash
git clone <your-repo>
cd <your-repo>
pip install datasets transformers torch torchvision tqdm pillow rich psutil pandas peft accelerate
```

---

## ▶️ 1. 4-GPUs FP32

Run the benchmark with full-precision (FP32), 4 GPUs, DDP:

```bash
torchrun --nproc_per_node=4 src/benchmark/benchmark_train_cost.py \
  --dataset="imdb" \
  --model="bert-base-uncased" \
  --batch_size=64 \
  --seq_len=128 \
  --epochs=3 \
  --mode ddp
```

---


## ▶️ 2. 4-GPUs Mixed Precision (AMP) + LoRA

Use LoRA to adapt only parts of the model (e.g. attention projections) with AMP enabled:

```bash
torchrun --nproc_per_node=4 src/benchmark/benchmark_train_cost.py \
  --dataset="imdb" \
  --model="bert-base-uncased" \
  --batch_size=64 \
  --seq_len=128 \
  --epochs=3 \
  --mode ddp \
  --amp \
  --lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target q_proj,v_proj
```

---

## ▶️ 3. DP vs DDP Comparison

### ➤ DataParallel (DP) — single process, slower:

```bash
python src/benchmark/benchmark_train_cost.py \
  --dataset="imdb" \
  --model="bert-base-uncased" \
  --batch_size=64 \
  --seq_len=128 \
  --mode dp
```

### ➤ DistributedDataParallel (DDP) — multi-process, scalable:

```bash
torchrun --nproc_per_node=4 src/benchmark/benchmark_train_cost.py \
  --dataset="imdb" \
  --model="bert-base-uncasedr" \
  --batch_size=64 \
  --seq_len=128 \
  --mode ddp
```

---

## 🧾 Output Format

After each run, the script produces:

```bash
benchmark_report_<timestamp>.json
benchmark_report_<timestamp>.csv
```

Each file includes:

| Field              | Description                        |
| ------------------ | ---------------------------------- |
| `steps_per_sec`    | Global training throughput         |
| `est_wall_hours`   | Projected wall-clock training time |
| `flops_per_step_T` | TFLOPs per training step           |
| `total_flops_P`    | Projected PFLOPs across all epochs |
| `trainable_params` | Trainable parameter count          |
| `lora`, `amp`      | LoRA and precision mode status     |
| `gpus`, `mode`     | GPU count and parallelism strategy |

You can aggregate CSV files to analyze throughput scaling across configurations and models.

---

## 📌 Notes

* If using **LoRA**, only the adapter layers are trainable—saving memory and compute.
* Mixed precision significantly improves speed on A100s and supports memory scaling.
* The dataset must be a **Hugging Face `load_dataset()`-compatible** format, such as one saved with `dataset.save_to_disk()`.

---

Let me know if you'd like a bash launcher for batch jobs or SLURM-based templates.
