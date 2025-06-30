# benchmark_train_cost.py
"""
Benchmark fine-tuning throughput, wall-clock, and FLOPs for a Hugging Face model
on a multimodal dataset (image + text).  Works in:
  • DP   (single process, DataParallel)
  • DDP  (torchrun, multi-process)

It saves a JSON *and* CSV file with the metrics.

Dependencies:
  pip install datasets transformers accelerate torch torchvision tqdm pillow rich psutil pandas
"""
import os, time, math, json, argparse, csv, random, warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch, torchvision.transforms as T
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset, disable_caching
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForImageClassification)
from tqdm.auto import tqdm
from rich import print
import pandas as pd
import psutil

disable_caching()   # avoid arrow tmp clutter
from datetime import timedelta

# ───────────────────────────────────────── utils ─────────────────────────────────────────
def init_distributed(rank, world_size):
    # Uses env. variables set by torchrun (or SLURM)
    if dist.is_initialized():
        return
    dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=1800)  # ✅ Correct
    )

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def est_flops(n_params, batch_tokens):
    # Kaplan et al. rule-of-thumb FLOPs/step
    return 6 * n_params * batch_tokens   # forward + backward

def image_preproc():
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor()
    ])

def save_report(report, out_prefix="benchmark_report"):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{out_prefix}_{ts}.json"
    csv_path  = f"{out_prefix}_{ts}.csv"
    with open(json_path, "w") as jf:
        json.dump(report, jf, indent=2)
    # flatten dict for CSV (single row)
    pd.DataFrame([report]).to_csv(csv_path, index=False)
    if report["rank"] == 0:
        print(f"[green]✓ Report saved to[/green] {json_path}  &  {csv_path}")

# ───────────────────────────────────────── main worker ────────────────────────────────────
# benchmark_train_cost.py  (LoRA-ready)

import os, time, math, json, argparse, random
from datetime import datetime
import psutil, torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from datasets import load_dataset, disable_caching
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForImageClassification)
from tqdm.auto import tqdm
from rich import print
import pandas as pd


# NEW ───────────────────────────────
from peft import LoraConfig, get_peft_model, TaskType
# ───────────────────────────────────


disable_caching()

# …  (utility functions unchanged: init_distributed, cleanup_distributed,
#     count_trainable, est_flops, image_preproc, save_report) …

# ───────────────────────── MAIN WORKER ──────────────────────────
def run(rank, world_size, args):
    use_ddp = args.mode == "ddp"
    device  = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    amp_ctx = autocast(enabled=args.amp)
    scaler  = GradScaler(enabled=args.amp)

    if use_ddp:
        init_distributed(rank, world_size)

    # ───── DATASET  (unchanged) ─────
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(len(ds), args.max_samples)))

    has_img = "image" in ds.features
    has_txt = "text" in ds.column_names or any(col for col in ds.column_names
                                               if ds[col].dtype == 'string')

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    def preprocess(ex):
        out = {}
        if has_txt:
            text = ex.get("text") or ex[next(k for k in ex if isinstance(ex[k], str))]
            out.update(tok(text, max_length=args.seq_len, truncation=True))
        if has_img:
            out["pixel_values"] = image_preproc()(ex["image"].convert("RGB"))
        return out

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    sampler = DistributedSampler(ds, world_size, rank, shuffle=True) if use_ddp else None
    per_gpu_bs = max(1, args.batch_size // world_size)
    dl = DataLoader(ds, batch_size=per_gpu_bs, sampler=sampler,
                    shuffle=sampler is None,
                    collate_fn=lambda b: tok.pad(b, return_tensors="pt"),
                    pin_memory=True)

    # ───── MODEL ─────
    if has_img and not has_txt:
        model = AutoModelForImageClassification.from_pretrained(args.model).to(device)
        batch_tok_per_samp = 1
        lora_task = TaskType.FEATURE_EXTRACTION
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2).to(device)
        batch_tok_per_samp = args.seq_len
        lora_task = TaskType.SEQ_CLS

    # NEW ─── apply LoRA if requested ───
    if args.lora:
        targets = [m.strip() for m in args.lora_target.split(",")]
        lconf = LoraConfig(
            task_type   = lora_task,
            r           = args.lora_r,
            lora_alpha  = args.lora_alpha,
            lora_dropout= args.lora_dropout,
            target_modules=targets,
            bias="none"
        )
        model = get_peft_model(model, lconf)
        model.print_trainable_parameters() if rank == 0 else None

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank)

    # ───── BENCHMARK  (loop unchanged) ─────
    model.train()
    loss_fn, warm, meas = torch.nn.CrossEntropyLoss(), 2, 10
    it, total_t = iter(dl), 0.0
    for step in range(warm + meas):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = torch.zeros(
            batch.get("input_ids", batch["pixel_values"]).size(0),
            dtype=torch.long, device=device)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        with amp_ctx:
            logits = model(**batch).logits
            loss   = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           model.parameters()), lr=1e-3))
        scaler.update()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if step >= warm: total_t += (t1 - t0)
        model.zero_grad(set_to_none=True)

    steps_ps = meas / total_t
    if use_ddp:
        steps_tensor = torch.tensor([steps_ps], device=device)
        dist.reduce(steps_tensor, 0, dist.ReduceOp.SUM)
        if rank == 0: steps_ps = steps_tensor.item() / world_size

    # ───── REPORT (rank-0) ─────
    if rank == 0:
        train_params = count_trainable(model.module if use_ddp else model)
        tot_steps    = math.ceil(len(ds) / args.batch_size) * args.epochs
        hours        = tot_steps / steps_ps / 3600
        flop_step    = est_flops(train_params, args.batch_size * batch_tok_per_samp)
        tot_flops    = flop_step * tot_steps

        rep = dict(timestamp=str(datetime.now()),
                   dataset=args.dataset, model=args.model,
                   lora=args.lora, lora_r=args.lora_r if args.lora else 0,
                   lora_target=args.lora_target if args.lora else "",
                   epochs=args.epochs, batch_size=args.batch_size,
                   seq_len=args.seq_len, amp=args.amp, mode=args.mode,
                   gpus=world_size, steps_per_sec=round(steps_ps,2),
                   est_wall_hours=round(hours,2),
                   flops_per_step_T=round(flop_step/1e12,2),
                   total_flops_P=round(tot_flops/1e15,2),
                   trainable_params=train_params,
                   rank=rank,
                   mem_gb=round(psutil.virtual_memory().total/2**30,1))
        save_report(rep)

    if use_ddp:
        cleanup_distributed()

# ───────────────────────── ENTRY ──────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model",   required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=10000)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--mode", choices=["dp", "ddp"], default="ddp")
    # NEW LoRA flags
    p.add_argument("--lora", action="store_true", help="enable LoRA fine-tuning")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", type=str, default="q_proj,v_proj",
                   help="comma-separated module names to LoRA-adapt")
    args = p.parse_args()

    ws = torch.cuda.device_count() if args.mode == "ddp" else 1
    if args.mode == "dp":
        run(rank=0, world_size=1, args=args)
    else:
        mp.spawn(run, args=(ws, args), nprocs=ws)

if __name__ == "__main__":
    main()


