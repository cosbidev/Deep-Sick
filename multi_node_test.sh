#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=A100:4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -J diagnostic_debug
#SBATCH --output=diagnostic_%J.out
#SBATCH --error=diagnostic_%J.err

set -euo pipefail

echo "=== Diagnostic Debug ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"

cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick
source activateEnv.sh

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((30000 + RANDOM % 10000))

echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

# Create comprehensive diagnostic script
cat > full_diagnostic.py << 'EOF'
import os
import sys
import torch
import subprocess

def run_diagnostics():
    print(f"\n=== SYSTEM DIAGNOSTICS ===")
    print(f"Hostname: {os.uname()[1]}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # Check environment
    print(f"\n=== ENVIRONMENT ===")
    env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "CUDA_VISIBLE_DEVICES", "SLURM_PROCID", "SLURM_LOCALID"]
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'NOT_SET')}")

    # Check CUDA
    print(f"\n=== CUDA DIAGNOSTICS ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory = torch.cuda.memory_allocated(i) / 1024**3
            print(f"GPU {i}: {props.name}, Memory used: {memory:.2f}GB")

    # Check processes
    print(f"\n=== PROCESS CHECK ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("NVIDIA-SMI output:")
        print(result.stdout)
    except:
        print("nvidia-smi not available")

    # Check ports
    print(f"\n=== PORT CHECK ===")
    master_port = os.environ.get("MASTER_PORT")
    if master_port:
        try:
            result = subprocess.run(['netstat', '-tulpn'], capture_output=True, text=True)
            lines = [line for line in result.stdout.split('\n') if master_port in line]
            if lines:
                print(f"Port {master_port} usage:")
                for line in lines:
                    print(line)
            else:
                print(f"Port {master_port} appears free")
        except:
            print("netstat check failed")

def test_basic_distributed():
    print(f"\n=== BASIC DISTRIBUTED TEST ===")

    try:
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        print(f"Setting CUDA device to {local_rank}")
        torch.cuda.set_device(local_rank)

        print(f"Initializing process group...")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.default_pg_timeout
        )

        print(f"✅ Process group initialized!")

        # Test communication
        x = torch.tensor([rank], device=f"cuda:{local_rank}")
        print(f"Created tensor: {x}")

        dist.all_reduce(x)
        print(f"✅ All-reduce successful! Result: {x.item()}")

        # Clean shutdown
        dist.destroy_process_group()
        print(f"✅ Process group destroyed!")

        return True

    except Exception as e:
        print(f"❌ Distributed test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accelerate():
    print(f"\n=== ACCELERATE TEST ===")

    try:
        from accelerate import Accelerator

        print("Creating accelerator...")
        accelerator = Accelerator()

        print(f"✅ Accelerator created!")
        print(f"Device: {accelerator.device}")
        print(f"Process index: {accelerator.process_index}")
        print(f"Num processes: {accelerator.num_processes}")

        # Test model preparation
        model = torch.nn.Linear(10, 1)
        model = accelerator.prepare(model)
        print(f"✅ Model prepared on {next(model.parameters()).device}")

        return True

    except Exception as e:
        print(f"❌ Accelerate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_diagnostics()

    # Test basic distributed first
    distributed_works = test_basic_distributed()

    # Only test accelerate if basic distributed works
    if distributed_works:
        accelerate_works = test_accelerate()
    else:
        print("Skipping accelerate test due to distributed failure")
        accelerate_works = False

    if distributed_works and accelerate_works:
        print(f"\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n❌ SOME TESTS FAILED")
        sys.exit(1)
EOF

# Run the diagnostic
srun bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'

  echo "[Rank $RANK] Starting diagnostics on $(hostname)"
  python full_diagnostic.py
'

exit_code=$?
echo "Diagnostics completed with exit code: $exit_code"

# Cleanup
rm -f full_diagnostic.py

exit $exit_coden
