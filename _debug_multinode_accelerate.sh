#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 2                         # two nodes
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-node=A100:4       # 4 GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -J focused_accelerate_debug
#SBATCH --output=focused_debug_%J.out
#SBATCH --error=focused_debug_%J.err

set -euo pipefail

echo "=== Focused Accelerate Debug (Working Methods Only) ==="
echo "Allocated nodes: $SLURM_JOB_NODELIST"

# Clean up any hanging processes first
echo "🧹 Cleaning up any hanging processes..."
srun --nodes=$SLURM_NNODES pkill -f "python.*accelerate" || true
srun --nodes=$SLURM_NNODES pkill -f "torch.*distributed" || true
sleep 2

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Use different ports for each method to avoid conflicts
MASTER_PORT_METHOD2=$((29500 + (SLURM_JOB_ID % 1000)))
MASTER_PORT_METHOD3=$((29600 + (SLURM_JOB_ID % 1000)))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT_METHOD2=$MASTER_PORT_METHOD2"
echo "MASTER_PORT_METHOD3=$MASTER_PORT_METHOD3"
echo "WORLD_SIZE=$SLURM_NTASKS"

# Move to project directory
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick

# Load environment
source activateEnv.sh
echo "✓ Environment activated"

# NCCL settings
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN  # Reduce noise

# Save the debug script
cat > accelerate_debug_test.py << 'EOF'
import os
import torch
from accelerate import Accelerator
import torch.distributed as dist


def debug_environment():
    """Debug the accelerate environment setup"""
    print(f"\n=== Debug Info from Process ===")
    print(f"Hostname: {os.uname()[1]}")

    # Environment variables
    env_vars = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS", "SLURM_NODEID",
        "CUDA_VISIBLE_DEVICES"
    ]

    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"{var}: {value}")

    # CUDA info
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")


def test_accelerate():
    """Test accelerate initialization and basic operations"""
    try:
        print("\n=== Testing Accelerate Initialization ===")

        # Initialize accelerator with minimal config
        accelerator = Accelerator(mixed_precision='no')  # Simplify for debug

        print(f"✅ Accelerator initialized successfully!")
        print(f"Process index: {accelerator.process_index}")
        print(f"Local process index: {accelerator.local_process_index}")
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Is main process: {accelerator.is_main_process}")
        print(f"Is local main process: {accelerator.is_local_main_process}")

        # Test tensor operations
        print("\n=== Testing Tensor Operations ===")
        x = torch.tensor([accelerator.process_index], device=accelerator.device, dtype=torch.float)
        print(f"Created tensor on {accelerator.device}: {x}")

        # Test all_reduce
        print("\n=== Testing All-Reduce ===")
        original_value = x.clone()
        accelerator.wait_for_everyone()

        # Use torch.distributed directly for all_reduce
        if dist.is_initialized():
            dist.all_reduce(x)
            print(f"All-reduce successful! Original: {original_value.item()}, Result: {x.item()}")
        else:
            print("❌ Distributed not initialized")

        # Test model preparation (dummy model)
        print("\n=== Testing Model Preparation ===")
        model = torch.nn.Linear(10, 1)
        model = accelerator.prepare(model)
        print(f"✅ Model prepared successfully on {next(model.parameters()).device}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("\n🎉 All accelerate tests passed!")

    except Exception as e:
        print(f"❌ Accelerate test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_environment()
    test_accelerate()
EOF

######################
### Method 2: Direct SLURM (Communication Test Style) - SHOULD WORK
######################
echo -e "\n=== Method 2: Testing with direct srun (communication test style) ==="

srun bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT_METHOD2"'

  echo "[Rank $RANK] Starting debug on $(hostname) with LOCAL_RANK=$LOCAL_RANK"

  python accelerate_debug_test.py
'

exit_code_method2=$?
echo "Method 2 completed with exit code: $exit_code_method2"

# Clean up between methods
echo "🧹 Cleaning up between methods..."
srun --nodes=$SLURM_NNODES pkill -f "python.*accelerate" || true
sleep 3

######################
### Method 3: TorchRun - SHOULD WORK
######################
echo -e "\n=== Method 3: Testing with torchrun ==="

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT_METHOD3 \
    --node_rank=$SLURM_NODEID \
    accelerate_debug_test.py

exit_code_method3=$?
echo "Method 3 completed with exit code: $exit_code_method3"

######################
### Method 4: Test simple distributed without accelerate
######################
echo -e "\n=== Method 4: Testing pure PyTorch distributed (no accelerate) ==="

cat > pure_pytorch_test.py << 'EOF'
import os
import torch
import torch.distributed as dist

def test_pure_pytorch():
    print(f"\n=== Pure PyTorch Distributed Test ===")
    print(f"Hostname: {os.uname()[1]}")

    # Environment variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    print(f"RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    print(f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

    # Set CUDA device
    torch.cuda.set_device(local_rank)
    print(f"Set CUDA device to: {local_rank}")

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    print(f"✅ Process group initialized!")

    # Test tensor operation
    x = torch.tensor([rank], device=f"cuda:{local_rank}")
    print(f"Created tensor: {x} on device: {x.device}")

    # Test all_reduce
    dist.all_reduce(x)
    print(f"✅ All-reduce successful! Result: {x.item()}")

    dist.destroy_process_group()
    print(f"✅ Process group destroyed successfully!")

if __name__ == "__main__":
    test_pure_pytorch()
EOF

MASTER_PORT_METHOD4=$((29700 + (SLURM_JOB_ID % 1000)))

srun bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT_METHOD4"'

  python pure_pytorch_test.py
'

exit_code_method4=$?
echo "Method 4 completed with exit code: $exit_code_method4"

######################
### Summary
######################
echo -e "\n=== Summary ==="
echo "Method 1 (accelerate launch): SKIPPED (known to fail with port conflicts)"
echo "Method 2 (direct srun): exit code $exit_code_method2"
echo "Method 3 (torchrun): exit code $exit_code_method3"
echo "Method 4 (pure pytorch): exit code $exit_code_method4"

# Cleanup
rm -f accelerate_debug_test.py pure_pytorch_test.py

# Return success if any working method succeeded
if [ $exit_code_method2 -eq 0 ] || [ $exit_code_method3 -eq 0 ] || [ $exit_code_method4 -eq 0 ]; then
    echo "✅ At least one working method succeeded!"
    exit 0
else
    echo "❌ All working methods failed"
    exit 1
fi
