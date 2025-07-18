#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:4
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-01:00:00
#SBATCH -J "DEBUG_MULTI_NODE"
#SBATCH --error=err_%J.err
#SBATCH --output=out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

# =============================================================================
# MULTI-NODE DEBUGGING SCRIPT FOR SLURM
# =============================================================================

set -e  # Exit on any error

echo "=== SLURM DEBUG SESSION STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Started at: $(date)"
echo "=========================================="

# =============================================================================
# 1. SLURM ENVIRONMENT INFORMATION
# =============================================================================

echo ""
echo "=== SLURM ENVIRONMENT ==="
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Tasks per node: $SLURM_TASKS_PER_NODE"
echo "Current node: $SLURMD_NODENAME"
echo "Node rank: $SLURM_PROCID"
echo "Local rank: $SLURM_LOCALID"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"

# =============================================================================
# 2. NETWORK CONNECTIVITY CHECK
# =============================================================================

echo ""
echo "=== NETWORK CONNECTIVITY CHECK ==="
echo "Checking inter-node connectivity..."

# Get all nodes in the job
nodes=$(scontrol show job $SLURM_JOB_ID | grep -o 'NodeList=[^[:space:]]*' | cut -d'=' -f2)
echo "All nodes in job: $nodes"

# Test ping between nodes
for node in $(scontrol show hostnames $nodes); do
    if [ "$node" != "$SLURMD_NODENAME" ]; then
        echo "Pinging $node from $SLURMD_NODENAME..."
        if ping -c 3 $node > /dev/null 2>&1; then
            echo "  ✓ $node is reachable"
        else
            echo "  ✗ $node is NOT reachable"
        fi
    fi
done

# =============================================================================
# 3. SYSTEM INFORMATION
# =============================================================================

echo ""
echo "=== SYSTEM INFORMATION ==="
echo "Hostname: $(hostname)"
echo "OS: $(uname -a)"
echo "CPU info: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "Total memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"

# =============================================================================
# 4. MOVE TO PROJECT DIRECTORY
# =============================================================================

echo ""
echo "=== SETTING UP ENVIRONMENT ==="
echo "Moving to project directory..."
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick

if [ $? -eq 0 ]; then
    echo "✓ Successfully changed to project directory"
    echo "Current directory: $(pwd)"
else
    echo "✗ Failed to change to project directory"
    exit 1
fi

# =============================================================================
# 5. MODULE LOADING
# =============================================================================

echo ""
echo "=== LOADING MODULES ==="
module purge
echo "✓ Modules purged"

modules=(
    "Python/3.11.3-GCCcore-12.3.0"
    "scikit-image/0.22.0"
    "scikit-learn/1.3.1"
    "h5py/3.9.0-foss-2023a"
    "CUDA/12.1.1"
)

for mod in "${modules[@]}"; do
    echo "Loading module: $mod"
    if module load $mod; then
        echo "  ✓ $mod loaded successfully"
    else
        echo "  ✗ Failed to load $mod"
        exit 1
    fi
done

echo "Loaded modules:"
module list

# =============================================================================
# 6. VIRTUAL ENVIRONMENT ACTIVATION
# =============================================================================

echo ""
echo "=== ACTIVATING VIRTUAL ENVIRONMENT ==="
if [ -f "Deep_Sick_env/bin/activate" ]; then
    source Deep_Sick_env/bin/activate
    echo "✓ Virtual environment activated"
    echo "Python path: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "✗ Virtual environment not found at Deep_Sick_env/bin/activate"
    exit 1
fi

# =============================================================================
# 7. CUDA AND GPU VERIFICATION
# =============================================================================

echo ""
echo "=== CUDA AND GPU VERIFICATION ==="
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check CUDA installation
if command -v nvcc &> /dev/null; then
    echo "✓ NVCC found: $(nvcc --version | grep release)"
else
    echo "✗ NVCC not found"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available"
    echo "GPU information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
else
    echo "✗ nvidia-smi not available"
fi

# Test PyTorch CUDA availability
echo ""
echo "=== PYTORCH CUDA TEST ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
else:
    print('CUDA not available in PyTorch')
"

# =============================================================================
# 8. DISTRIBUTED TRAINING SETUP CHECK
# =============================================================================

echo ""
echo "=== DISTRIBUTED TRAINING SETUP ==="

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Alternative calculation if SLURM_GPUS_PER_NODE is not set
if [ -z "$WORLD_SIZE" ] || [ "$WORLD_SIZE" -eq 0 ]; then
    export WORLD_SIZE=$SLURM_NTASKS
fi

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Test distributed PyTorch setup
echo ""
echo "=== DISTRIBUTED PYTORCH TEST ==="
python -c "
import torch
import torch.distributed as dist
import os

print('Testing distributed PyTorch setup...')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
print(f'MPI available: {torch.distributed.is_mpi_available()}')

# Test tensor creation on GPU
if torch.cuda.is_available():
    device = torch.device(f'cuda:{int(os.environ.get(\"LOCAL_RANK\", 0))}')
    print(f'Using device: {device}')
    try:
        tensor = torch.randn(100, 100).to(device)
        print(f'✓ Successfully created tensor on {device}')
        print(f'  Tensor shape: {tensor.shape}')
        print(f'  Tensor device: {tensor.device}')
    except Exception as e:
        print(f'✗ Error creating tensor on GPU: {e}')
else:
    print('✗ CUDA not available for tensor operations')
"

# =============================================================================
# 9. WANDB SETUP
# =============================================================================

echo ""
echo "=== WANDB SETUP ==="
export WANDB_MODE=online
echo "WANDB_MODE: $WANDB_MODE"

# Test wandb import
python -c "
try:
    import wandb
    print('✓ wandb imported successfully')
    print(f'wandb version: {wandb.__version__}')
except ImportError as e:
    print(f'✗ wandb import failed: {e}')
"

# =============================================================================
# 10. PYTHON PATH SETUP
# =============================================================================

echo ""
echo "=== PYTHON PATH SETUP ==="
# Note: workspaceFolder variable was incomplete in original script
export PYTHONPATH="${PWD}:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# =============================================================================
# 11. INTER-NODE COMMUNICATION TEST
# =============================================================================

echo ""
echo "=== INTER-NODE COMMUNICATION TEST ==="

# Create a simple distributed test script in the current directory (accessible to all nodes)
cat > ./dist_test_${SLURM_JOB_ID}.py << 'EOF'
import torch
import torch.distributed as dist
import os

def test_distributed():
    # Initialize process group
    try:
        # Check required environment variables
        required_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']
        for var in required_vars:
            if var not in os.environ:
                print(f"✗ Missing environment variable: {var}")
                return
            else:
                print(f"  {var}: {os.environ[var]}")

        print(f"✓ All required environment variables are set")

        # Initialize process group
        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"✓ Process group initialized successfully")
        print(f"  Rank: {dist.get_rank()}")
        print(f"  World size: {dist.get_world_size()}")

        # Test all-reduce operation
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
            tensor = torch.ones(1).to(device) * dist.get_rank()
            print(f"  Before all-reduce (rank {dist.get_rank()}): {tensor.item()}")

            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            print(f"  After all-reduce (rank {dist.get_rank()}): {tensor.item()}")

            expected = sum(range(dist.get_world_size()))
            if abs(tensor.item() - expected) < 1e-6:
                print("  ✓ All-reduce test passed")
            else:
                print(f"  ✗ All-reduce test failed (expected {expected})")
        else:
            print("  ⚠ CUDA not available, skipping GPU all-reduce test")

        dist.destroy_process_group()
        print("✓ Process group destroyed successfully")

    except Exception as e:
        print(f"✗ Distributed test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_distributed()
EOF

# Run distributed test on all nodes with proper SLURM task distribution
echo "Running distributed communication test..."
echo "Note: This test will run on each SLURM task across all nodes"

# Use srun with proper task distribution
srun --ntasks=$WORLD_SIZE --ntasks-per-node=$SLURM_GPUS_PER_NODE python ./dist_test_${SLURM_JOB_ID}.py

# =============================================================================
# 12. STORAGE AND PERMISSIONS CHECK
# =============================================================================

echo ""
echo "=== STORAGE AND PERMISSIONS CHECK ==="
echo "Current directory permissions:"
ls -la

echo ""
echo "Disk usage:"
df -h .

echo ""
echo "Available space in current directory:"
du -sh .

# Test write permissions
test_file="test_write_${SLURM_JOB_ID}.tmp"
if touch $test_file 2>/dev/null; then
    echo "✓ Write permissions OK"
    rm -f $test_file
else
    echo "✗ No write permissions in current directory"
fi

# =============================================================================
# 13. SUMMARY AND NEXT STEPS
# =============================================================================

echo ""
echo "=== DEBUG SUMMARY ==="
echo "Completed at: $(date)"
echo ""
echo "Next steps for debugging:"
echo "1. Check the output above for any ✗ (failed) tests"
echo "2. Verify that all GPUs are visible and accessible"
echo "3. Ensure inter-node communication is working"
echo "4. Test your actual training script with a small dataset"
echo ""
echo "Common issues to check:"
echo "- Network connectivity between nodes"
echo "- CUDA/GPU driver compatibility"
echo "- Python package versions"
echo "- Distributed training environment variables"
echo "- Storage permissions and available space"
echo ""
echo "=== DEBUG SESSION COMPLETED ==="

# Clean up temporary files
rm -f ./dist_test_${SLURM_JOB_ID}.py

# Keep the job alive for manual inspection if needed
echo ""
echo "Job will end shortly. Check the output file for detailed results."
echo "To run your actual training script, replace this debug section with your training code."