import os
from datetime import timedelta
import torch.distributed as dist


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
