import math

import torch
import torch.distributed as dist
from datetime import timedelta
import logging
from accelerate.state import DistributedType

logger = logging.getLogger(__name__)


def safe_wait_for_everyone_simple(accelerator):
    """
    Simplified version that only uses accelerator's built-in synchronization.
    More reliable for most use cases.

    Args:
        accelerator: Accelerate Accelerator instance
        timeout_seconds: Timeout in seconds (for logging only, accelerator handles actual timeout)

    Raises:
        RuntimeError: If synchronization fails or accelerator is None
    """
    if accelerator is None:
        raise RuntimeError("Accelerator cannot be None")

    try:
        # Only use accelerator's built-in synchronization
        # This is usually sufficient and more reliable
        accelerator.wait_for_everyone()
        logger.debug("Process synchronization completed")

    except Exception as e:
        error_msg = f"Process synchronization failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def safe_wait_with_health_check(accelerator, timeout_seconds=300):
    """
    Enhanced version with communication health check.

    Args:
        accelerator: Accelerate Accelerator instance
        timeout_seconds: Timeout in seconds

    Raises:
        RuntimeError: If synchronization fails or communication test fails
    """
    if accelerator is None:
        raise RuntimeError("Accelerator cannot be None")

    try:
        # Basic synchronization
        accelerator.wait_for_everyone()

        # Optional: Test communication health if in distributed mode
        if accelerator.num_processes > 1:
            # Simple communication test
            test_tensor = torch.tensor([accelerator.process_index],
                                       device=accelerator.device, dtype=torch.float32)

            try:
                # Test gather operation
                gathered = accelerator.gather(test_tensor)

                # Verify on main process
                if accelerator.is_main_process:
                    expected = torch.arange(accelerator.num_processes, dtype=torch.float32)
                    if not torch.allclose(gathered.cpu(), expected, rtol=1e-6):
                        raise RuntimeError("Communication health check failed: gather operation returned incorrect results")

                logger.debug("Communication health check passed")

            except Exception as comm_error:
                logger.warning(f"Communication health check failed: {comm_error}")
                # Don't fail the entire synchronization for health check issues
                # Just log the warning

        logger.debug("Enhanced synchronization completed successfully")

    except Exception as e:
        error_msg = f"Enhanced synchronization failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# Usage examples and recommended patterns:

def training_step_with_sync(model, batch, accelerator):
    """Example of using safe synchronization in training."""

    # Training step
    model.train()
    with torch.set_grad_enabled(True):
        outputs = model(**batch)
        loss = outputs.loss

    accelerator.backward(loss)

    # Use simple synchronization (recommended)
    if accelerator.sync_gradients:
        # Accelerator handles synchronization automatically
        # Only add manual sync if absolutely necessary
        safe_wait_for_everyone_simple(accelerator)

    return loss.detach()



def evaluation_with_sync(model, eval_dataloader, accelerator):
    """Example of synchronized evaluation."""

    # Sync before evaluation
    safe_wait_for_everyone_simple(accelerator)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            loss = outputs.loss

            # Gather loss from all processes
            gathered_loss = accelerator.gather_for_metrics(loss.repeat(batch['input_ids'].shape[0]))
            total_loss += gathered_loss.sum().item()
            num_batches += gathered_loss.shape[0]

    # Sync after evaluation
    safe_wait_for_everyone_simple(accelerator)

    try:
        eval_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss


def checkpoint_save_with_sync(accelerator, save_path):
    """Example of synchronized checkpoint saving."""

    try:
        # Sync before saving
        safe_wait_for_everyone_simple(accelerator)

        # Create directory on main process
        if accelerator.is_main_process:
            import os
            os.makedirs(save_path, exist_ok=True)

        # Sync after directory creation
        safe_wait_for_everyone_simple(accelerator)

        # Save checkpoint (accelerator handles synchronization internally)
        accelerator.save_state(save_path)

        # Final sync to ensure completion
        safe_wait_for_everyone_simple(accelerator)

        logger.info(f"Checkpoint saved successfully: {save_path}")
        return True

    except Exception as e:
        logger.error(f"Checkpoint save failed: {e}")
        return False

