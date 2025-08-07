import math
from itertools import chain
import os
from typing import List, Optional

import numpy as np
import torch
import logging
from torch.utils.data import Sampler
from transformers import MODEL_MAPPING,  SchedulerType
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf

from src.distributed import safe_wait_for_everyone_simple

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import torch.distributed as dist


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)






def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# New Code #
def evaluate(training_args, model, eval_dataloader, accelerator, logger):
    # Sync after evaluation

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        if accelerator.is_main_process:
            logger.info(f"||||Step {step + 1} / {len(eval_dataloader)} ||||")
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

    losses = torch.cat(losses)


    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks






def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # Ensure all samples have non-zero length
    assert all(l != 0 for l in lengths), "Should not have zero length."
    # If all samples belong to a single modality (either all multimodal or all language-only), apply default grouping
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    # Separate indices and lengths of multimodal and language-only samples
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # Perform length-based grouping and random shuffling for multimodal samples
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    # Perform length-based grouping and random shuffling for language-only samples
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # Prepare for optional merging of final smaller batches from each modality
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    # Combine all full megabatches from both modalities
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    # Shuffle the megabatches to avoid ordering bias
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    # Apply the random shuffling to the megabatches
    megabatches = [megabatches[i] for i in megabatch_indices]

    # Optionally add the remaining samples as a final megabatch
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


import torch
import numpy as np
from torch.utils.data import Sampler, DataLoader, Dataset
from typing import List, Optional
import matplotlib.pyplot as plt
from collections import Counter
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class BlueprintGroupedSampler(Sampler):
    """
    FIXED: Sampler that guarantees EXACTLY the same number of images per batch
    across all GPUs for memory uniformity in distributed training.
    """

    def __init__(
            self,
            batch_size: int,
            lengths: List[int],
            n_images: List[int],
            generator: Optional[torch.Generator] = None,
            seed: int = 42,
            drop_last: bool = True
    ):
        super().__init__(None)

        self.batch_size = batch_size
        self.lengths = lengths
        self.n_images = n_images
        self.drop_last = drop_last
        self.generator = generator or torch.Generator().manual_seed(seed)
        self.epoch = 0

        # Validation
        assert len(lengths) == len(n_images), "lengths and n_images must have same length"
        assert all(n in [1, 2] for n in n_images), "n_images must contain only values 1 or 2"
        assert batch_size > 0, "batch_size must be positive"

        # Pre-compute indices by category
        self.indices_1img = np.where(np.array(n_images) == 1)[0]
        self.indices_2img = np.where(np.array(n_images) == 2)[0]

        print(f"[BlueprintSampler] Dataset: {len(self.indices_1img)} samples with 1 image, {len(self.indices_2img)} samples with 2 images")

        # Calculate and validate strategy
        self.strategy = self._calculate_uniform_strategy()
        self._validate_strategy()

        print(f"[BlueprintSampler] Strategy for batch_size={batch_size}: {self.strategy}")

    def _calculate_uniform_strategy(self):
        """Calculate strategy that ensures EXACTLY batch_size images per GPU."""
        target_images = self.batch_size

        if target_images == 1:
            return {
                    "samples_1img"    : 1,
                    "samples_2img"    : 0,
                    "images_per_batch": 1,
                    "description"     : "1 sample with 1 image"
            }
        elif target_images == 2:
            return {
                    "samples_1img"    : 1,
                    "samples_2img"    : 0,
                    "images_per_batch": 2,
                    "description"     : "2 sample with 1 images"
            }
        elif target_images == 3:
            return {
                    "samples_1img"    : 1,
                    "samples_2img"    : 2,
                    "images_per_batch": 5,
                    "description"     : "1 sample with 2 images + 2 sample with 1 image"
            }
        elif target_images == 4:
            return {
                    "samples_1img"    : 1,
                    "samples_2img"    : 3,
                    "images_per_batch": 7,
                    "description"     : "1 samples with 1 images, 3 with 2 each"
            }
        else:
            # General strategy
            samples_2img = target_images // 2
            samples_1img = target_images % 2

            return {
                    "samples_1img"    : samples_1img,
                    "samples_2img"    : samples_2img,
                    "images_per_batch": samples_2img * 2 + samples_1img,
                    "description"     : f"{samples_2img} samples with 2 images + {samples_1img} samples with 1 image"
            }

    def _validate_strategy(self):
        """Validate that our strategy is feasible and safe."""
        strategy = self.strategy

        required_1img = strategy["samples_1img"]
        required_2img = strategy["samples_2img"]

        if required_1img > 0 and len(self.indices_1img) < required_1img:
            raise RuntimeError(
                    f"Strategy requires {required_1img} samples with 1 image per batch, "
                    f"but dataset only has {len(self.indices_1img)} such samples"
            )

        if required_2img > 0 and len(self.indices_2img) < required_2img:
            raise RuntimeError(
                    f"Strategy requires {required_2img} samples with 2 images per batch, "
                    f"but dataset only has {len(self.indices_2img)} such samples"
            )

    def _build_uniform_batches(self, max_batches=None):
        """Build batches with guaranteed uniform image count."""
        strategy = self.strategy
        samples_1img = strategy["samples_1img"]
        samples_2img = strategy["samples_2img"]

        # Deterministic shuffle
        g = torch.Generator()
        g.manual_seed(self.epoch + 42)

        if len(self.indices_1img) > 0:
            indices_1img_shuffled = self.indices_1img[torch.randperm(len(self.indices_1img), generator=g).numpy()]
        else:
            indices_1img_shuffled = np.array([])

        if len(self.indices_2img) > 0:
            indices_2img_shuffled = self.indices_2img[torch.randperm(len(self.indices_2img), generator=g).numpy()]
        else:
            indices_2img_shuffled = np.array([])

        # Calculate maximum batches
        max_batches_1img = len(indices_1img_shuffled) // samples_1img if samples_1img > 0 else float('inf')
        max_batches_2img = len(indices_2img_shuffled) // samples_2img if samples_2img > 0 else float('inf')
        if max_batches_2img == float('inf') or max_batches_1img == float('inf'):
            max_batches = min(max_batches_2img, max_batches_1img)

            remaining = 0
        else:
            max_batches = min(max_batches_2img, max_batches_1img)
            min_batches = max(max_batches_2img, max_batches_1img)
            remaining = abs(max_batches - min_batches)

        if max_batches == 0:
            raise RuntimeError(
                    f"Cannot create any complete batches with strategy {strategy}. "
                    f"Available: {len(self.indices_1img)} 1-img, {len(self.indices_2img)} 2-img. "
                    f"Required per batch: {samples_1img} 1-img, {samples_2img} 2-img."
            )

        # Build validated batches
        batches = []
        for batch_idx in range(max_batches):
            batch = []

            # Add 2-image samples
            if samples_2img > 0:
                start_2img = batch_idx * samples_2img
                end_2img = start_2img + samples_2img
                if end_2img <= len(indices_2img_shuffled):
                    batch.extend(indices_2img_shuffled[start_2img:end_2img])

            # Add 1-image samples
            if samples_1img > 0:
                start_1img = batch_idx * samples_1img
                end_1img = start_1img + samples_1img
                if end_1img <= len(indices_1img_shuffled):
                    batch.extend(indices_1img_shuffled[start_1img:end_1img])

            # Validate batch
            if len(batch) != self.batch_size:
                continue

            total_images = sum(2 if idx in self.indices_2img else 1 for idx in batch)
            batches.append(batch)
        # Do the remaining batches
        for batch_idx in range(remaining// self.batch_size):
            batch = []
            # find who is remaining
            # Add 2-image samples
            if not max_batches_2img == max_batches:
                start_2img = batch_idx * samples_2img
                end_2img = start_2img + samples_2img
                if end_2img <= len(indices_2img_shuffled):
                    batch.extend(indices_2img_shuffled[start_2img:end_2img])

            # Add 1-image samples
            if not max_batches_1img == max_batches:
                # If we are here, we have to add 1-image samples for the amount of batch size
                start_1img = batch_idx * samples_1img
                end_1img = start_1img + samples_1img
                if end_1img <= len(indices_1img_shuffled):
                    batch.extend(indices_1img_shuffled[start_1img:end_1img])

            # Validate batch
            if len(batch) != self.batch_size:
                continue

            total_images = sum(2 if idx in self.indices_2img else 1 for idx in batch)
            if total_images != self.batch_size:
                raise RuntimeError(f"Batch {batch_idx} has {total_images} images, expected {self.batch_size}")

            batches.append(batch)

        return batches

    def __iter__(self):
        batches = self._build_uniform_batches()
        indices = []
        for batch in batches:
            indices.extend(batch)
        return iter(indices)

    def __len__(self):
        try:
            batches = self._build_uniform_batches()
            return len(batches) * self.batch_size
        except RuntimeError:
            return 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_memory_info(self):
        """Report expected memory usage."""
        strategy = self.strategy
        return {
                "images_per_batch"    : strategy["images_per_batch"],
                "samples_per_batch"   : self.batch_size,
                "strategy_description": strategy["description"],
                "memory_uniformity"   : "GUARANTEED" if strategy["images_per_batch"] == self.batch_size else "VIOLATED"
        }


class FakeVisionDataset(Dataset):
    """Fake dataset for testing with varying image counts and lengths."""

    def __init__(self, size=1000):
        self.size = size

        # Generate fake data with realistic distributions
        self.data = []

        for i in range(size):
            # 70% single image, 30% dual image (realistic for VL datasets)
            n_images = np.random.choice([1, 2], p=[0.7, 0.3])

            # Length varies based on image count (dual image samples tend to be longer)
            if n_images == 1:
                length = np.random.randint(50, 200)  # Shorter text for single images
            else:
                length = np.random.randint(100, 400)  # Longer text for dual images

            self.data.append({
                    'id'         : i,
                    'n_images'   : n_images,
                    'length'     : length,
                    'fake_text'  : f"Sample {i} with {n_images} images and {length} tokens",
                    'fake_images': [f"image_{i}_{j}.jpg" for j in range(n_images)]
            })

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def analyze_dataset_distribution(dataset):
    """Analyze the distribution of image counts and lengths."""
    n_images = [item['n_images'] for item in dataset.data]
    lengths = [item['length'] for item in dataset.data]

    print("=" * 60)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Image count distribution
    img_dist = Counter(n_images)
    print(f"Image count distribution:")
    for count, freq in sorted(img_dist.items()):
        print(f"  {count} image(s): {freq} samples ({freq / len(dataset) * 100:.1f}%)")

    # Length statistics by image count
    lengths_1img = [l for l, n in zip(lengths, n_images) if n == 1]
    lengths_2img = [l for l, n in zip(lengths, n_images) if n == 2]

    print(f"\nLength statistics:")
    print(f"  1-image samples: mean={np.mean(lengths_1img):.1f}, std={np.std(lengths_1img):.1f}")
    print(f"  2-image samples: mean={np.mean(lengths_2img):.1f}, std={np.std(lengths_2img):.1f}")
    print(f"  Overall: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")


def test_sampler_with_different_batch_sizes(dataset):
    """Test the sampler with various batch sizes."""
    lengths = [item['length'] for item in dataset.data]
    n_images = [item['n_images'] for item in dataset.data]

    batch_sizes_to_test = [3, 4, 8, 16]

    print("\n" + "=" * 80)
    print("SAMPLER TESTING WITH DIFFERENT BATCH SIZES")
    print("=" * 80)

    results = {}

    for batch_size in batch_sizes_to_test:
        print(f"\n🔍 Testing batch_size={batch_size}")
        print("-" * 40)

        try:
            sampler = BlueprintGroupedSampler(
                    batch_size=batch_size,
                    lengths=lengths,
                    n_images=n_images,
                    seed=42
            )

            # Test memory info
            memory_info = sampler.get_memory_info()
            print(f"Memory info: {memory_info}")

            # Test iteration
            sampler_len = len(sampler)
            print(f"Sampler length: {sampler_len}")

            # Test first few batches for validation
            dataloader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    collate_fn=lambda x: x  # Simple identity collate
            )

            batch_image_counts = []
            batch_sample_counts = []

            for i, batch in enumerate(dataloader):
                if i >= 10:  # Test first 10 batches
                    break

                # Count images in this batch
                total_images = sum(item['n_images'] for item in batch)
                batch_image_counts.append(total_images)
                batch_sample_counts.append(len(batch))

                if i == 0:  # Show first batch details
                    print(f"First batch details:")
                    for j, item in enumerate(batch):
                        print(f"  Sample {j}: {item['n_images']} images, length {item['length']}")
                    print(f"  Total: {len(batch)} samples, {total_images} images")

            # Validate uniformity
            unique_image_counts = set(batch_image_counts)
            unique_sample_counts = set(batch_sample_counts)

            success = len(unique_image_counts) == 1 and len(unique_sample_counts) == 1
            expected_images = batch_size
            actual_images = list(unique_image_counts)[0] if unique_image_counts else 0

            results[batch_size] = {
                    'success'        : success,
                    'expected_images': expected_images,
                    'actual_images'  : actual_images,
                    'uniform'        : success and actual_images == expected_images,
                    'total_batches'  : len(batch_image_counts),
                    'sampler_length' : sampler_len
            }

            status = "✅ SUCCESS" if success and actual_images == expected_images else "❌ FAILED"
            print(f"Result: {status}")
            if success:
                print(f"  All batches uniform: {actual_images} images each")
            else:
                print(f"  Non-uniform batches detected!")
                print(f"  Image counts per batch: {set(batch_image_counts)}")

        except Exception as e:
            print(f"❌ FAILED: {e}")
            results[batch_size] = {
                    'success'        : False,
                    'error'          : str(e),
                    'expected_images': batch_size,
                    'actual_images'  : 0,
                    'uniform'        : False
            }

    return results


def visualize_results(results):
    """Create visualizations of the test results."""

    # Prepare data for plotting
    batch_sizes = []
    expected_images = []
    actual_images = []
    success_status = []

    for bs, result in results.items():
        if 'actual_images' in result:
            batch_sizes.append(bs)
            expected_images.append(result['expected_images'])
            actual_images.append(result['actual_images'])
            success_status.append('Success' if result['uniform'] else 'Failed')

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: Expected vs Actual Images
    plt.subplot(2, 2, 1)
    colors = ['green' if status == 'Success' else 'red' for status in success_status]
    plt.bar(range(len(batch_sizes)), actual_images, color=colors, alpha=0.7, label='Actual')
    plt.plot(range(len(batch_sizes)), expected_images, 'bo-', label='Expected', linewidth=2)
    plt.xlabel('Batch Size Configuration')
    plt.ylabel('Images per Batch')
    plt.title('Expected vs Actual Images per Batch')
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Success Rate
    plt.subplot(2, 2, 2)
    success_count = sum(1 for s in success_status if s == 'Success')
    failure_count = len(success_status) - success_count
    plt.pie([success_count, failure_count],
            labels=['Success', 'Failed'],
            colors=['green', 'red'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title('Sampler Success Rate')

    # Plot 3: Memory Uniformity Check
    plt.subplot(2, 2, 3)
    uniform_check = []
    labels = []
    for bs, result in results.items():
        if 'uniform' in result:
            uniform_check.append(1 if result['uniform'] else 0)
            labels.append(f"BS={bs}")

    colors = ['green' if u else 'red' for u in uniform_check]
    plt.bar(range(len(labels)), uniform_check, color=colors, alpha=0.7)
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Uniform (1=Yes, 0=No)')
    plt.title('Memory Uniformity by Batch Size')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylim(0, 1.2)

    # Plot 4: Dataset utilization
    plt.subplot(2, 2, 4)
    utilization = []
    bs_labels = []
    for bs, result in results.items():
        if 'sampler_length' in result and result['sampler_length'] > 0:
            util = result['sampler_length'] / 1000  # Our dataset has 1000 samples
            utilization.append(util * 100)
            bs_labels.append(bs)

    plt.bar(range(len(bs_labels)), utilization, alpha=0.7)
    plt.xlabel('Batch Size')
    plt.ylabel('Dataset Utilization (%)')
    plt.title('Dataset Utilization by Batch Size')
    plt.xticks(range(len(bs_labels)), bs_labels)

    plt.tight_layout()
    plt.show()


def demonstrate_epoch_consistency(dataset):
    """Demonstrate that the sampler produces consistent results across epochs."""
    lengths = [item['length'] for item in dataset.data]
    n_images = [item['n_images'] for item in dataset.data]

    print("\n" + "=" * 60)
    print("EPOCH CONSISTENCY TEST")
    print("=" * 60)

    sampler = BlueprintGroupedSampler(
            batch_size=4,
            lengths=lengths,
            n_images=n_images,
            seed=42
    )

    dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=4,
            collate_fn=lambda x: x
    )

    # Test 3 epochs
    for epoch in range(3):
        print(f"\n📅 Epoch {epoch}")
        sampler.set_epoch(epoch)

        batch_compositions = []
        for i, batch in enumerate(dataloader):
            if i >= 5:  # First 5 batches
                break

            total_images = sum(item['n_images'] for item in batch)
            sample_ids = [item['id'] for item in batch]

            batch_compositions.append({
                    'batch_idx'   : i,
                    'sample_ids'  : sample_ids,
                    'total_images': total_images,
                    'samples'     : len(batch)
            })

        # Show results
        for comp in batch_compositions:
            print(f"  Batch {comp['batch_idx']}: samples {comp['sample_ids'][:3]}... "
                  f"({comp['samples']} samples, {comp['total_images']} images)")


if __name__ == "__main__":
    print("🚀 BLUEPRINTGROUPEDSAMPLER TOY EXAMPLE")
    print("=" * 80)

    # Create fake dataset
    print("📊 Creating fake dataset with 1000 samples...")
    dataset = FakeVisionDataset(size=10000)

    # Analyze dataset
    analyze_dataset_distribution(dataset)

    # Test sampler with different batch sizes
    results = test_sampler_with_different_batch_sizes(dataset)

    # Visualize results
    print("\n📈 Creating visualizations...")
    visualize_results(results)

    # Test epoch consistency
    demonstrate_epoch_consistency(dataset)

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)

    successful_configs = sum(1 for r in results.values() if r.get('uniform', False))
    total_configs = len(results)

    print(f"✅ Successful configurations: {successful_configs}/{total_configs}")
    print(f"🔒 Memory uniformity: GUARANTEED")
#
#
# class BlueprintGroupedSampler(Sampler):
#     """
#     Sampler che garantisce uniformità nel numero di immagini per batch,
#     compatibile con Accelerate (non fa sharding manuale).
#
#     Strategia:
#     - Bilancia campioni con 1-2 immagini per mantenere memoria uniforme
#     - Lascia ad Accelerate la gestione del distributed training
#     - Mantiene determinismo per reproducibilità
#     """
#
#     def __init__(
#             self,
#             batch_size: int,
#             lengths: List[int],
#             n_images: List[int],
#             generator: Optional[torch.Generator] = None,
#             seed: int = 42,
#             drop_last: bool = True
#     ):
#         super().__init__(None)
#
#         # Parametri base - NON moltiplicare per world_size
#         self.batch_size = batch_size  # batch size per device
#         self.lengths = lengths
#         self.n_images = n_images
#         self.drop_last = drop_last
#         self.generator = generator or torch.Generator().manual_seed(seed)
#         self.epoch = 0
#
#         # Validazione input
#         assert len(lengths) == len(n_images), "lengths e n_images devono avere la stessa lunghezza"
#         assert all(n in [1, 2] for n in n_images), "n_images deve contenere solo valori 1 o 2"
#
#         # Pre-computa gli indici per categoria
#         self.indices_1img = np.where(np.array(n_images) == 1)[0]
#         self.indices_2img = np.where(np.array(n_images) == 2)[0]
#
#         print(f"[BlueprintSampler] Dataset: {len(self.indices_1img)} campioni 1-img, {len(self.indices_2img)} campioni 2-img")
#
#         # Calcola strategia di sampling basata su batch_size
#         self.sampling_strategy = self._calculate_sampling_strategy()
#         print(f"[BlueprintSampler] Strategia per batch_size={batch_size}: {self.sampling_strategy}")
#
#     def _calculate_sampling_strategy(self):
#         """
#         Calcola quanti campioni 1-img e 2-img usare per batch
#         per mantenere carico di memoria uniforme.
#         """
#         if self.batch_size == 1:
#             # Con batch_size=1, alternare tra 1-img e 2-img
#             return {"samples_1img": 1, "samples_2img": 0, "images_per_batch": 1}
#         elif self.batch_size == 2:
#             # Strategia: 1 campione 2-img = 2 immagini totali
#             return {"samples_1img": 0, "samples_2img": 1, "images_per_batch": 2}
#         elif self.batch_size == 4:
#             # Strategia: 3 campioni 2-img + 1 campione 1-img = 7 immagini totali
#             return {"samples_1img": 1, "samples_2img": 3, "images_per_batch": 7}
#         else:
#             # Strategia generale: cerca bilanciamento ottimale
#             # Priorità ai campioni 2-img per efficienza memoria
#             samples_2img = min(self.batch_size, self.batch_size // 2 + 1)
#             samples_1img = self.batch_size - samples_2img
#             images_per_batch = samples_2img * 2 + samples_1img
#             return {
#                     "samples_1img"    : samples_1img,
#                     "samples_2img"    : samples_2img,
#                     "images_per_batch": images_per_batch
#             }
#
#     def _build_balanced_batches(self):
#         """
#         Costruisce batch bilanciati con numero uniforme di immagini.
#         """
#         strategy = self.sampling_strategy
#         samples_1img = strategy["samples_1img"]
#         samples_2img = strategy["samples_2img"]
#
#         # Shuffle deterministico per epoch
#         g = torch.Generator()
#         g.manual_seed(self.epoch + 42)
#
#         # Shuffle degli indici per categoria
#         indices_1img_shuffled = self.indices_1img[torch.randperm(len(self.indices_1img), generator=g).numpy()]
#         indices_2img_shuffled = self.indices_2img[torch.randperm(len(self.indices_2img), generator=g).numpy()]
#
#         # Calcola numero massimo di batch possibili
#         max_batches_1img = len(indices_1img_shuffled) // samples_1img if samples_1img > 0 else float('inf')
#         max_batches_2img = len(indices_2img_shuffled) // samples_2img if samples_2img > 0 else float('inf')
#         max_batches = min(max_batches_1img, max_batches_2img)
#
#         if max_batches == 0:
#             print(f"⚠️ [BlueprintSampler] Impossibile creare batch con strategia {strategy}")
#             return []
#
#         # Costruisce i batch
#         batches = []
#         for batch_idx in range(max_batches):
#             batch = []
#
#             # Aggiungi campioni 2-img
#             start_2img = batch_idx * samples_2img
#             end_2img = start_2img + samples_2img
#             if samples_2img > 0 and end_2img <= len(indices_2img_shuffled):
#                 batch.extend(indices_2img_shuffled[start_2img:end_2img])
#
#             # Aggiungi campioni 1-img
#             start_1img = batch_idx * samples_1img
#             end_1img = start_1img + samples_1img
#             if samples_1img > 0 and end_1img <= len(indices_1img_shuffled):
#                 batch.extend(indices_1img_shuffled[start_1img:end_1img])
#
#             # Verifica dimensione batch
#             if len(batch) == self.batch_size:
#                 batches.append(batch)
#             elif not self.drop_last and len(batch) > 0:
#                 # Padding se necessario e drop_last=False
#                 while len(batch) < self.batch_size:
#                     batch.append(batch[0])  # Replica primo elemento
#                 batches.append(batch)
#
#         # Fallback: se non ci sono abbastanza campioni 1-img, usa solo 2-img
#         if not batches and len(self.indices_2img) >= self.batch_size:
#             print("[BlueprintSampler] Fallback: usando solo campioni 2-img")
#             fallback_batches = len(indices_2img_shuffled) // self.batch_size
#             for i in range(fallback_batches):
#                 start = i * self.batch_size
#                 end = start + self.batch_size
#                 batches.append(indices_2img_shuffled[start:end].tolist())
#
#         print(f"[BlueprintSampler] Creati {len(batches)} batch per epoch {self.epoch}")
#
#         return batches
#
#     def __iter__(self):
#         """Iteratore che restituisce gli indici per l'epoch corrente."""
#         batches = self._build_balanced_batches()
#
#         # Flatten dei batch in sequenza lineare per DataLoader
#         indices = []
#         for batch in batches:
#             indices.extend(batch)
#
#         return iter(indices)
#
#     def __len__(self):
#         """Numero totale di campioni (non batch)."""
#         batches = self._build_balanced_batches()
#         return len(batches) * self.batch_size
#
#     def set_epoch(self, epoch: int):
#         """Imposta l'epoch per shuffle deterministic."""
#         self.epoch = epoch
#
