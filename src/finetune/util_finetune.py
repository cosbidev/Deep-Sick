import math
import os



import numpy as np
import random

from typing import List
from collections import Counter

import torch
from torch.utils.data import Sampler, DataLoader, Dataset, DistributedSampler
from transformers import MODEL_MAPPING

from src.dataset import load_parquet_image_dataset

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import torch.distributed as dist


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)




# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# New Code #
def evaluate(training_args, model, eval_dataloader, accelerator, logger):
    # Sync after evaluation

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):

        if step <= 2:
            pass
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


class DistributedSamplerWithBluprint(DistributedSampler):
    """
    A distributed sampler that can be used with a blueprint dataset.
    This is useful for distributed training where each process needs to sample from the same dataset.
    """
    def __init__(
            self, dataset,
            batch_size: int,
            lengths: List[int],
            n_images: List[int],
            **kwargs
            ):
        super().__init__(dataset, **kwargs)

        # Parametri base - NON moltiplicare per world_size
        self.batch_size = batch_size  # batch size per device
        self.lengths = lengths
        self.n_images = n_images

        # Validazione input
        assert len(lengths) == len(n_images), "lengths e n_images devono avere la stessa lunghezza"
        assert all(n in [1, 2] for n in n_images), "n_images deve contenere solo valori 1 o 2"

        # Pre-computa gli indici per categoria
        self.indices_1img = np.where(np.array(n_images) == 1)[0]
        self.indices_2img = np.where(np.array(n_images) == 2)[0]

        print(f"[BlueprintSampler] Dataset: {len(self.indices_1img)} campioni 1-img, {len(self.indices_2img)} campioni 2-img")

        # Calcola strategia di sampling basata su batch_size
        self.sampling_strategy = self._calculate_sampling_strategy()
        print(f"[BlueprintSampler] Strategia per batch_size={batch_size}: {self.sampling_strategy}")
    def len_total(self):
        """
        Restituisce il numero totale di campioni nel dataset.
        """
        return self.num_samples * self.num_replicas


    def _calculate_sampling_strategy(self):
        """
        Calcola quanti campioni 1-img e 2-img usare per batch
        per mantenere carico di memoria uniforme.
        """
        if self.batch_size == 1:
            # Con batch_size=1, alternare tra 1-img e 2-img
            return {"samples_1img": 1, "samples_2img": 0, "images_per_batch": 1}
        elif self.batch_size == 2:
            # Strategia: 1 campione 2-img = 2 immagini totali
            return {"samples_1img": 0, "samples_2img": 1, "images_per_batch": 2}
        elif self.batch_size == 4:
            # Strategia: 3 campioni 2-img + 1 campione 1-img = 7 immagini totali
            return {"samples_1img": 2, "samples_2img": 2, "images_per_batch": 7}
        else:
            # Strategia generale: cerca bilanciamento ottimale
            # Priorità ai campioni 2-img per efficienza memoria
            samples_2img = min(self.batch_size, self.batch_size // 2 + 1)
            samples_1img = self.batch_size - samples_2img
            images_per_batch: int = samples_2img * 2 + samples_1img
            return {
                    "samples_1img"    : samples_1img,
                    "samples_2img"    : samples_2img,
                    "images_per_batch": images_per_batch}

    def __iter__(self):

        strategy = self.sampling_strategy
        samples_1img = strategy["samples_1img"]
        samples_2img = strategy["samples_2img"]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices_2img_shuffled = torch.randperm(len(self.indices_2img), generator=g).numpy()
            indices_1img_shuffled = torch.randperm(len(self.indices_1img), generator=g).numpy()

            img1_indices = self.indices_1img[indices_1img_shuffled]
            img2_indices = self.indices_2img[indices_2img_shuffled]
        else:
            # Use the original order if not shuffling
            img1_indices = self.indices_1img
            img2_indices = self.indices_2img


        # Calcola numero massimo di batch possibili
        batches = []
        indices = []

        divider_max_samples = samples_2img * self.num_replicas
        if  len(img2_indices) % divider_max_samples != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            img2_indices = img2_indices[:len(img2_indices) - (len(img2_indices) % divider_max_samples)]
        batches_across_devices = len(img2_indices) / divider_max_samples


        for batch_idx in range(int(batches_across_devices)):
            batch = []

            # Aggiungi campioni 2-img
            start_2img = batch_idx * (samples_2img * self.num_replicas)
            end_2img = start_2img + (samples_2img * self.num_replicas)
            if samples_2img > 0 and end_2img <= len(img2_indices):
                batch.extend(img2_indices[start_2img:end_2img])

            # Aggiungi campioni 1-img
            start_1img = batch_idx * (samples_1img * self.num_replicas)
            end_1img = start_1img + (samples_1img * self.num_replicas)
            if samples_1img > 0 and end_1img <= len(img1_indices):
                batch.extend(img1_indices[start_1img:end_1img])

            # Verifica dimensione batch
            if len(batch) == self.batch_size * self.num_replicas:
                batches.append(batch)

        # Remaining images
        if end_1img != len(img1_indices):
            remaining_1img = img1_indices[end_1img:]
            if len(remaining_1img) % (self.batch_size * self.num_replicas) != 0:
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                remaining_1img = remaining_1img[:len(remaining_1img) - (len(remaining_1img) % (self.batch_size * self.num_replicas))]


        remaining_batches_across_devices = len(remaining_1img) / (self.batch_size * self.num_replicas)
        for batch_idx in range(int(remaining_batches_across_devices)):
            batch = []

            # Aggiungi campioni 1-img
            start_1img = batch_idx * (self.batch_size * self.num_replicas)
            end_1img = start_1img + (self.batch_size * self.num_replicas)
            if samples_1img > 0 and end_1img <= len(remaining_1img):
                batch.extend(remaining_1img[start_1img:end_1img])

            # Verifica dimensione batch
            if len(batch) == self.batch_size * self.num_replicas:
                batches.append(batch)

        if len(batches)// self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            batches = batches[:len(batches) - (len(batches) % self.num_replicas)]



        print(f"[BlueprintSampler] Creati {len(batches)} batch per epoch {self.epoch}")
        for batch in batches:
            indices.extend(batch)


        self.num_samples = math.ceil(len(indices) / self.num_replicas)  # type: ignore[arg-type]


        # subsample
        indices = indices[self.rank: len(indices): self.num_replicas]
        sub = [2, 2, 2, 2]

        # Method 1: Sliding window
        lst = [self.n_images[i] for i in indices]
        found = any(lst[i:i + len(sub)] == sub for i in range(len(lst) - len(sub) + 1))

        if found:
            print("Found:not correct (2,2,2,2) batch", found)
        assert len(indices) == self.num_samples


        return iter(indices)

class FakeVisionDataset(Dataset):
    """Fake dataset for testing with varying image counts and lengths."""

    def __init__(self, n_images=None, lenghts=None):
        # Generate fake data with controlled distribution
        self.data = []
        for i in range(len(n_images)):
            self.data.append({
                    'id'         : i,
                    'n_images'   : n_images[i],
                    'length'     : lenghts[i],
                    'fake_text'  : f"Sample {i} with {n_images[i]} images and {lenghts[i]} tokens",
                    'fake_images': [f"image_{i}_{j}.jpg" for j in range(n_images[i])]
            })
    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data[idx]


def analyze_dataset_distribution(dataset):
    """Analyze the distribution of image counts and lengths."""
    n_images = [item['n_images'] for item in dataset.data]
    lengths = [item['length'] for item in dataset.data]

    print("=" * 70)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Image count distribution
    img_dist = Counter(n_images)
    print(f"Image count distribution:")
    for count, freq in sorted(img_dist.items()):
        print(f"  {count} image(s): {freq} samples ({freq / len(dataset) * 100:.1f}%)")

    # Length statistics
    lengths_1img = [l for l, n in zip(lengths, n_images) if n == 1]
    lengths_2img = [l for l, n in zip(lengths, n_images) if n == 2]

    print(f"\nLength statistics:")
    if lengths_1img:
        print(f"  1-image samples: mean={np.mean(lengths_1img):.1f}, std={np.std(lengths_1img):.1f}")
    if lengths_2img:
        print(f"  2-image samples: mean={np.mean(lengths_2img):.1f}, std={np.std(lengths_2img):.1f}")
    print(f"  Overall: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")

def load_and_prepare_datasets(data_args):
    """Load and prepare the datasets."""
    if data_args.dataset_dir is not None:
        if os.path.exists(data_args.dataset_dir + '_tok'):
            data_args.dataset_dir = data_args.dataset_dir + '_tok'
        else:
            raise ValueError('The dataset directory does not exist in tok version. Please create the format_code.')

        raw_datasets = load_parquet_image_dataset(
                dataset_dir=data_args.dataset_dir,
                split_list=["train", "val"],
                cache_dir=data_args.cache_dir,
                keep_in_memory=True,
                num_proc=data_args.preprocessing_num_workers,
        )

        if data_args.data_debug:
            # Reduce the dataset size for debugging purposes
            for split in raw_datasets.keys():
                raw_datasets[split] = raw_datasets[split].select(range(500))

    elif data_args.dataset_name is None:
        raise ValueError(
                "You need to specify either a dataset name or a dataset directory. "
                "Use --dataset_name for a HF dataset or --dataset_dir to specify the dataset folder in local (parquet)."
        )

    return raw_datasets["train"], raw_datasets["val"]



def test_adaptive_sampler(dataset):
    """Test the adaptive sampler with different batch sizes."""
    lengths = [item['length'] for item in dataset.data]
    n_images = [item['n_images'] for item in dataset.data]

    batch_sizes_to_test = [4]

    print("\n" + "=" * 80)
    print("ADAPTIVE BLUEPRINT SAMPLER TESTING")
    print("=" * 80)

    results = {}

    for batch_size in batch_sizes_to_test:
        print(f"\n🎯 Testing batch_size={batch_size}")
        print("-" * 50)

        try:
            # First, let's debug what target and strategies should be
            print(f"Debug: Calculating target images for batch_size={batch_size}")

            # Manual calculation to debug
            if batch_size == 1:
                target = 1
            elif batch_size == 2:
                target = 3
            elif batch_size == 4:
                target = 7
            elif batch_size == 8:
                target = 13

            print(f"Expected target: {target} images")

            # Debug valid combinations
            print("Valid combinations for this batch_size:")
            valid_combinations = []
            for samples_2img in range(batch_size + 1):
                samples_1img = batch_size - samples_2img
                if samples_1img >= 0:
                    total_images = samples_2img * 2 + samples_1img
                    if total_images <= target:
                        valid_combinations.append((samples_2img, samples_1img, total_images))
                        print(f"  {samples_2img}×2img + {samples_1img}×1img = {total_images} images")

            # Find combinations that match our target
            matching = [combo for combo in valid_combinations if combo[2] == target]
            print(f"Combinations matching target {target}: {matching}")

            if not matching:
                print(f"⚠️ No valid combinations for batch_size={batch_size}, target={target}")
                print(f"Available totals: {[combo[2] for combo in valid_combinations]}")
                # Adjust target to nearest feasible
                feasible_targets = [combo[2] for combo in valid_combinations]
                target = min(feasible_targets, key=lambda x: abs(x - target))
                print(f"Adjusting target to nearest feasible: {target}")

            sampler = DistributedSamplerWithBluprint(
                    dataset=dataset,
                    batch_size=batch_size,
                    lengths=lengths,
                    n_images=n_images,
                    num_replicas=16,  # Assuming 16 ranks for testing
                    rank=0,  # Single rank for testing
                    shuffle=True,
                    seed=42,
                    drop_last=True
            )
            # Test with DataLoader
            dataloader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    collate_fn=lambda x: x
            )
            # Analyze batches
            batch_image_counts = []
            batch_sample_counts = []
            strategy_usage = {"primary": 0, "fallback": 0}

            for i, batch in enumerate(dataloader):
                total_images = sum(item['n_images'] for item in batch)
                #if total_images != target:
                print(f"Batch {i}: {len(batch)} samples, {total_images} images")
                batch_image_counts.append(total_images)
                batch_sample_counts.append(len(batch))

                # Determine which strategy was used
                count_1img = sum(1 for item in batch if item['n_images'] == 1)
                count_2img = sum(1 for item in batch if item['n_images'] == 2)

                print(f"Batch {i}: {len(batch)} samples, {total_images} images")
                print(f"  Batch {i}: {count_2img}×2img + {count_1img}×1img = {total_images} images total")

            # Validate results
            unique_image_counts = set(batch_image_counts)
            uniform_images = len(unique_image_counts) == 1


            results[batch_size] = {
                    'uniform_images': uniform_images,
                    'actual_images' : list(unique_image_counts),
                    'strategy_usage': strategy_usage,
                    'total_batches' : len(batch_image_counts)
            }

        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[batch_size] = {
                    'error'         : str(e),
                    'uniform_images': False,
                    'correct_target': False
            }

    return results


def demonstrate_strategy_switching(dataset):
    """Demonstrate adaptive strategy switching when 2-img samples run out."""
    print("\n" + "=" * 70)
    print("STRATEGY SWITCHING DEMONSTRATION")
    print("=" * 70)

    # Create a dataset with limited 2-image samples to force switching
    limited_2img_data = []
    count_1img = 0
    count_2img = 0

    for item in dataset:
        if item['n_images'] == 2 and count_2img < 10:  # Only 10 2-image samples
            limited_2img_data.append(item)
            count_2img += 1
        elif item['n_images'] == 1 and count_1img < 100:  # Many 1-image samples
            limited_2img_data.append(item)
            count_1img += 1

    print(f"Limited dataset: {count_1img} 1-image samples, {count_2img} 2-image samples")


if __name__ == "__main__":
    from easydict import EasyDict
    print("🎯 ADAPTIVE BLUEPRINTGROUPEDSAMPLER DEMONSTRATION")
    print("=" * 80)
    print("Key Feature: FIXED image count per batch with adaptive strategy switching")
    print("Example: batch_size=4 → ALWAYS 7 images per batch")
    print("  • Primary: 3×2img + 1×1img = 7 images")
    print("  • Fallback: 7×1img = 7 images (when 2-img samples run out)")
    print("=" * 80)

    # Create fake dataset
    print("\n📊 Creating fake dataset with 1000 samples...")
    CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")

    data_args = dict(
            dataset_dir="/Users/filruff/Desktop/PHD/PROGETTI/Deep-Sick/data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings",
            cache_dir=CACHE_DIR,
            data_debug=False,
            preprocessing_num_workers=1,
    )
    data_args = EasyDict(data_args)

    dataset_train, dataset_eval = load_and_prepare_datasets(data_args)
    # CREATE A DATASET FAKE WITH THE SAME STRUCTURE AND CONTENTS OF dataset
    dataset_train = FakeVisionDataset(n_images=dataset_train['n_of_images'], lenghts=dataset_train['length'])
    """Test the adaptive sampler with different batch sizes."""

    # Analyze dataset
    analyze_dataset_distribution(dataset_train)

    # Test adaptive sampler
    results_train = test_adaptive_sampler(dataset_train)

    # Demonstrate strategy switching
    demonstrate_strategy_switching(dataset_train)


    # Final summary
    print("\n" + "=" * 80)
    print("🎯 ADAPTIVE SAMPLER SUMMARY ------ TRAINING")
    print("=" * 80)

    for bs, result in results_train.items():

        if 'target_images' in result:
            target = result['target_images']
            success = result['uniform_images'] and result['correct_target']
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"Batch size {bs}: Target {target} images → {status}")

            if success and 'strategy_usage' in result:
                usage = result['strategy_usage']
                print(f"  Strategy usage: {usage['primary']} primary + {usage['fallback']} fallback batches")

    # Eval dataset
    print("\n📊 Evaluating dataset with adaptive sampler...")
    # CREATE A DATASET FAKE WITH THE SAME STRUCTURE AND CONTENTS OF dataset
    dataset_eval = FakeVisionDataset(n_images=dataset_eval['n_of_images'], lenghts=dataset_eval['length'])
    """Test the adaptive sampler with different batch sizes."""

    # Analyze dataset
    analyze_dataset_distribution(dataset_eval)

    # Test adaptive sampler
    results_eval = test_adaptive_sampler(dataset_eval)

    # Demonstrate strategy switching
    demonstrate_strategy_switching(dataset_eval)

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 ADAPTIVE SAMPLER SUMMARY ------ EVALUATION")
    print("=" * 80)

    for bs, result in results_eval.items():

        if 'target_images' in result:
            target = result['target_images']
            success = result['uniform_images'] and result['correct_target']
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"Batch size {bs}: Target {target} images → {status}")

            if success and 'strategy_usage' in result:
                usage = result['strategy_usage']
                print(f"  Strategy usage: {usage['primary']} primary + {usage['fallback']} fallback batches")



def compute_SFT(model, batch):
    """
    TODO Complete the loss for SFT (Supervised Fine-Tuning) tasks.
    Compute the loss for SFT (Supervised Fine-Tuning) tasks.
    """
    # Ensure model is in training mode
    model.train()

    # Forward pass
    outputs = model(**batch)

    # Extract loss
    loss = outputs.loss


    return loss

def compute_DFT(model, batch):
    """
    TODO Complete the loss for DFT (Dynamic SUpervised Fine-Tuning) tasks.
    Compute the loss for SFT (Supervised Fine-Tuning) tasks.
    """
    # Ensure model is in training mode
    model.train()

    # Forward pass
    outputs = model(**batch)

    # Extract loss
    loss = outputs.loss


    return loss

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return