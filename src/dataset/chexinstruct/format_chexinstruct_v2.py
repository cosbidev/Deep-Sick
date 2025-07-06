import argparse
import sys
import os
import re
import pickle
import random
import gc
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import multiprocessing as mp
import psutil
from PIL import ImageFile, Image
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Sequence, disable_caching

ImageFile.LOAD_TRUNCATED_IMAGES = True  # let PIL load incomplete files
disable_caching()

# Add src to path
sys.path.append('./')
from src.dataset import load_parquet_image_dataset, save_dataset_as_parquet
from src.models import (
    data_format_gemma_conversation,
    data_format_qwen25vl_conversation,
    data_format_llava15_conversation
)

# Model formatting function mapping
MODEL_FORMATTERS = {
        'gemma'   : data_format_gemma_conversation,
        'llava15' : data_format_llava15_conversation,
        'qwen25vl': data_format_qwen25vl_conversation,
}


# ─────────────────────  DATASET CREATION FUNCTIONS  ────────────────────

def _flatten_one_sample(*args):
    """Flatten one sample from the raw dataset."""
    sample, missing, sampling_rate = args
    out = []

    img_paths = sample.get("image_path", [])
    if isinstance(img_paths, str):
        img_paths = [img_paths]

    if img_paths and any(p in missing for p in img_paths):
        return out  # skip due to missing image

    qa_pairs = sample.get("qa_pair", [])
    if not qa_pairs:
        return out
    if isinstance(qa_pairs, dict):
        qa_pairs = [qa_pairs]

    k = max(1, int(sampling_rate * len(qa_pairs)))
    for idx in random.sample(range(len(qa_pairs)), k):
        qa = qa_pairs[idx]

        ex = {
                "instruction": qa.get("q", ""),
                "response"   : qa.get("a", "")
        }
        if img_paths:
            ex["image"] = img_paths

        out.append(ex)

    return out


def _unpack_and_flatten(s, missing, sampling_rate):
    """Wrapper function for parallel processing."""
    return _flatten_one_sample(s, missing, sampling_rate)


def load_split_parallel(pkl_path, missing, sampling_rate=0.3, max_workers=32, chunksize=512):
    """Load and process a pickle split in parallel."""
    print(f"⏳  Loading split: {os.path.basename(pkl_path)}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    flattened = []

    # Create a worker function with missing and sampling_rate frozen in
    worker_fn = partial(_unpack_and_flatten, missing=missing, sampling_rate=sampling_rate)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(worker_fn, raw, chunksize=chunksize)

        # Wrap the iterator with tqdm to show progress
        for ex_list in tqdm(results, total=len(raw), desc=f"▸ {os.path.basename(pkl_path)}"):
            flattened.extend(ex_list)

    feat = Features({
            "instruction": Value("string"),
            "response"   : Value("string"),
            "image"      : Sequence(HFImage())
    })

    del raw
    gc.collect()

    return Dataset.from_list(flattened, features=feat)


def create_dataset_from_pickles(
        data_dir: str,
        sampling_rate: float = 0.01,
        max_workers: int = 32,
        chunksize: int = 512,
        seed: int = 42
        ):
    """Create HuggingFace dataset from pickle files."""
    random.seed(seed)

    pickle_splits = {
            "train": os.path.join(data_dir, "data_train_chexinstruct.pkl"),
            "val"  : os.path.join(data_dir, "data_val_chexinstruct.pkl"),
            "test" : os.path.join(data_dir, "data_test_chexinstruct.pkl")
    }

    # Load missing images
    missing_images = []
    missing_file = "missing_images.txt"
    if os.path.exists(missing_file):
        pat = re.compile(r": (.*?) does not exist")
        with open(missing_file) as fh:
            missing_images = [
                    m.group(1) for ln in fh
                    if (m := pat.search(ln)) and not os.path.exists(m.group(1))
            ]

    print(f"Found {len(missing_images)} missing images to skip")
    print("Starting dataset creation...")

    # Load all splits
    dataset_dict = {}
    for split_name, pkl_path in pickle_splits.items():
        if os.path.exists(pkl_path):
            dataset_dict[split_name] = load_split_parallel(
                    pkl_path, missing_images, sampling_rate, max_workers, chunksize
            )
        else:
            print(f"Warning: {pkl_path} not found, skipping {split_name} split")

    return DatasetDict(dataset_dict)


# ─────────────────────  FORMATTING FUNCTIONS  ──────────────────────

def get_optimal_num_proc():
    """Get optimal number of processes based on system resources."""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024 ** 3)

    # Conservative approach: use fewer processes if memory is limited
    if memory_gb < 16:
        return min(4, cpu_count // 2)
    elif memory_gb < 32:
        return min(8, cpu_count // 2)
    elif memory_gb < 64:
        return min(16, cpu_count // 2)
    else:
        cpu_count //= 2
    print("Using optimal number of processes:", cpu_count)

    return cpu_count if cpu_count > 0 else 1


def has_valid_pil_images(example, log_bad=None):
    """
    Validate that every PIL.Image in `example["image"]` truly decodes.

    Parameters
    ----------
    example : dict
        A single HF-Dataset row with an "image" key.
        The value can be a PIL.Image or a list of PIL.Image objects.
    log_bad : str | None
        Path to a text file where failures are appended.

    Returns
    -------
    bool
        True  -> keep this example
        False -> drop it
    """
    imgs = example["image"]
    if imgs is None:
        return True  # no image, keep example (e.g. text-only)
    if not isinstance(imgs, list):
        imgs = [imgs]  # unify to list

    try:
        # Force full pixel decode for every image.
        for img in imgs:
            img.load()  # raises OSError on real corruption
        return True

    except Exception as e:
        if log_bad is not None:
            with open(log_bad, "a") as f:
                f.write(f"{getattr(img, 'filename', '⟨in-memory⟩')} - {e}\n")
        return False


def format_dataset(
        model_name: str,
        input_path: str,
        output_path: str,
        splits: list,
        verbose: bool = False
        ):
    """Format dataset for specific model."""
    num_proc = get_optimal_num_proc()
    formatter = MODEL_FORMATTERS[model_name]

    # file where we'll collect bad-image messages
    log_path = str(Path(output_path) / "corrupted_images.txt")

    if verbose:
        print(f"Loading ChexInstruct dataset from: {input_path}")

    chexinstruct_data = load_parquet_image_dataset(input_path)
    formatted_data = {}

    for split in splits:
        if split not in chexinstruct_data:
            print(f"Warning: Split '{split}' not found. Skipping…")
            continue

        if verbose:
            print(f"Processing {split} split…")

        ds = chexinstruct_data[split]

        # Pre-filter corrupted images
        filter_fn = partial(has_valid_pil_images, log_bad=log_path)
        ds = ds.filter(
                filter_fn,
                num_proc=num_proc,
                desc="Filtering corrupted images"
        )

        # Apply model-specific formatting
        ds = ds.map(
                formatter,
                num_proc=num_proc,
                batch_size=256,
                desc="Formatting dataset"
        ).filter(lambda x: x is not None,
                 desc="Removing failed samples")

        formatted_data[split] = ds
        if verbose:
            print(f"  – {split}: {len(ds):,} samples after formatting")

    if verbose:
        print(f"Saving formatted dataset to: {output_path}")

    save_dataset_as_parquet(formatted_data, output_path)
    print(f"✅ Dataset formatted for {model_name.upper()} and saved to {output_path}.")
    print(f"📝 Corrupted images (if any) listed in: {log_path}")


    return formatted_data


def validate_paths(input_path: str, output_path: str):
    """Validate input and output paths."""
    input_dir = Path(input_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir


# ─────────────────────  ARGUMENT PARSING  ──────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
            description='Create and format ChexInstruct dataset for different multimodal models',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Examples:
      # Create dataset from pickle files
      python format_chexinstruct.py --create --data-dir data_chexinstruct --output data_chexinstruct/hf_parquet
    
      # Format existing dataset
      python format_chexinstruct.py --model gemma --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_gemma_format
    
      # Create and format in one command
      python format_chexinstruct.py --create --data-dir data_chexinstruct --model gemma --output data_chexinstruct/hf_parquet_gemma_format
        """
    )

    # Mode selection
    parser.add_argument(
            '--create',
            action='store_true',
            help='Create dataset from pickle files'
    )

    parser.add_argument(
            '--format',
            action='store_true',
            help='Format existing dataset (default if --create not specified)'
    )

    # Dataset creation arguments
    parser.add_argument(
            '--data-dir',
            type=str,
            help='Directory containing pickle files (required for --create)'
    )

    parser.add_argument(
            '--sampling-rate',
            type=float,
            default=0.01,
            help='Sampling rate for QA pairs (default: 0.01)'
    )

    parser.add_argument(
            '--max-workers',
            type=int,
            default=32,
            help='Maximum number of worker processes (default: 32)'
    )

    parser.add_argument(
            '--chunksize',
            type=int,
            default=512,
            help='Chunk size for parallel processing (default: 512)'
    )

    parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed (default: 42)'
    )

    # Formatting arguments
    parser.add_argument(
            '--model',
            type=str,
            choices=['gemma', 'llava15', 'qwen25vl'],
            help='Target model format (required for formatting)'
    )

    parser.add_argument(
            '--input',
            type=str,
            help='Input dataset path (required for formatting)'
    )

    parser.add_argument(
            '--output',
            type=str,
            required=True,
            help='Output directory path'
    )

    parser.add_argument(
            '--splits',
            type=str,
            nargs='+',
            default=['train', 'val', 'test'],
            help='Dataset splits to process (default: train val test)'
    )

    parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
    )

    return parser.parse_args()


# ─────────────────────  MAIN FUNCTION  ──────────────────────

def main():
    args = parse_arguments()

    # Determine mode
    if args.create and args.format:
        print("Error: Cannot specify both --create and --format")
        sys.exit(1)

    if not args.create and not args.format:
        # Default to format mode if neither specified
        args.format = True

    try:
        # Create dataset from pickle files
        if args.create:
            if not args.data_dir:
                print("Error: --data-dir is required when using --create")
                sys.exit(1)

            if args.verbose:
                print("Creating dataset from pickle files...")

            dataset = create_dataset_from_pickles(
                    data_dir=args.data_dir,
                    sampling_rate=args.sampling_rate,
                    max_workers=args.max_workers,
                    chunksize=args.chunksize,
                    seed=args.seed
            )

            # If model is specified, format the dataset
            if args.model:
                if args.verbose:
                    print(f"Formatting dataset for {args.model}...")

                # Save intermediate raw dataset
                intermediate_dir = str(Path(args.output).parent / "hf_parquet_intermediate")
                save_dataset_as_parquet(dataset, intermediate_dir)

                # Format the dataset
                formatted_data = format_dataset(
                        model_name=args.model,
                        input_path=intermediate_dir,
                        output_path=args.output,
                        splits=args.splits,
                        verbose=args.verbose
                )

                # Clean up intermediate files if formatting succeeded
                import shutil
                shutil.rmtree(intermediate_dir)

            else:
                # Just save the raw dataset
                save_dataset_as_parquet(dataset, args.output)
                print(f"✅ Dataset created and saved to: {args.output}")
                print("Dataset creation completed successfully.")
                print("You can now load it with `load_dataset` from the `datasets` library.")
                print(f"Example usage:\n"
                      f"```python\n"
                      f"from datasets import load_dataset\n"
                      f"dataset = load_dataset('path/to/{args.output}')\n"
                      f"```")

        # Format existing dataset
        elif args.format:
            if not args.model:
                print("Error: --model is required when formatting")
                sys.exit(1)

            if not args.input:
                print("Error: --input is required when formatting")
                sys.exit(1)

            validate_paths(args.input, args.output)

            format_dataset(
                    model_name=args.model,
                    input_path=args.input,
                    output_path=args.output,
                    splits=args.splits,
                    verbose=args.verbose
            )

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()