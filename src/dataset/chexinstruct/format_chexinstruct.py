import argparse
import sys

from datasets import disable_caching
sys.path.append('./')  # Ensure the src directory is in the path
from src.dataset import load_parquet_image_dataset, save_dataset_as_parquet
from src.models import (
    data_format_gemma_conversation,
    data_format_qwen25vl_conversation,
    data_format_llava15_conversation
)
import multiprocessing as mp
import psutil
# Model formatting function mapping
MODEL_FORMATTERS = {
        'gemma'   : data_format_gemma_conversation,
        'llava15' : data_format_llava15_conversation,
        'qwen25vl': data_format_qwen25vl_conversation,
}
# ---------------------------------------------------------------------
# ⬇️  NEW / MOVED IMPORTS
# ---------------------------------------------------------------------
from pathlib import Path
from functools import partial
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True     # let PIL load incomplete files
disable_caching()
def parse_arguments():
    parser = argparse.ArgumentParser(
            description='Format ChexInstruct dataset for different multimodal models',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python format_chexinstruct.py --model gemma --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_gemma_format
  python format_chexinstruct.py --model llava15 --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_llava15_format
  python format_chexinstruct.py --model qwen25vl --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_qwen25vl_format
        """
    )

    parser.add_argument(
            '--model',
            type=str,
            required=True,
            choices=['gemma', 'llava15', 'qwen25vl'],
            help='Target model format (gemma, llava15, qwen25vl)'
    )

    parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='Input dataset path (directory containing parquet files)'
    )

    parser.add_argument(
            '--output',
            type=str,
            required=True,
            help='Output directory path for formatted dataset'
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


def validate_paths(input_path: str, output_path: str):
    """Validate input and output paths."""
    input_dir = Path(input_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir



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



# ---------------------------------------------------------------------
# ⬇️  UTILITIES
# ---------------------------------------------------------------------
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
        return True                        # no image, keep example (e.g. text-only)
    if not isinstance(imgs, list):
        imgs = [imgs]                       # unify to list

    try:
        # Force full pixel decode for every image.
        for img in imgs:
            img.load()                      # raises OSError on real corruption
        return True
    
    except Exception as e:
        if log_bad is not None:
            with open(log_bad, "a") as f:
                f.write(f"{getattr(img, 'filename', '⟨in-memory⟩')} - {e}\n")
        return False


# ---------------------------------------------------------------------
# ⬇️  MAIN PIPELINE (PATCHED)
# ---------------------------------------------------------------------
def format_dataset(model_name: str,
                   input_path: str,
                   output_path: str,
                   splits: list,
                   verbose: bool = False):

    num_proc = get_optimal_num_proc()
    formatter = MODEL_FORMATTERS[model_name]

    # file where we’ll collect bad-image messages
    log_path = str(Path(output_path) / "corrupted_images.txt")

    if verbose:
        print(f"Loading ChexInstruct dataset from: {input_path}")

    chexinstruct_data = load_parquet_image_dataset(input_path)
    formatted_data    = {}

    for split in splits:
        if split not in chexinstruct_data:
            print(f"Warning: Split '{split}' not found. Skipping…")
            continue

        if verbose:
            print(f"Processing {split} split…")

        ds = chexinstruct_data[split]

        # -----------------------------------------------------------------
        # 1️⃣  PRE-FILTER  (multiprocess)
        # -----------------------------------------------------------------
        filter_fn = partial(has_valid_pil_images, log_bad=log_path)
        ds = ds.filter(
            filter_fn,
            num_proc=num_proc,          # multiprocess filtering
            desc="Filtering corrupted images"
        )

        # -----------------------------------------------------------------
        # 2️⃣  MAP  ➜  MODEL-SPECIFIC FORMAT
        # -----------------------------------------------------------------
        ds = ds.map(
            formatter,
            num_proc=num_proc,
            batch_size=256,
            desc="Formatting dataset"
        ).filter(lambda x: x is not None,   # drop examples the formatter skipped
                 desc="Removing failed samples")

        formatted_data[split] = ds
        if verbose:
            print(f"  – {split}: {len(ds):,} samples after formatting")

    if verbose:
        print(f"Saving formatted dataset to: {output_path}")

    save_dataset_as_parquet(formatted_data, output_path)
    print(f"✅ Dataset formatted for {model_name.upper()} and saved to “{output_path}”.")
    print(f"📝 Corrupted images (if any) listed in: {log_path}")

    return formatted_data
def main():
    args = parse_arguments()

    try:
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