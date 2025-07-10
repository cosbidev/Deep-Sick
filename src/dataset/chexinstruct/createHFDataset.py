import os, re, pickle, random, gc
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence, load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
import sys
import multiprocessing as mp
from itertools import islice
import numpy as np

sys.path.append('./')
sys.path.append('../')
os.getcwd()

from src.dataset import save_dataset_as_parquet, load_parquet_image_dataset
from src.dataset import data_format_gemma_conversation_chexinstruct, data_format_qwen25vl_conversation, data_format_llava15_conversation

# Model formatting function mapping
MODEL_FORMATTERS = {
        'gemma'   : data_format_gemma_conversation_chexinstruct,
        'llava15' : data_format_llava15_conversation,
        'qwen25vl': data_format_qwen25vl_conversation,
}





def parse_arguments():
    parser = argparse.ArgumentParser(
            description='Format ChexInstruct dataset for different multimodal models',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=
            """
            Examples:
              python src/dataset/chexinstruct/createHFDataset.py --model gemma --input data_chexinstruct --output data_chexinstruct/hf_parquet_gemma_format
              python src/dataset/chexinstruct/createHFDataset.py --model llava15 --input data_chexinstruct --output data_chexinstruct/hf_parquet_llava15_format
              python src/dataset/chexinstruct/createHFDataset.py --model qwen25vl --input data_chexinstruct --output data_chexinstruct/hf_parquet_qwen25vl_format
            """
    )

    parser.add_argument('--model', type=str, default='gemma', choices=['gemma', 'llava15', 'qwen25vl'],
                        help='Target model format (gemma, llava15, qwen25vl)')
    parser.add_argument('--input', type=str, default='data_chexinstruct',
                        help='Input dataset path (directory containing json files)')
    parser.add_argument('--output', type=str, default="data_chexinstruct/hf_parquet_gemma_format",
                        help='Output directory path for formatted dataset')
    parser.add_argument('--splits', type=str, nargs='+', default=['val'],
                        help='Dataset splits to process (default: train val test)')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of worker processes (default: CPU count)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of samples to process per chunk (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for processing (default: 100)')


    return parser.parse_args()


# ─────────────────────  OPTIMIZED SAMPLE PROCESSING  ────────────────────

def process_sample_batch(batch_data):
    """Process a batch of samples in a single worker process"""
    samples_batch, missing_images, sampling_rate, formatter, task, load_path = batch_data

    pattern = r'\[[^\]]+\]'
    results = []

    for sample in samples_batch:
        try:
            # Quick task filtering
            task_data = sample.get("unique_id", "")
            if not task_data:
                continue

            match1 = re.search(pattern, task_data)
            if not match1:
                continue

            task_text = match1.group()
            if task.lower() not in task_text.lower():
                continue

            # Check image paths
            img_paths = sample.get("image_path", [])
            if not img_paths:
                continue

            if isinstance(img_paths, str):
                if not os.path.exists(img_paths) or img_paths in missing_images:
                    continue
                img_paths = [img_paths]
            elif isinstance(img_paths, list):
                if not img_paths:
                    continue
                # Filter out missing images
                img_paths = [p for p in img_paths if os.path.exists(p) and p not in missing_images]
                if not img_paths:
                    continue

            # Process QA pairs
            qa_pairs = sample.get("qa_pair", [])
            if not qa_pairs:
                continue

            if isinstance(qa_pairs, dict):
                qa_pairs = [qa_pairs]

            # Sample QA pairs
            k = max(1, int(sampling_rate * len(qa_pairs)))
            selected_indices = random.sample(range(len(qa_pairs)), min(k, len(qa_pairs)))

            for idx in selected_indices:
                qa = qa_pairs[idx]

                ex = {
                        "instruction": qa.get("q", ""),
                        "response"   : qa.get("a", ""),
                        "messages"   : None,
                        "task"       : task_text,
                        "image"      : img_paths
                }

                # Format the sample
                kwargs_format = {"samples": ex, "single": True, 'load_path': load_path}
                formatted_result = formatter(**kwargs_format)
                ex.update(formatted_result)

                # Only keep samples with valid messages
                if ex.get('messages') and len(ex['messages']) > 0:
                    results.append(ex)

        except Exception as e:
            # Log error but continue processing
            print(f"Error processing sample: {e}")
            continue

    return results


def chunk_generator(data, chunk_size):
    """Generator that yields chunks of data"""
    iterator = iter(data)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def load_split_optimized(pkl_path, missing_images, sampling_rate=0.3, max_workers=None, chunk_size=1000, **kwargs):
    """Optimized split loading with proper multiprocessing and memory management"""

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 16)  # Don't use too many processes

    print(f"⏳ Loading split: {os.path.basename(pkl_path)} with {max_workers} workers")

    try:
        # Load data in chunks to avoid memory issues
        with open(pkl_path, "rb") as f:
            raw_data = pickle.load(f, encoding="latin1")

        print(f"📊 Total samples in {os.path.basename(pkl_path)}: {len(raw_data)}")

        # Convert missing_images to set for faster lookup
        missing_set = set(missing_images)

        # Prepare parameters for workers
        formatter = kwargs.get('formatter')
        task = kwargs.get('task', '')
        load_path = kwargs.get('load_path', True)

        all_results = []

        # Process data in chunks
        chunks = list(chunk_generator(raw_data, chunk_size))
        total_chunks = len(chunks)

        print(f"🔄 Processing {total_chunks} chunks of size {chunk_size}")

        # Prepare batch data for workers
        batch_data_list = [
                (chunk, missing_set, sampling_rate, formatter, task, load_path)
                for chunk in chunks
        ]

        # Use ProcessPoolExecutor with proper context management
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                    executor.submit(process_sample_batch, batch_data): i
                    for i, batch_data in enumerate(batch_data_list)
            }

            # Process completed futures with progress bar
            with tqdm(total=total_chunks, desc=f"▸ {os.path.basename(pkl_path)}") as pbar:
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        all_results.extend(chunk_results)
                        pbar.update(1)
                        pbar.set_postfix({"samples": len(all_results)})
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                        pbar.update(1)

        # Clean up raw data immediately
        del raw_data
        del chunks
        del batch_data_list
        gc.collect()

        print(f"✅ Processed {len(all_results)} valid samples from {os.path.basename(pkl_path)}")

        if not all_results:
            print(f"⚠️  No valid samples found in {os.path.basename(pkl_path)}")
            return None

        # Create features schema
        feat = Features({
                "instruction": Value("string"),
                "response"   : Value("string"),
                "task"       : Value("string"),
                "image"      : Sequence(Image(decode=False)),
                "messages"   : Sequence(Sequence({
                        "role"   : Value("string"),
                        "content": Sequence({
                                "type"      : Value("string"),
                                "text"      : Value("string"),
                                "image"     : Image(decode=False),
                                "image_path": Value("string"),
                        })
                })),
        })

        # Create dataset from results
        dataset = Dataset.from_list(all_results, features=feat)

        # Clean up results
        del all_results
        gc.collect()

        return dataset

    except Exception as e:
        print(f"❌ Error loading {pkl_path}: {e}")
        return None


def load_missing_images_optimized(missing_file="missing_images.txt"):
    """Optimized loading of missing images list"""
    missing_images = set()
    if os.path.exists(missing_file):
        pat = re.compile(r": (.*?) does not exist")
        with open(missing_file, 'r') as fh:
            for line in fh:
                match = pat.search(line)
                if match:
                    img_path = match.group(1)
                    if not os.path.exists(img_path):
                        missing_images.add(img_path)
    return missing_images


if __name__ == "__main__":
    # ─────────────────────────  CONFIG  ──────────────────────────
    args = parse_arguments()

    # Set random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # Configuration
    SAMPLING_QA = 0.01
    task = ''
    load_path = True

    # Get parameters from args
    data_dir = args.input
    output_dir = args.output
    model = args.model
    max_workers = args.max_workers
    chunk_size = args.chunk_size

    formatter = MODEL_FORMATTERS[model]

    # Define pickle file paths
    pickle_splits = {
            #"train": os.path.join(data_dir, "data_train_chexinstruct.pkl"),
            "val"  : os.path.join(data_dir, "data_val_chexinstruct.pkl"),
            "test" : os.path.join(data_dir, "data_test_chexinstruct.pkl")
    }

    # Load missing images efficiently
    print("📋 Loading missing images list...")
    missing_images = load_missing_images_optimized("missing_images.txt")
    print(f"📋 Found {len(missing_images)} missing images")

    print("🚀 Starting optimized dataset creation...")

    # ─────────────────────  LOAD ALL SPLITS  ─────────────────────
    dataset_dict = {}

    for split_name, pkl_path in pickle_splits.items():
        if split_name in args.splits and os.path.exists(pkl_path):
            print(f"\n🔄 Processing {split_name} split...")

            dataset = load_split_optimized(
                    pkl_path=pkl_path,
                    missing_images=missing_images,
                    sampling_rate=SAMPLING_QA,
                    max_workers=max_workers,
                    chunk_size=chunk_size,
                    formatter=formatter,
                    task=task,
                    load_path=load_path
            )

            if dataset is not None:
                dataset_dict[split_name] = dataset
                print(f"✅ {split_name}: {len(dataset)} samples")
            else:
                print(f"⚠️  Skipping {split_name} (no valid samples)")
        else:
            if split_name in args.splits:
                print(f"⚠️  {pkl_path} not found, skipping {split_name}")

    if not dataset_dict:
        print("❌ No valid datasets created!")
        sys.exit(1)

    # Create final dataset
    final_dataset = DatasetDict(dataset_dict)

    # ─────────────────────  SAVE TO DISK  ────────────────────────
    final_output_dir = os.path.join(output_dir, f"{model}_{task}" if task else model)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"\n💾 Saving dataset to: {final_output_dir}")
    save_dataset_as_parquet(dataset_dict, final_output_dir)

    # ─────────────────────  FINAL SUMMARY  ──────────────────────
    print("\n" + "=" * 50)
    print("✅ Dataset creation completed successfully!")
    print(f"📁 Output directory: {final_output_dir}")
    print("\n📊 Dataset summary:")

    total_samples = 0
    for split_name, dataset in dataset_dict.items():
        samples = len(dataset)
        total_samples += samples
        print(f"  • {split_name}: {samples:,} samples")

    print(f"  • Total: {total_samples:,} samples")

    print("\n📖 Usage example:")
    print(f"```python")
    print(f"from datasets import load_dataset")
    print(f"dataset = load_dataset('{final_output_dir}')")
    print(f"```")

    # Test loading
    print("\n🔍 Testing dataset loading...")
    try:
        loaded_dict = load_parquet_image_dataset(final_output_dir)
        print("✅ Dataset loads successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")