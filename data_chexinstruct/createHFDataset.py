import os, re, pickle, random, gc
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence, load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from src import save_dataset_as_parquet, load_parquet_image_dataset

# ─────────────────────  SAMPLE FLATTENER  ────────────────────
def _flatten_one_sample(*args):
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




# ─────────────────────  PARALLEL SPLIT LOADER  ───────────────

def _unpack_and_flatten(s, missing, sampling_rate):
    return _flatten_one_sample(s, missing, sampling_rate)

def load_split_parallel(pkl_path, missing, sampling_rate=0.3):
    print(f"⏳  Loading split: {os.path.basename(pkl_path)}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    flattened = []

    # Create a worker function with missing and sampling_rate frozen in
    worker_fn = partial(_unpack_and_flatten, missing=missing, sampling_rate=sampling_rate)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(worker_fn, raw, chunksize=CHUNKSIZE)

        # Wrap the iterator with tqdm to show progress
        for ex_list in tqdm(results, total=len(raw), desc=f"▸ {os.path.basename(pkl_path)}"):
            flattened.extend(ex_list)

    feat = Features({
        "instruction": Value("string"),
        "response": Value("string"),
        "image": Sequence(Image())
    })

    del raw
    gc.collect()

    return Dataset.from_list(flattened, features=feat)
if __name__ == "__main__":

    # ─────────────────────────  CONFIG  ──────────────────────────
    SEED = 42
    random.seed(SEED)
    MAX_WORKERS = 32
    CHUNKSIZE = 512
    SAMPLING_QA = 0.01

    data_dir = "data_chexinstruct/"
    pickle_splits = {
            "train": os.path.join(data_dir, "data_train_chexinstruct.pkl"),
            "val" : os.path.join(data_dir, "data_val_chexinstruct.pkl"),
            "test": os.path.join(data_dir, "data_test_chexinstruct.pkl")
    }

    # ─────────────────────  MISSING IMAGES  ──────────────────────
    pat = re.compile(r": (.*?) does not exist")
    with open("missing_images.txt") as fh:
        missing_images = [
                m.group(1) for ln in fh
                if (m := pat.search(ln)) and not os.path.exists(m.group(1))
        ]
    print("Starting dataset creation...")

    # ─────────────────────  LOAD ALL SPLITS  ─────────────────────
    dataset_dict = {
    split_name: load_split_parallel(pkl_path, missing_images)
    for split_name, pkl_path in pickle_splits.items()
    }

    dataset = DatasetDict(dataset_dict)

    output_dir = "data_chexinstruct/hf_parquet"
    save_dataset_as_parquet(dataset_dict, output_dir)

    # ─────────────────────  SAVE TO DISK  ────────────────────────
    print(f"✅ Dataset saved to: {output_dir}")
    print("Dataset creation completed successfully.")
    print("You can now load it with `load_dataset` from the `datasets` library.")
    print("Example usage:")
    print(f"```python\n"
          f"from datasets import load_dataset\n"
          f"dataset = load_dataset('path/to/{output_dir}')\n"
          f"```")
    loaded_dict = load_parquet_image_dataset(output_dir)
