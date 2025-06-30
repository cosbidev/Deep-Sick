import random, os, re, pickle, json
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence
from tqdm import tqdm

# ────────────────────────────────────────────────
# 1.  ❱❱  CONFIGURE
# ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

data_dir       = "data_chexinstruct/"
pickle_splits  = {
    "train": os.path.join(data_dir, "data_train_chexinstruct.pkl"),
    "val"  : os.path.join(data_dir, "data_val_chexinstruct.pkl"),
    "test" : os.path.join(data_dir, "data_test_chexinstruct.pkl")
}

#  ➜  List of extra keys you want to keep from each qa_pair
EXTRA_QA_KEYS = ["unique_id"]     # ← edit / extend as needed

# ────────────────────────────────────────────────
# 2.  ❱❱  LOAD LIST OF MISSING IMAGES
# ────────────────────────────────────────────────
with open("missing_images.txt", "r") as f:
    missing_images_raw = [ln.strip() for ln in f]

missing_images = []
pat = re.compile(r": (.*?) does not exist")
for ln in missing_images_raw:
    m = pat.search(ln)
    if m and not os.path.exists(m.group(1)):
        missing_images.append(m.group(1))


# ────────────────────────────────────────────────
# 3.  ❱❱  SPLIT-LOADER
# ────────────────────────────────────────────────
def load_split(pkl_path, missing, sampling_rate=0.30):
    flattened, extra_keys_found = [], set()

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    for sample in tqdm(data, desc=f"▸ {os.path.basename(pkl_path)}", unit="item"):
        img_paths = sample.get("image_path", [])
        if isinstance(img_paths, str):                 # normalize
            img_paths = [img_paths]

        if img_paths and any(p in missing for p in img_paths):
            continue                                   # skip if ALL imgs missing

        qa_pairs = sample.get("qa_pair", [])
        if not qa_pairs:
            continue

        if isinstance(qa_pairs, dict):                 # ensure list
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

            # pull through any additional fields you listed
            for key in EXTRA_QA_KEYS:
                if key in qa:
                    ex[key] = sample[key]

            flattened.append(ex)

    # ── Build the Features dynamically ──────────────────────────
    feat_dict = {
        "instruction": Value("string"),
        "response"   : Value("string"),
        "image"      : Sequence(Image())               # can be empty / missing
    }
    for k in extra_keys_found:
        feat_dict[k] = Value("string")

    return Dataset.from_list(flattened, features=Features(feat_dict))

# ────────────────────────────────────────────────
# 4.  ❱❱  BUILD DATASETDICT
# ────────────────────────────────────────────────
dataset = DatasetDict({
    split: load_split(path, missing_images)
    for split, path in pickle_splits.items()
})

print(dataset)
print(dataset["train"][0])          # quick sanity-check

# ────────────────────────────────────────────────
# 5.  ❱❱  SAVE TO DISK
# ────────────────────────────────────────────────
cache_dir = os.path.join(data_dir, "hf_cache")
os.makedirs(cache_dir, exist_ok=True)
dataset.save_to_disk(cache_dir)
print(f"✅  Dataset cached at: {cache_dir}")
