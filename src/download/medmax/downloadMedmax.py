import os
from huggingface_hub import login
from datasets import load_dataset, DatasetDict

# === Configuration ===
HF_TOKEN = "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"  # 🔐 Replace with your token
DATASET_NAME = "mint-medmax/medmax_data"  # 🧠 Public or private dataset repo
CACHE_DIR = "./data/medmax"  # 📁 Local download directory
SAVE_PATH = "./exported_dataset"  # 📁 Optional path to save as CSV/JSON
SAVE_FORMAT = "csv"  # Choose between "csv" or "json"
SELECT_SPLITS = ["train", "validation"]  # Adjust as needed
SAVE_DATA = True  # Set to True to save the dataset to disk
# === Step 1: Authenticate ===
def authenticate_huggingface(token: str):
    try:
        login(token=token)
        print("✅ Successfully authenticated with Hugging Face.")
    except Exception as e:
        print("❌ Authentication failed:", e)
        exit(1)


# === Step 2: Load Dataset ===
def download_dataset(dataset_name: str, cache_dir: str, splits=None) -> DatasetDict:
    try:
        if splits:
            dataset = {
                split: load_dataset(dataset_name, split=split, cache_dir=cache_dir)
                for split in splits
            }
        else:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        print(f"✅ Dataset '{dataset_name}' downloaded to: {cache_dir}")
        return dataset
    except Exception as e:
        print("❌ Dataset loading failed:", e)
        exit(1)


dataset = download_dataset(
    DATASET_NAME, CACHE_DIR, SELECT_SPLITS
)

print(f"✅ Download complete. Splits available: {list(dataset.keys())}")
print(f"Dataset cached in: {CACHE_DIR}")


# === Step 3: Save Dataset (Optional) ===
# ------------------------------
# OPTIONAL: SAVE DATA TO DISK
# ------------------------------
if SAVE_DATA:
    os.makedirs(SAVE_PATH, exist_ok=True)
    for split_name, split_data in dataset.items():
        save_path = os.path.join(SAVE_PATH, f"{split_name}.{SAVE_FORMAT}")
        print(f"Saving split '{split_name}' to {save_path}")

        if SAVE_FORMAT == "json":
            split_data.to_json(save_path)
        elif SAVE_FORMAT == "csv":
            split_data.to_csv(save_path)
        else:
            raise ValueError("Unsupported format: use 'json' or 'csv'")

    print(f"📁 All splits saved to {save_path}")