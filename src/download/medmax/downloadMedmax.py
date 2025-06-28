
from datasets import load_dataset
if __name__ == "__main__":
    # Print the dataset structure

    print("Downloading and caching the MedMax dataset...")
    # Specify the target local directory for caching
    local_cache_dir = "./data/medmax"

    # Load and cache the dataset
    ds = load_dataset("mint-medmax/medmax_data", cache_dir=local_cache_dir)

    # Optionally, save the dataset to a local file
    ds.save_to_disk(local_cache_dir)
    print(f"Dataset saved to {local_cache_dir}")

    # If you want to load it back later, you can use:
    # loaded_ds = load_dataset(local_cache_dir, data_files="data.json")
    # print(loaded_ds)