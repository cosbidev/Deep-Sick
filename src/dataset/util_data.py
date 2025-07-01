import os
import torch
from datasets import load_dataset, DatasetDict
import torchvision.transforms as T
def load_parquet_image_dataset(dataset_dir: str) -> DatasetDict:
    """
    Loads a DatasetDict from a directory of split-based parquet files,
    and casts the 'image' column to Sequence[Image].
    """
    split_files = {
        split_file.replace("data_", "").replace(".parquet", ""): os.path.join(dataset_dir, split_file)
        for split_file in os.listdir(dataset_dir)
        if split_file.endswith(".parquet")
    }

    dataset_dict = load_dataset("parquet", data_files=split_files)

    return dataset_dict
def save_dataset_as_parquet(dataset_dict, output_dir):
    """
    Save a DatasetDict to Parquet format without copying image files.

    Args:
        dataset_dict (DatasetDict): Hugging Face dataset with an "image" column (list or str).
        output_dir (str): Destination directory where Parquet files will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)


    for split in dataset_dict:
        parquet_path = os.path.join(output_dir, f"data_{split}.parquet")
        dataset_dict[split].to_parquet(parquet_path)
        print(f"✅ Saved split '{split}' to: {parquet_path}")



def image_preproc(img_size: int=224) -> T.Compose:
    """
    Returns a torchvision transform for image preprocessing.
    Args:
        img_size (int): Size to which the image will be resized.

    Returns:
        Transforms.Compose: A composed transform that resizes, crops, and converts images to tensors.

    """
    return T.Compose(
            [T.Resize(img_size + 16), # Resize to a larger size to ensure center crop works well
             T.CenterCrop(img_size), # Center crop to the desired size
             T.ToTensor() # Convert PIL Image or numpy.ndarray to tensor
             ]
    )



class SimpleCollator:
    """Simple collator that can be pickled"""

    def __init__(self, tokenizer, has_text=True, has_images=False):
        self.tokenizer = tokenizer
        self.has_text = has_text
        self.has_images = has_images

    def __call__(self, batch):
        if self.has_text and not self.has_images:
            # Text only
            return self.tokenizer.pad(batch, return_tensors="pt", padding=True)
        elif self.has_images and not self.has_text:
            # Images only
            return {"pixel_values": torch.stack([item["pixel_values"] for item in batch])}
        else:
            # Mixed or other cases
            result = {}
            if self.has_text:
                text_keys = ["input_ids", "attention_mask", "token_type_ids"]
                text_batch = [{k: item[k] for k in text_keys if k in item} for item in batch]
                result.update(self.tokenizer.pad(text_batch, return_tensors="pt", padding=True))

            if self.has_images:
                result["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])

            return result
