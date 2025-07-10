import os
import torch
from datasets import load_dataset, DatasetDict, IterableDataset
import PIL
import torchvision.transforms as T

__all__ = [
    "load_parquet_image_dataset",
    "save_dataset_as_parquet",
    "image_preproc",
    "SimpleCollator",
    "get_text_column",
    "load_dataset",
]



def has_valid_pil_images(examples, log_bad=None):
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
    # if examples is None:
    #     return False
    # if examples['image'] is None:
    #     return False
    mask_list = []
    for imgs in examples['image']:


        for img in imgs:
            if img == None:
                mask_list.append(False)
        try:
            # no image, keep example (e.g. text-only)
            if not isinstance(imgs, list):
                imgs = [imgs]  # unify to list
            # Force full pixel decode for every image.
            for img in imgs:
                img.load()  # raises OSError on real corruption
        except Exception as e:
            if log_bad is not None:
                with open(log_bad, "a") as f:
                    f.write(f"{getattr(img, 'filename', '⟨in-memory⟩')} - {e}\n")
            mask_list.append(False)
        except PIL.UnidentifiedImageError:
            if log_bad is not None:
                with open(log_bad, "a") as f:
                    f.write(f"{getattr(img, 'filename', '⟨in-memory⟩')} - {e}\n")
            mask_list.append(False)
        except TypeError:
            print(' Type error')
            if log_bad is not None:
                with open(log_bad, "a") as f:
                    f.write(f"{getattr(img, 'filename', '⟨in-memory⟩')} - {e}\n")
            mask_list.append(False)
        mask_list.append(True)

    return mask_list

def load_parquet_image_dataset(dataset_dir: str, split_list: list = []) -> DatasetDict:
    """
    Loads a DatasetDict from a directory of split-based parquet files,
    and casts the 'image' column to Sequence[Image].
    """
    split_files = {}
    assert all([split in ["train", "val", "test"] for split in split_list])

    for split in split_list:
        for split_file in os.listdir(dataset_dir):
            if split in split_file and split_file.endswith(".parquet") :
                split_files[split_file.replace("data_", "").replace(".parquet", "")] = os.path.join(dataset_dir, split_file)

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



def get_text_column(dataset):
    """Find the text column in dataset"""
    text_candidates = ["text", "content", "sentence", "review", "comment", "document"]

    for col in text_candidates:
        if col in dataset.features:
            return col

    # Look for any string column
    for col_name, feature in dataset.features.items():
        if hasattr(feature, 'dtype') and feature.dtype == "string":
            return col_name

    raise ValueError("No text column found in dataset")


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
                result["pixel_values"] = torch.stack([torch.Tensor(item["pixel_values"]) for item in batch])

            return result

