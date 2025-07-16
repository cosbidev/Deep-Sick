import base64
import math
import os
from io import BytesIO
from typing import Union, Dict, Tuple, List

import torch
from datasets import load_dataset, DatasetDict
import PIL
import torchvision.transforms as T
import requests

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


__all__ = [
    "load_parquet_image_dataset",
    "save_dataset_as_parquet",
    "image_preproc",
    "SimpleCollator",
    "get_text_column",
    "load_dataset",
]


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


def load_parquet_image_dataset(dataset_dir: str, split_list: list = [], **kwargs) -> DatasetDict:
    """
    Loads a DatasetDict from a directory of split-based parquet files,
    and casts the 'image' column to Sequence[Image].
    """
    split_files = {}
    assert all([split in ["train", "val", "test"] for split in split_list])

    for split in split_list:
        for split_file in os.listdir(dataset_dir):
            if split in split_file and split_file.endswith(".parquet") :
                split_files[split_file.replace("data_", "").replace(".parquet", "") if 'tokenized' not in split_file else
                split_file.replace("data_", "").replace(".parquet", "").replace('tokenized_', '')] = os.path.join(dataset_dir, split_file)


    dataset_dict = load_dataset("parquet", data_files=split_files, **kwargs)
    return dataset_dict


def save_dataset_as_parquet(dataset_dict, output_dir, name_file):
    """
    Save a DatasetDict to Parquet format without copying image files.

    Args:
        dataset_dict (DatasetDict): Hugging Face dataset with an "image" column (list or str).
        output_dir (str): Destination directory where Parquet files will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)


    for split in dataset_dict:
        parquet_path = os.path.join(output_dir, f"{name_file}_data_{split}.parquet")
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



def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(
    ele: Dict,
    size_factor: int = IMAGE_FACTOR,
) -> PIL.Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
        if isinstance(image, dict) and "url" in image:
            image = image["url"]
    pass
    image_obj = None
    if isinstance(image, PIL.Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = PIL.Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = PIL.Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = PIL.Image.open(BytesIO(data))
    else:
        image_obj = PIL.Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image



def extract_vision_info(conversations: Union[List[Dict], List[List[Dict]]]) -> List[Dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: Union[List[Dict], List[List[Dict]]],
) -> Tuple[Union[List[PIL.Image.Image], None], Union[List[Union[torch.Tensor, List[PIL.Image.Image]]], None]]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        # elif "video" in vision_info:
        #     video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs


def get_padding_tokens_ids(tokenizer):
    global IMAGE_TOKENS

    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    image_tokens = IMAGE_TOKENS
    if hasattr(tokenizer, "image_token"):
        image_tokens = IMAGE_TOKENS + [tokenizer.image_token]
    pass

    padding_token_ids = tokenizer.convert_tokens_to_ids(image_tokens)
    if hasattr(tokenizer, "pad_token_id"):
        padding_token_ids.append(tokenizer.pad_token_id)
    pass

    padding_token_ids = list(x for x in padding_token_ids if x is not None)
    padding_token_ids = list(set(padding_token_ids))
    padding_token_ids = torch.IntTensor(padding_token_ids)
    return padding_token_ids



def _get_dtype(dtype):
    __DTYPE_MAP = {
        "float32": torch.float32,
        torch.float32: torch.float32,
        "float16": torch.float16,
        torch.float16: torch.float16,
        "bfloat16": torch.bfloat16,
        torch.bfloat16: torch.bfloat16,
    }
    if   dtype is None or dtype == None: return None
    elif dtype in __DTYPE_MAP: return __DTYPE_MAP[dtype]
    else:
        print(f"Unsloth: {dtype} is not recognized, so we'll default to None")
        return None




