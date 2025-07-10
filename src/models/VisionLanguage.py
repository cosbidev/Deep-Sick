
from dataclasses import dataclass
from typing import Any
import PIL.Image
LANCZOS = PIL.Image.Resampling.LANCZOS
import torch
import base64
from io import BytesIO
import math
import requests
from typing import Union, Tuple, List, Dict



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



@dataclass
class VisionLanguageDataCollator:
    """Base collator for vision-language models"""

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def get_collate_fn(self):
        """
        Get the collate function for this collator
        Returns:
            A callable that collates a batch of data
        """
        return self.collate_fn
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data into a format suitable for the model
        Args:
            batch: List of dictionaries containing data samples
        Returns:
            A dictionary with collated tensors
        """
        raise NotImplementedError("Subclasses must implement collate_fn")
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __call__")


@dataclass
class VisionLanguageModel(torch.nn.Module):
    """Base class for vision-language models"""
    def __init__(self, model_name: str, collator):
        self.model_name = model_name
        self._collator = collator

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the model
        Args:
            inputs: Processed inputs from the collator
        Returns:
            Model outputs
        """
        raise NotImplementedError("Subclasses must implement forward method")

    @property
    def get_collator(self):
        """
        Get the data collator for this model
        Returns:
            Data collator instance
        """
        return self._collator


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
