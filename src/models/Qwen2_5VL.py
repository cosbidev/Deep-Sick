
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoProcessor
from .VisionLanguage import VisionLanguageDataCollator, VisionLanguageModel
from qwen_vl_utils import process_vision_info
#from transformers import Qwen2_5VLForConditionalGeneration TODO find a solution the import of the model

from typing import Any, Dict, List




class Qwen25VLModel(VisionLanguageModel):
    """
    Base class for Qwen2.5-VL models
    Handles vision-language tasks with Qwen2.5-VL-specific processing
    """

    def __init__(self, model_name: str, collator: VisionLanguageDataCollator, tokenizer: str = None, **kwargs):
        """
        Initialize Qwen2.5-VL model with specific collator and tokenizer
        Args:
            model_name: Name of the model
            collator: Data collator for processing inputs
            tokenizer: Optional tokenizer for text processing
        """
        super().__init__(model_name, collator, tokenizer)
        self.tokenizer = tokenizer
        self.model = Qwen2_5VLForConditionalGeneration.from_pretrained(model_name, **kwargs)



class Qwen25VLCollator(VisionLanguageDataCollator):
    """
    Data collator for Qwen2.5-VL models
    Handles images, videos, and text with proper preprocessing
    """

    def __init__(
            self,
            dimension: str = "3b",
            min_pixels: Optional[int] = None,
            max_pixels: Optional[int] = None,
            max_length: int = 2048
            ):
        assert dimension in ["3b", "7b"]

        model_name_dict = {'3b' : 'Qwen/Qwen2.5-VL-3B-Instruct',
                           '7b' : 'Qwen/Qwen2.5-VL-7B-Instruct'}
        # Select model name based on dimension
        model_name = model_name_dict[dimension]


        # Initialize processor with optional pixel constraints
        if min_pixels and max_pixels:
            processor = AutoProcessor.from_pretrained(
                    model_name,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
            )
        else:
            processor = AutoProcessor.from_pretrained(model_name)

        super().__init__(processor, max_length=max_length)
        self.model_name = model_name


    @staticmethod
    def _format_messages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert example to Qwen2.5-VL message format"""
        messages = []

        if "messages" in example:
            return example["messages"]

        # Handle single example format
        content = []

        # Add images
        if "images" in example:
            images = example["images"] if isinstance(example["images"], list) else [example["images"]]
            for img in images:
                if isinstance(img, str):
                    # Handle different image formats
                    if img.startswith("data:image"):
                        content.append({"type": "image", "image": img})
                    elif img.startswith("http"):
                        content.append({"type": "image", "image": img})
                    else:
                        content.append({"type": "image", "image": f"file://{img}"})
                else:
                    # Handle PIL Image or numpy array
                    content.append({"type": "image", "image": img})

        # Add videos
        if "videos" in example:
            videos = example["videos"] if isinstance(example["videos"], list) else [example["videos"]]
            for video in videos:
                content.append({"type": "video", "video": video})

        # Add text
        if "text" in example:
            content.append({"type": "text", "text": example["text"]})
        elif "question" in example:
            content.append({"type": "text", "text": example["question"]})

        messages.append({"role": "user", "content": content})

        # Add assistant response if available
        if "answer" in example or "response" in example:
            assistant_text = example.get("answer", example.get("response", ""))
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

        return messages

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch for Qwen2.5-VL"""
        processed_batch = []

        for example in batch:
            messages = self._format_messages(example)

            # Apply chat template
            text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
            )

            # Process vision inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Process with processor
            inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True
            )

            processed_batch.append({
                    'input_ids'     : inputs.input_ids[0],
                    'attention_mask': inputs.attention_mask[0],
                    'pixel_values'  : inputs.get('pixel_values'),
                    'image_grid_thw': inputs.get('image_grid_thw'),
                    'video_grid_thw': inputs.get('video_grid_thw')
            })

        # Pad and stack tensors
        return self._collate_tensors(processed_batch)


    @staticmethod
    def _collate_tensors(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate tensors with proper padding"""
        collated = {}

        # Handle text inputs
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]

        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_masks):
            pad_len = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
            padded_attention_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))

        collated['input_ids'] = torch.stack(padded_input_ids)
        collated['attention_mask'] = torch.stack(padded_attention_masks)

        # Handle vision inputs
        pixel_values = [item.get('pixel_values') for item in batch if item.get('pixel_values') is not None]
        if pixel_values:
            collated['pixel_values'] = torch.cat(pixel_values, dim=0)

        # Handle grid information
        for key in ['image_grid_thw', 'video_grid_thw']:
            values = [item.get(key) for item in batch if item.get(key) is not None]
            if values:
                collated[key] = torch.cat(values, dim=0)

        return collated