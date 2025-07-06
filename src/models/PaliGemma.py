"""
Vision-Language Model Collators for Qwen2.5-VL, PaliGemma, and other VL models
Handles multi-modal inputs (text, images, videos) with proper preprocessing and batching
"""

import torch
from typing import List, Dict, Any
from transformers import AutoProcessor
import base64
from PIL import Image
import io

from .VisionLanguage import VisionLanguageDataCollator, VisionLanguageModel
from transformers import (
    AutoProcessor, PaliGemmaForConditionalGeneration
)





class PaliGemmaModel(VisionLanguageModel):
    """
    Base class for PaliGemma models
    Handles vision-language tasks with PaliGemma-specific processing
    """

    def __init__(self, model_name: str, collator: VisionLanguageDataCollator, tokenizer: str = None, **kwargs):
        """
        Initialize PaliGemma model with specific collator and tokenizer
        Args:
            model_name:
            collator:
            tokenizer:
            **kwargs:
        """
        super().__init__(model_name, collator, tokenizer)
        self.tokenizer = tokenizer
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, **kwargs)







class PaliGemmaCollator(VisionLanguageDataCollator):
    """
    Data collator for PaliGemma models
    Handles image-text pairs with PaliGemma's specific formatting
    """

    def __init__(
            self,
            dimension: str = "3b",  # Default to 3B model
            max_length: int = 512,
            image_size: int = 224
            ):

        model_name_dict = {
                        '3b': 'google/paligemma2-3b-pt-224',
                        '3b_mix': 'google/paligemma2-3b-mix-224'
        }
        model_name = model_name_dict[dimension]
        processor = AutoProcessor.from_pretrained(model_name)
        super().__init__(processor, max_length=max_length)
        self.model_name = model_name
        self.image_size = image_size

    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Format prompt for PaliGemma"""
        if "prompt" in example:
            return example["prompt"]
        elif "question" in example:
            return example["question"]
        elif "text" in example:
            return example["text"]
        else:
            return ""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch for PaliGemma"""
        texts = []
        images = []
        labels = []

        for example in batch:
            # Format text prompt
            prompt = self._format_prompt(example)
            texts.append(prompt)

            # Handle image
            if "image" in example:
                img = example["image"]
                if isinstance(img, str):
                    if img.startswith("data:image"):
                        # Decode base64 image
                        img_data = base64.b64decode(img.split(",")[1])
                        img = Image.open(io.BytesIO(img_data))
                    else:
                        img = Image.open(img)
                images.append(img)
            else:
                # Create dummy image if none provided
                images.append(Image.new('RGB', (self.image_size, self.image_size), color='white'))

            # Handle labels for training
            if "answer" in example or "label" in example:
                label_text = example.get("answer", example.get("label", ""))
                labels.append(label_text)

        # Process with processor
        if labels:
            # Training mode - include labels
            inputs = self.processor(
                    text=texts,
                    images=images,
                    suffix=labels,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
            )
        else:
            # Inference mode
            inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
            )

        return inputs
