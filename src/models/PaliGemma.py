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

from .VisionLanguage import VisionLanguageDataCollator



class PaliGemmaCollator(VisionLanguageDataCollator):
    """
    Data collator for PaliGemma models
    Handles image-text pairs with PaliGemma's specific formatting
    """

    def __init__(
            self,
            model_name: str = "google/paligemma-3b-pt-224",
            max_length: int = 512,
            image_size: int = 224
            ):

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


class LLaVACollator(VisionLanguageDataCollator):
    """
    Data collator for LLaVA models
    Handles conversation format with images
    """

    def __init__(
            self,
            model_name: str = "llava-hf/llava-1.5-7b-hf",
            max_length: int = 2048
            ):

        processor = AutoProcessor.from_pretrained(model_name)
        super().__init__(processor, max_length=max_length)
        self.model_name = model_name

    def _format_conversation(self, example: Dict[str, Any]) -> str:
        """Format conversation for LLaVA"""
        if "conversations" in example:
            # Handle conversation format
            conversation = ""
            for turn in example["conversations"]:
                role = turn.get("from", "user")
                value = turn.get("value", "")
                if role == "human":
                    conversation += f"USER: {value}\n"
                elif role == "gpt":
                    conversation += f"ASSISTANT: {value}\n"
            return conversation
        else:
            # Handle simple question-answer format
            question = example.get("question", example.get("text", ""))
            return f"USER: {question}\nASSISTANT:"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch for LLaVA"""
        texts = []
        images = []

        for example in batch:
            # Format conversation
            conversation = self._format_conversation(example)
            texts.append(conversation)

            # Handle image
            if "image" in example:
                img = example["image"]
                if isinstance(img, str):
                    img = Image.open(img)
                images.append(img)
            else:
                images.append(None)

        # Process with processor
        inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
        )

        return inputs

