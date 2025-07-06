from .VisionLanguage import VisionLanguageDataCollator
from transformers import AutoProcessor
import torch
from typing import Dict, List, Any
from PIL import Image


class LLaVACollator(VisionLanguageDataCollator):
    """
    Data collator for LLaVA models
    Handles conversation format with images
    """

    def __init__(
            self,
            dimension: str = '3b',
            max_length: int = 2048
    ):

        model_name_dict = { '3b ': 'llava-hf/llava-1.5-3b-hf',
                            '7b ': 'llava-hf/llava-1.5-7b-hf',
                            '13b': 'llava-hf/llava-1.5-13b-hf' }
        model_name = model_name_dict[dimension]

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

