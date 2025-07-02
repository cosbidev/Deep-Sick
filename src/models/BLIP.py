
from typing import List, Dict, Any
import torch
from transformers import AutoProcessor
from PIL import Image
from .VisionLanguage import VisionLanguageDataCollator



class InstructBLIPCollator(VisionLanguageDataCollator):
    """
    Data collator for InstructBLIP models
    Handles instruction-following with images
    """

    def __init__(
            self,
            model_name: str = "Salesforce/instructblip-vicuna-7b",
            max_length: int = 512
            ):

        processor = AutoProcessor.from_pretrained(model_name)

        super().__init__(processor, max_length=max_length)
        self.model_name = model_name

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch for InstructBLIP"""
        prompts = []
        images = []

        for example in batch:
            # Format prompt
            prompt = example.get("prompt", example.get("question", example.get("text", "")))
            prompts.append(prompt)

            # Handle image
            if "image" in example:
                img = example["image"]
                if isinstance(img, str):
                    img = Image.open(img)
                images.append(img)
            else:
                images.append(Image.new('RGB', (224, 224), color='white'))

        # Process with processor
        inputs = self.processor(
                images=images,
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
        )

        return inputs


