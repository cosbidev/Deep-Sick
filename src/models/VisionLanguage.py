from dataclasses import dataclass
from typing import Any
import PIL.Image
import torch
from typing import List, Dict

LANCZOS = PIL.Image.Resampling.LANCZOS
@dataclass
class VisionLanguageDataCollator:
    """Base collator for vision-language models"""

    def __init__(self, processor):
        self.processor = processor

    def get_tokenize_function(self):
        """
        Get the collate function for this collator
        Returns:
            A callable that collates a batch of data
        """
        return self.tokenize_function

    def tokenize_function(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data into a format suitable for the model
        Args:
            batch: List of dictionaries containing data samples
        Returns:
            A dictionary with collated tensors
        """
        raise NotImplementedError("Subclasses must implement collate_fn")
    def preprocess_images(self, images: List[PIL.Image.Image]) -> List[PIL.Image.Image]:
        """
        Preprocess a list of images
        Args:
            images: List of PIL Image objects
        Returns:
            Preprocessed list of images
        """
        raise NotImplementedError("Subclasses must implement preprocess_images")

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


