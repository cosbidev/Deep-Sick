import torch
from dataclasses import dataclass
from typing import List, Dict, Any





@dataclass
class VisionLanguageDataCollator:
    """Base collator for vision-language models"""

    def __init__(self, processor, tokenizer=None, max_length: int = 2048):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __call__")

