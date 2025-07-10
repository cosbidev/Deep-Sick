import io

from .VisionLanguage import VisionLanguageDataCollator
from transformers import AutoProcessor
from typing import Any, Dict, List
import torch
from PIL import Image

# For multi-image cases
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


class GemmaCollator(VisionLanguageDataCollator):
    """
    Data collator for LLaVA models
    Handles conversation format with images
    """

    def __init__(
            self,
            model_id: str = "google/gemma-3-4b-it",
            max_length: int = 2048,
            **kwargs: Any
    ):
        # Initialize the Gemma collator with model ID and max length
        self.system = False

        self.model_name = model_id
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Use eos token as pad token
        processor.tokenizer.padding_side = "right"
        super().__init__(processor, max_length=max_length)


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

    def collate_fn(self, examples):
        # Remove the system prompt:
        if not self.system:

            examples = [example for example in examples if "system" not in example["messages"][0].get("role", "")]

        texts = [self.processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).strip() for example in examples]
        if "image" in examples[0]:  # single-image
            images = [
                    [img.convert("RGB") for img in example["image"]]
                    for example in examples
            ]
        else:  # multi-image
            images = [process_vision_info(example["messages"]) for example in examples]

        # Tokenize the texts and process the images
        batch = self.processor(
                text=texts, images=images, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        # Mask image tokens
        image_token_id = [
                self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.special_tokens_map["boi_token"])
        ]
        # Mask tokens for not being used in the loss computation
        # Remove The part related to the Question and System prompt

        #labels[labels == self.processor.tokenizer.eos_token_id] = -100



        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch  # Return the prepared batch


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



