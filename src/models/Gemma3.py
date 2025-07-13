import io

from .VisionLanguage import VisionLanguageDataCollator
from transformers import AutoProcessor
from typing import Any, Dict, List
import torch
from PIL import Image
import os


# For multi-image cases

def process_vision_info(messages: list[dict]) -> list[Image.Image]:

    image_inputs = []
    for msg in messages:
        content = msg[0].get("content", [])
        if not isinstance(content, list):
            content = [content]
        local_content_list = []
        for c in content:
            if isinstance(c, dict) and ("image" in c.values() or c.get("type") == "image"):
                if "image" in c.values():
                    image = c["image"]
                else:
                    image = c

                img = None
                if isinstance(image, Image.Image):
                    img = image
                elif isinstance(image, dict) and "bytes" in image:
                    try:
                        img = Image.open(io.BytesIO(image["bytes"]))
                    except Exception as e:
                        print(f"Failed to decode image from bytes: {e}")

                elif isinstance(image, str) and os.path.exists(image):
                    img = Image.open(image)
                if isinstance(img, Image.Image):
                    local_content_list.append(img.convert("RGB"))
        image_inputs.append(local_content_list)
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

    @staticmethod
    def _format_conversation(example):
        """
        Trasforma una struttura con 'text', 'image', 'image_path' (liste parallele per ciascun turno)
        nel formato:
        [
          {"role": "user", "content": [{"type": ..., "text": ..., "image": ...}, ...]},
          {"role": "assistant", "content": [{"type": ..., "text": ..., "image": ...}]}
        ]
        """
        chat = []
        for msg in example:
            role = msg["role"]
            content_obj = msg["content"]

            # Support both single-dict and list-of-dict formats
            if isinstance(content_obj, dict):
                contents = [content_obj]
            else:
                contents = content_obj

            all_chunks = []
            for c in contents:
                types = c.get("type", [])
                texts = c.get("text", [])
                images = c.get("image_path", [])

                for idx, ttype in enumerate(types):
                    if ttype == "text":
                        text_value = texts[idx] if idx < len(texts) else ""
                        if text_value and isinstance(text_value, str):
                            all_chunks.append({"type": "text", "text": text_value.strip()})
                    elif ttype == "image":
                        image_obj = images[idx] if idx < len(images) else None
                        if image_obj is not None:
                            all_chunks.append({"type": "image", "image": image_obj})

            if all_chunks:
                chat.append({"role": role, "content": all_chunks})

        return chat

    def tokenize_function(self, examples):
        """
        [{'role': ['system', 'user', 'assistant'],
        'content': [{'type': ['text'], 'text': ['As a radiology AI assistant, generate comprehensive findings incorporating the clinical indication. Tailor your radiological assessment to address the specific clinical question.'],
        'image': [None],
        'image_path': [None]},
         {'type': ['image', 'image', 'text'],
         'text': ['', '', 'You are provided with one or multiple chest X-ray image(s). Given the indication: "___F with new onset ascites // eval for infection", write a detailed findings section for the diagnostic report.'],
         'image': [None, None, None],
         'image_path': ['data/mimic-cxr/files_512/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg', 'data/mimic-cxr/files_512/p10/p10000032/s50414267/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg', None]},
         {'type': ['text'], 'text': ['There is no focal consolidation, pleural effusion or pneumothorax.  Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal.  Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.'],
         'image': [None], 'image_path': [None]}]}]
        Args:
            examples:

        Returns:

        """

        # Remove the system prompt:

        images = examples.get('image')
        messages = examples.get('messages')
        instructions = examples.get('instruction')
        response = examples.get('response')
        if not self.system:
            messages = [[[{'role': m[0]['role'][r + 1], 'content': m[0]['content'][r + 1]} for r, ind in enumerate([r for r in m[0]['role'] if r != 'system'])]] for m in messages]
        else:
            messages = [[[{'role': m[0]['role'][r], 'content': m[0]['content'][r]} for r, ind in enumerate(m[0]['role'])]] for m in messages]

        formatted_messages = [self._format_conversation(m[0]) for m in messages]

        texts = self.processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)

        return {
            'images': images,
            'texts': texts,
            'formatted_messages': formatted_messages
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


        """Process batch for LLaVA"""
        text_prompt = [el.get('texts', None) for el in batch]
        formatted_messages = [el.get('formatted_messages', None) for el in batch]

        images = process_vision_info(formatted_messages)

        # Tokenize the texts and process the images
        batch = self.processor(
                text=text_prompt, images=images, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors
        # Ensure the input_ids are padded correctly
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        # Mask image tokens
        image_token_id = [
                self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.special_tokens_map["boi_token"])
        ]

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100


        batch["labels"] = labels
        return batch  # Return the prepared batch




