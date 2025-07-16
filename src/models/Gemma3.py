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
            max_length: int = 1500,
            **kwargs: Any
    ):
        # Initialize the Gemma collator with model ID and max length
        self.system = False
        self.max_l = 0
        self.max_length = max_length
        self.model_name = model_id
        processor = AutoProcessor.from_pretrained(model_id,
                                                  use_fast=True,
                                                  max_length=max_length,
                                                  padding="max_length",  # ← ensures uniform input shape
                                                  truncation=True,  # ← avoids overflow
                                                  **kwargs)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Use eos token as pad token
        processor.tokenizer.padding_side = "right"
        super().__init__(processor)

    @staticmethod
    def _format_conversation(example, max_images=2):
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
                        if idx + 1 > max_images:
                            continue
                        image_obj = images[idx] if idx < len(images) else None
                        if image_obj is not None:
                            all_chunks.append({"type": "image", "image": image_obj})

            if all_chunks:
                chat.append({"role": role, "content": all_chunks})

        return chat


    def load_images(self, images, max_images=2, load_image_bytes=False):
        """
        Load image bytes from a nested list of image dicts with 'path' keys.

        Args:
            images (List[List[Dict]]): Each inner list contains dicts like {'path': 'path/to/image.jpg'}

        Returns:
            List[List[bytes]]: Nested list of image bytes, up to `max_images` per sample.
        """
        image_nested = []
        for image_group in images:
            group_bytes = []
            count = 0
            for img in image_group:
                if count >= max_images:
                    break
                path = img.get("path")
                if path and os.path.exists(path):
                    try:
                        if load_image_bytes:
                            with open(path, "rb") as f:
                                group_bytes.append(f.read())
                                count += 1
                        else:
                            # Return the paths instead of bytes
                            group_bytes.append(path)
                            count += 1
                    except Exception as e:
                        print(f"[WARNING] Could not load image {path}: {e}")
                else:
                    print(f"[WARNING] Invalid or missing path: {path}")
            image_nested.append(group_bytes)
        return image_nested

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
        images = self.load_images(images)
        messages = examples.get('messages')
        if not self.system:
            messages = [[[{'role': m[0]['role'][r + 1], 'content': m[0]['content'][r + 1]} for r, ind in enumerate([r for r in m[0]['role'] if r != 'system'])]] for m in messages]
        else:
            messages = [[[{'role': m[0]['role'][r], 'content': m[0]['content'][r]} for r, ind in enumerate(m[0]['role'])]] for m in messages]

        formatted_messages = [self._format_conversation(m[0]) for m in messages]

        texts = self.processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
        # Tokenize the texts
        length = [self.processor.tokenizer(m)['input_ids'].__len__() for m in texts]
        n_of_images = [im.__len__() for im in images]
        length_total = [l + len(m) * self.processor.image_seq_length for l, m in zip(length, images)]

        if max(length_total) > self.max_l:
            self.max_l = max(length_total)



        return {
                "n_of_images"      : n_of_images,
                'length'            : length_total,
                'images'            : images,
                'texts'             : texts,
                'formatted_messages': formatted_messages
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        """Process batch for LLaVA"""
        text_prompt = [el.get('texts', None) for el in batch]
        formatted_messages = [el.get('formatted_messages', None) for el in batch]

        images = process_vision_info(formatted_messages)

        # Tokenize the texts and process the images
        batch = self.processor(

                    text=text_prompt,
                    images=images,
                    return_tensors="pt",
                    padding="max_length",  # ← ensures fixed-length
                    truncation=True,
                    max_length=self.max_length  # ← defined in __init__, e.g. 1024 or 2048

        )




        # Encode texts and images into tensors
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
