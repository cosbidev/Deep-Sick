import io
import os
from typing import Any, Dict, List

import PIL
import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor
from .VisionLanguage import VisionLanguageDataCollator
import threading

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GemmaCollator(VisionLanguageDataCollator):
    """
    Data collator ottimizzato per LLaVA models con cache e parallelizzazione
    """

    def __init__(
            self,
            model_id: str = "google/gemma-3-4b-it",
            max_length: int = 1500,
            enable_image_cache: bool = True,
            cache_size: int = 10000,
            use_fast_validation: bool = True,
            **kwargs: Any
    ):
        self.system = False
        self.max_l = 0
        self.max_length = max_length
        self.model_name = model_id
        self.enable_image_cache = enable_image_cache
        self.use_fast_validation = use_fast_validation

        # Cache per validazione immagini
        if enable_image_cache:
            self._path_cache = {}
            self._cache_lock = threading.Lock()
            self._cache_size = cache_size

        processor = AutoProcessor.from_pretrained(model_id,
                                                  use_fast=True,
                                                  max_length=max_length,
                                                  padding="max_length",
                                                  truncation=True,
                                                  **kwargs)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
        super().__init__(processor)

    @staticmethod
    def _format_conversation(example, max_images=2):
        """Mantiene la stessa logica originale"""
        chat = []
        assert max_images > 0
        for msg in example:
            role = msg["role"]
            content_obj = msg["content"]

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

    def _is_valid_image_fast(self, path):
        """Validazione ultra-veloce basata su header"""
        try:
            if not path:
                return False

            # Check cache first
            if self.enable_image_cache:
                with self._cache_lock:
                    if path in self._path_cache:
                        return self._path_cache[path]

            # Quick file existence and size check
            if not os.path.exists(path) or os.path.getsize(path) < 100:
                result = False
            else:
                # Check file header
                with open(path, 'rb') as f:
                    header = f.read(32)
                    result = (header.startswith(b'\xff\xd8\xff') or  # JPEG
                              header.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
                              header.startswith(b'GIF87a') or header.startswith(b'GIF89a') or  # GIF
                              header.startswith(b'WEBP', 8))  # WebP

            # Cache result
            if self.enable_image_cache:
                with self._cache_lock:
                    # Manage cache size
                    if len(self._path_cache) >= self._cache_size:
                        # Remove oldest 20% of entries
                        keys_to_remove = list(self._path_cache.keys())[:self._cache_size // 5]
                        for key in keys_to_remove:
                            del self._path_cache[key]

                    self._path_cache[path] = result

            return result

        except Exception:
            return False

    def _validate_image_pil(self, path):
        """Validazione completa con PIL (fallback)"""
        try:
            with Image.open(path) as img_obj:
                img_obj.verify()
            return True
        except Exception as e:
            print(f"[WARNING] Error validating image {path}: {e.with_traceback()}")

            return False

    def load_images_optimized(self, images, max_images=2, load_image_bytes=False):
        """
        Versione ottimizzata per HF datasets.map() con cache e validazione veloce
        """
        image_nested = []

        for image_group in images:
            group_items = []
            loaded = 0

            for img in image_group:
                if loaded >= max_images:
                    break

                path = img.get("path")
                if not path:
                    continue

                # Genera percorsi candidati
                candidate_paths = [path]

                # Trova primo percorso valido
                valid_path = None
                for candidate in candidate_paths:
                    if self.use_fast_validation:
                        # Prima validazione veloce
                        if self._is_valid_image_fast(candidate):
                            valid_path = candidate
                            break
                    else:
                        # Validazione completa PIL
                        if os.path.exists(candidate) and self._validate_image_pil(candidate):
                            valid_path = candidate

                            break

                if valid_path:
                    if load_image_bytes:
                        try:
                            with open(valid_path, "rb") as f:
                                group_items.append(f.read())
                        except Exception as e:
                            print(f"[WARNING] Error reading image bytes {valid_path}: {e}")
                            continue
                    else:
                        group_items.append(valid_path)
                    loaded += 1

            image_nested.append(group_items)

        return image_nested


    def tokenize_function(self, examples):
        """
        Versione ottimizzata per HF datasets.map()
        """
        # Carica immagini in modo ottimizzato
        images = examples.get('image')
        images = self.load_images_optimized(images)

        # Verifica che le immagini siano valide
        lenght_checker = [len(img_group) for img_group in images]
        # Use the lenght_checker to filter out empty image groups
        index_to_keep = [l>0 for i, l in enumerate(lenght_checker)]


        # Processa messaggi
        messages = examples.get('messages')
        b_size = len(messages)
        # Mantieni solo i messaggi con immagini valide
        images = [img_group for i, img_group in enumerate(images) if index_to_keep[i]]
        lenght_checker = [l for i, l in enumerate(lenght_checker) if index_to_keep[i]]
        messages = [m for i, m in enumerate(messages) if index_to_keep[i]]

        if not self.system:
            messages = [[[{'role': m[0]['role'][r + 1], 'content': m[0]['content'][r + 1]}
                          for r, ind in enumerate([r for r in m[0]['role'] if r != 'system'])]]
                        for m in messages]
        else:
            messages = [[[{'role': m[0]['role'][r], 'content': m[0]['content'][r]}
                          for r, ind in enumerate(m[0]['role'])]]
                        for m in messages]
        formatted_messages = [self._format_conversation(m[0], max_images=l) for m, l in zip(messages, lenght_checker)]

        # Applica chat template
        texts = self.processor.apply_chat_template(formatted_messages,
                                                   tokenize=False,
                                                   add_generation_prompt=False)

        # Calcola lunghezze (ottimizzato per batch)
        tokenized_lengths = []
        for text in texts:
            tokens = self.processor.tokenizer(text, add_special_tokens=False)['input_ids']
            tokenized_lengths.append(len(tokens))

        n_of_images = [len(img_group) for img_group in images]
        length_total = [length + n_imgs * self.processor.image_seq_length
                        for length, n_imgs in zip(tokenized_lengths, n_of_images)]

        if length_total and max(length_total) > self.max_l:
            self.max_l = max(length_total)

        if len(formatted_messages) != b_size:
            # Add None to maintain batch size
            if len(texts) < b_size:
                for _ in range(b_size - len(texts)):
                    texts.append(None)
            if len(images) < b_size:
                for _ in range(b_size - len(images)):
                    images.append(None)
            if len(formatted_messages) < b_size:
                for _ in range(b_size - len(formatted_messages)):
                    formatted_messages.append(None)
            if len(n_of_images) < b_size:
                for _ in range(b_size - len(n_of_images)):
                    n_of_images.append(None)
            if len(length_total) < b_size:
                for _ in range(b_size - len(length_total)):
                    length_total.append(None)


        return {
                "n_of_images"       : n_of_images,
                'length'            : length_total,
                'images'            : images,
                'texts'             : texts,
                'formatted_messages': formatted_messages
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch for LLaVA - mantiene la logica originale"""
        text_prompt = [el.get('texts', None) for el in batch]
        formatted_messages = [el.get('formatted_messages', None) for el in batch]

        images = process_vision_info(formatted_messages)

        # Tokenize the texts and process the images
        batch = self.processor(
                text=text_prompt,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
        )

        # Encode texts and images into tensors
        labels = batch["input_ids"].clone()
        image_token_id = [
                self.processor.tokenizer.convert_tokens_to_ids(
                        self.processor.tokenizer.special_tokens_map["boi_token"]
                )
        ]

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    def clear_cache(self):
        """Pulisce la cache delle immagini"""
        if self.enable_image_cache:
            with self._cache_lock:
                self._path_cache.clear()

    def get_cache_stats(self):
        """Restituisce statistiche sulla cache"""
        if self.enable_image_cache:
            with self._cache_lock:
                return {
                        "cache_size" : len(self._path_cache),
                        "cache_limit": self._cache_size
                }
        return {"cache_disabled": True}


# Mantieni la funzione originale per compatibilità
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    """Mantiene la logica originale per process_vision_info"""
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
                elif isinstance(image, str):
                    try:
                        if not os.path.exists(image):
                            img = Image.open(image.replace("train-512", "train"))
                        else:
                            img = Image.open(image)
                    except PIL.UnidentifiedImageError as e:
                        print(f"Failed to open image from path: {image}", e)
                        if os.path.exists(image.replace("train-512", "train")):
                            img = Image.open(image.replace("train-512", "train"))
                else:
                    print(f"Unsupported: ", image)
                    img = None

                if isinstance(img, Image.Image):
                    local_content_list.append(img.convert("RGB"))
        image_inputs.append(local_content_list)
    return image_inputs