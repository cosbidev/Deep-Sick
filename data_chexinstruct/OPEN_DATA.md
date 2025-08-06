# Open Data Format

This dataset was processed and saved using the 🤗 Hugging Face `datasets` library.

## 📂 Location

The dataset is saved in Hugging Face format at:

```
data_chexinstruct/hf_cache
```

## 📥 How to Load

To load the dataset into your Python project:

```python
from datasets import load_from_disk

dataset = load_from_disk("data_chexinstruct/hf_cache")
print(dataset)
```

## 📄 Format Description

Each sample has the following fields:

- `instruction` *(string)*: User query or instruction
- `response` *(string)*: Expected model output
- `image` *(optional, list of images)*: One or more images (if available)

Some samples may be **text-only** (no image), others are **multimodal** (image + instruction → response).

## Example Sample

```python
sample = dataset["train"][0]
print(sample["instruction"])
print(sample["response"])

if "image" in sample and sample["image"]:
    sample["image"][0].show()
```
