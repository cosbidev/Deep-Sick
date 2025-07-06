import os
import random
from typing import Any, Dict


def data_format_qwen25vl_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format data for Qwen2.5-VL conversation format.
    Qwen2.5-VL uses a messages format similar to OpenAI but with specific image handling.
    """
    try:
        example["messages"] = [
                {
                        "role"   : "user",
                        "content": [
                                {
                                        "type" : "image",
                                        "image": example["image"]
                                },
                                {
                                        "type": "text",
                                        "text": example['instruction']
                                }
                        ]
                },
                {
                        "role"   : "assistant",
                        "content": [
                                {
                                        "type": "text",
                                        "text": example['response']
                                }
                        ]
                }
        ]
        return example


    except Exception as e:
        print(f"❌ Skipping image due to error: {e}")
        return None  # will be dropped if `remove_columns` or `filter` is used


def data_format_qwen25vl_conversation_alt(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Alternative Qwen2.5-VL format that uses image_url structure.
    Some Qwen2.5-VL implementations expect this format.
    """
    example["messages"] = [
            {
                    "role"   : "user",
                    "content": [
                            {
                                    "type"     : "image_url",
                                    "image_url": {
                                            "url": example["image"]  # Assumes image is a URL or base64 string
                                    }
                            },
                            {
                                    "type": "text",
                                    "text": example['instruction']
                            }
                    ]
            },
            {
                    "role"   : "assistant",
                    "content": example['response']  # Simple string format for assistant response
            }
    ]
    return example



def data_format_gemma_conversation(example: dict[str, Any]) -> dict[str, Any]:
    # Find the lenght of the image list
    N_images = len(example.get("images", []))
    content_list = []
    for i in range(N_images):
        content_list.append(
                {
                        "type": "image",
                }
        )
    content_list.append(example['instruction'])


    example["messages"] = [
            {
                    "role"   : "user",
                    "content": content_list
            },
            {
                    "role"   : "assistant",
                    "content": [
                            {
                                    "type": "text",
                                    "text": example['response'],
                            },
                    ],
            },
    ]
    return example




def data_format_llava15_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format data for LLaVA 1.5 conversation format.
    LLaVA 1.5 uses a specific conversation structure with 'human' and 'gpt' roles.
    """
    example["conversations"] = [
            {
                    "from" : "human",
                    "value": f"<image>\n{example['instruction']}"
            },
            {
                    "from" : "gpt",
                    "value": example['response']
            }
    ]
    # Keep the image field as LLaVA expects it separately
    # example["image"] should already exist in the input
    return example

