import io
from typing import Any, Dict
import PIL
from PIL import Image
import gc
import os
from .system_prompt import get_random_prompt_for_task
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


# def data_format_gemma_conversation(example: dict[str, Any]) -> dict[str, Any]:
#     # Find the lenght of the image list
#     N_images = len(example.get("images", []))
#     content_list = []
#     for i in range(N_images):
#         content_list.append(
#                 {
#                         "type": "image",
#                 }
#         )
#     content_list.append(example['instruction'])
#
#
#     example["messages"] = [
#             {
#                     "role"   : "user",
#                     "content": content_list
#             },
#             {
#                     "role"   : "assistant",
#                     "content": [
#                             {
#                                     "type": "text",
#                                     "text": example['response'],
#                             },
#                     ],
#             },
#     ]
#     return example

def data_format_gemma_conversation_chexinstruct(samples: dict[str, any], single=True, load_path=True) -> dict[str, list]:
    formatted_samples = {"messages": []}

    system_prompt = get_random_prompt_for_task(samples['task'])


    if single:
        # If single is True, we assume samples is a single sample
        samples = {"instruction": [samples["instruction"]],
                   "response"   : [samples["response"]],
                   "image"      : [samples["image"]]}

    for cont in range(len(samples["instruction"])):
        images = []
        for imgs in samples["image"][cont]:
            try:
                # Just validate that the file exists and is accessible
                if os.path.exists(imgs):
                    if load_path:
                        images.append({
                                "type"      : "image",
                                "image_path": imgs,
                                "text"      : "",  # Empty but present for schema consistency
                                "image"     : None  # Will be ignored during processing
                        })
                    else:

                        # Original behavior - load the actual image
                        with open(imgs, "rb") as f:
                            img_bytes = f.read()
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        images.append({
                                "type"      : "image",
                                "image"     : image,
                                "text"      : "",  # Empty but present for schema consistency
                                "image_path": ""  # Empty but present for schema consistency
                        })
                        del image  # Free memory after processing the image
                        gc.collect()
                else:
                    print(f"<FileNotFound> Skipping image: {imgs} (file not found)")
                    continue

            except Exception as e:
                # Handle various exceptions that may arise from image processing
                print(f"<Exception> Skipping image due to error: {e}")
                continue
            except PIL.UnidentifiedImageError as e:
                print(f"<UnidentifiedImageError> Skipping image due to unidentified error: {e}")
                continue
            except TypeError as e:
                print(f"<TypeError> Skipping image due to type error: {e}")
                continue

        if len(images) == 0:
            return {"messages": []}  # If no valid images, return empty messages

        formatted_samples["messages"].append([
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": images + [{"type": "text", "text": samples["instruction"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["response"][cont]}]},
        ])

    return formatted_samples




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
