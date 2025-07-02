import sys
sys.path.append("./")
from src.models import get_collator



if __name__ == "__main__":
    # Example with Qwen2.5-VL

    qwen_collator = get_collator(model_str="qwen2.5-vl",
            model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
    )


    # Example batch
    batch = [
            {
                    "images": ["/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/brax/images-512/id_0001830a-25d11913-d83bb0a2-187d0d05-34c39b4d/Study_20239083.70884272.94302810.12925298.98200873/Series_15212202.97798035.45695801.88765464.66294995/image-46511799-03167093-99478710-19287831-78140979.png"],
                    "text"  : "Describe this image in detail.",
                    "answer": "The CXR shows a normal chest with no signs of pneumonia or other abnormalities."
            },
            {
                    "messages": [
                            {
                                    "role"   : "user",
                                    "content": [
                                            {"type": "image", "image": "/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/brax/images-512/id_0001830a-25d11913-d83bb0a2-187d0d05-34c39b4d/Study_20239083.70884272.94302810.12925298.98200873/Series_15212202.97798035.45695801.88765464.66294995/image-46511799-03167093-99478710-19287831-78140979.png"},
                                            {"type": "text", "text": "What do you see?"}
                                    ]
                            }
                    ]
            }
    ]

    # Process batch
    processed = qwen_collator(batch)
    print("Processed batch keys:", processed.keys())

    # Example with PaliGemma
    paligemma_collator = get_collator(model_str="paligemma",
            model_name="google/paligemma-3b-pt-224"
    )

    pali_batch = [
            {
                    "image" : "path/to/image.jpg",
                    "prompt": "describe en",
                    "answer": "A photo of a cat"
            }
    ]

    pali_processed = paligemma_collator(pali_batch)
    print("PaliGemma processed batch keys:", pali_processed.keys())