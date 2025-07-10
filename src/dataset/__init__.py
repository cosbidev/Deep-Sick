from .util_data import load_parquet_image_dataset, save_dataset_as_parquet, image_preproc, SimpleCollator, train_on_responses_only
from .formatters import data_format_llava15_conversation, data_format_qwen25vl_conversation, data_format_gemma_conversation_chexinstruct
from .system_prompt import get_random_prompt_for_task, get_prompt_for_task



def get_dataset_by_folder(dataset_name: str) -> callable:
    """
    Returns a dataset callable function by the.
    """
    if "chexinstruct" in dataset_name.lower():
        return load_parquet_image_dataset
    else:
        raise NotImplementedError(dataset_name)

