from .util_models import count_trainable, est_flops, count_tokens, count_tokens_worker, est_vlm_flops
from .VisionLanguage import VisionLanguageDataCollator, UnslothVisionDataCollator
from .Qwen2_5VL import Qwen25VLCollator, Qwen25VLModel
from .Gemma3 import GemmaCollator

# Factory function to model appropriate collator
def get_collator(model_id, **kwargs) -> VisionLanguageDataCollator:
    """
    Factory function to get the appropriate collator for a given model

    Args:
        model_name: Name of the model on Hugging Face
        **kwargs: Additional arguments for the collator

    Returns:
        Appropriate collator instance
    """
    model_name_lower = model_id.lower()
    if "qwen25vl" in model_name_lower or "qwen2_5_vl" in model_name_lower:
        print("|| Using Qwen2.5-VL collator ...")
        return Qwen25VLCollator(**kwargs)
    elif "gemma" in model_name_lower:
        print("|| Using Gemma collator ...")
        return GemmaCollator(**kwargs)
    elif "unsloth" in model_name_lower:
        return UnslothVisionDataCollator(**kwargs)
    else:
        raise ValueError(f"No collator available for model: {model_id}")


def get_model(model_id: str, **kwargs):
    """
    Factory function to get the appropriate model based on family name

    Args:
        model_id: Name of the model family
        **kwargs: Additional arguments for model initialization

    Returns:
        Model instance
    """
    model_name_lower = model_id.lower()
    if "qwen25vl" in model_name_lower:
        print("|| Instancing Qwen25VL ...")
        return Qwen25VLModel(model_id, get_collator(model_id, **kwargs), **kwargs)
