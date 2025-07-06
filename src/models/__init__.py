from .util_models import count_trainable, est_flops, count_tokens, count_tokens_worker, est_vlm_flops
from .VisionLanguage import VisionLanguageDataCollator, UnslothVisionDataCollator
from .Qwen2_5VL import Qwen25VLCollator, Qwen25VLModel
from .PaliGemma import PaliGemmaCollator, PaliGemmaModel
from .LLaVA1_5 import LLaVACollator # TODO LLaVAModel
from .Gemma3 import *
from .formatters import *


# Factory function to model appropriate collator
def get_collator(model_family, **kwargs) -> VisionLanguageDataCollator:
    """
    Factory function to get the appropriate collator for a given model

    Args:
        model_name: Name of the model on Hugging Face
        **kwargs: Additional arguments for the collator

    Returns:
        Appropriate collator instance
    """
    model_name_lower = model_family.lower()
    if "qwen25vl" in model_name_lower or "qwen2_5_vl" in model_name_lower:
        print("|| Using Qwen2.5-VL collator ...")
        return Qwen25VLCollator(**kwargs)
    elif "paligemma" in model_name_lower:
        print("|| Using PaliGemma collator ...")
        return PaliGemmaCollator(**kwargs)
    elif "llava" in model_name_lower:
        return LLaVACollator(**kwargs)
    elif "instructblip" in model_name_lower:
        print("|| Using InstructBLIP collator ...")
        return InstructBLIPCollator(**kwargs)
    else:
        raise ValueError(f"No collator available for model: {model_family}")


def get_model(model_family: str, **kwargs):
    """
    Factory function to get the appropriate model based on family name

    Args:
        model_family: Name of the model family
        **kwargs: Additional arguments for model initialization

    Returns:
        Model instance
    """
    model_name_lower = model_family.lower()
    if "qwen25vl" in model_name_lower:
        print("|| Instancing Qwen25VL ...")
        return Qwen25VLModel(model_family, get_collator(model_family, **kwargs), **kwargs)
    elif "paligemma" in model_name_lower:
        print("|| Instancing PaliGemma ...")
        return PaliGemmaModel(model_family, get_collator(model_family, **kwargs), **kwargs)
    elif "llava" in model_name_lower:
        print("|| Instancing LLaVA ...")
        return LLaVAModel(model_family, get_collator(model_family, **kwargs), **kwargs)

