from .util_models import count_trainable, est_flops, count_tokens, count_tokens_worker, est_vlm_flops
from .VisionLanguage import VisionLanguageDataCollator
from .Qwen2_5VL import Qwen25VLCollator
from .PaliGemma import PaliGemmaCollator
from .BLIP import InstructBLIPCollator


# Factory function to get appropriate collator
def get_collator(model_str: str, **kwargs) -> VisionLanguageDataCollator:
    """
    Factory function to get the appropriate collator for a given model

    Args:
        model_name: Name of the model on Hugging Face
        **kwargs: Additional arguments for the collator

    Returns:
        Appropriate collator instance
    """
    model_name_lower = model_str.lower()

    if "qwen2.5-vl" in model_name_lower or "qwen2_5_vl" in model_name_lower:
        return Qwen25VLCollator(**kwargs)
    elif "paligemma" in model_name_lower:
        return PaliGemmaCollator(**kwargs)
    elif "llava" in model_name_lower:
        return LLaVACollator(**kwargs)
    elif "instructblip" in model_name_lower:
        return InstructBLIPCollator(**kwargs)
    else:
        raise ValueError(f"No collator available for model: {model_str}")

