from .VisionLanguage import VisionLanguageDataCollator
from .Qwen2_5VL import Qwen25VLCollator, Qwen25VLModel
from .Gemma3 import GemmaCollator
from .peft import DeepSpeedCompatibleModelParameterManager


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


# Esempio di utilizzo:
def  configure_model_for_training(
        model,
        strategy: str = "lora_gaussian",
        **kwargs
):
    """
    Funzione di convenienza per configurare un modello per il training.

    Args:
        model: Il modello da configurare
        strategy: Strategia di fine-tuning:
            - "lora_standard": LoRA con inizializzazione Kaiming-uniform
            - "lora_gaussian": LoRA con inizializzazione Gaussiana
            - "lora_pissa": LoRA con inizializzazione PiSSA
            - "lora_pissa_fast": LoRA con inizializzazione PiSSA veloce
            - "lora_olora": LoRA con inizializzazione OLoRA
            - "dora": DoRA (Weight-Decomposed LoRA)
            - "dora_rslora": DoRA con rank-stabilized LoRA
            - "qlora": QLoRA-style (all-linear)
            - "manual_adapter": Adapter manuale (sblocca layer specifici)
            - "modules_to_save": LoRA con moduli aggiuntivi da salvare
        **kwargs: Parametri aggiuntivi per la configurazione

    Returns:
        Modello configurato
    """
    manager = DeepSpeedCompatibleModelParameterManager(model, **kwargs)

    return manager.apply_lora_with_deepspeed_safety(model, **kwargs)



