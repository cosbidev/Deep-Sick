from .VisionLanguage import VisionLanguageDataCollator
from .Qwen2_5VL import Qwen25VLCollator, Qwen25VLModel
from .Gemma3 import GemmaCollator, GemmaInference
from .sft_trainer import GemmaSFTTrainer
from .peft import DeepSpeedCompatibleModelParameterManager
import torch
from transformers import AutoModelForCausalLM, AutoConfig

def setup_model_and_config(model_args, args, data_args):
    """Setup model configuration and load the model."""
    assert model_args.model_name_or_path, "You need to specify a model name or path"

    # Attention implementation check
    if args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    # Configuration overrides
    customized_kwargs = dict()
    overwrite_config = {}
    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if overwrite_config:
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    # Load model
    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=data_args.cache_dir,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
            **customized_kwargs
    )
    # TEST
    args.layer_to_unfreeze = ["lm_head", "multi_modal_projector"]
    print("✅ Base model loaded successfully")

    # CRITICAL: Configure PEFT/LoRA BEFORE any other operations
    if args.lora_enable:
        print("🔄 Applying LoRA configuration...")

        model = configure_model_for_training(
                model,
                strategy=args.peft_strategy,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                layer_to_unfreeze=args.layer_to_unfreeze,
                finetune_vision_layers=args.finetune_vision_layers,
                finetune_language_layers=args.finetune_language_layers,
                finetune_attention_modules=args.finetune_attention_modules,
                finetune_mlp_modules=args.finetune_mlp_modules,
        )

        # Verify that we have trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("❌ No trainable parameters found after LoRA configuration!")

        print(f"✅ LoRA applied successfully with {len(trainable_params)} trainable parameter groups")

    # # Disable gradient checkpointing to avoid conflicts with DeepSpeed
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_disable()

    # Set hidden size for compatibility
    try:
        model.config.hidden_size = model.model.language_model.embed_tokens.embedding_dim
    except:
        model.config.hidden_size = 2560

    return model




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
        **kwargs
):
    """
    Funzione di convenienza per configurare un modello per il training.

    Args:
        model: Il modello da configurare
        strategy: Strategia di fine-tuning:
            - 'lora': Applica LoRA al modello
            - 'full': Addestra tutti i parametri del modello
            - 'freeze': Congela tutti i parametri del modello
        **kwargs: Parametri aggiuntivi per la configurazione

    Returns:
        Modello configurato
    """
    manager = DeepSpeedCompatibleModelParameterManager(model, **kwargs)

    return manager.apply_lora_with_deepspeed_safety(model, **kwargs)



