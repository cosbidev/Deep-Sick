import re
import torch
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Optional
from collections import Counter


def get_peft_regex(
        model,
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        target_modules: List[str] = None,
        vision_tags: Optional[List[str]] = None,
        language_tags: Optional[List[str]] = None,
        attention_tags: Optional[List[str]] = None,
        mlp_tags: Optional[List[str]] = None
) -> str:
    """
    Create a regex pattern to apply LoRA to only select layers of a model.
    This version is optimized for DeepSpeed compatibility.
    """

    if vision_tags is None:
        vision_tags = ["vision", "image", "visual", "patch"]
    if language_tags is None:
        language_tags = ["language", "text"]
    if attention_tags is None:
        attention_tags = ["self_attn", "attention", "attn"]
    if mlp_tags is None:
        mlp_tags = ["mlp", "feed_forward", "ffn", "dense"]

    if not finetune_vision_layers and not finetune_language_layers:
        raise RuntimeError(
                "No layers to finetune - please select to finetune the vision and/or the language layers!"
        )
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError(
                "No modules to finetune - please select to finetune the attention and/or the mlp modules!"
        )

    # Get only linear layers
    modules = model.named_modules()
    linear_modules = [name for name, module in modules if isinstance(module, torch.nn.Linear)]
    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)

    # Isolate lm_head / projection matrices if count == 1
    if target_modules is None:
        only_linear_modules = []
        projection_modules = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
    else:
        assert (type(target_modules) is list)
        only_linear_modules = list(target_modules)

    # Create regex matcher
    regex_model_parts = []
    if finetune_vision_layers:
        regex_model_parts += vision_tags
    if finetune_language_layers:
        regex_model_parts += language_tags
    regex_components = []
    if finetune_attention_modules:
        regex_components += attention_tags
    if finetune_mlp_modules:
        regex_components += mlp_tags

    regex_model_parts = "|".join(regex_model_parts)
    regex_components = "|".join(regex_components)

    match_linear_modules = r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"
    regex_matcher = \
        r".*?(?:" + regex_model_parts + \
        r").*?(?:" + regex_components + \
        r").*?" + match_linear_modules + ".*?"

    # Also account for model.layers.0.self_attn/mlp type modules like Qwen
    if finetune_language_layers:
        regex_matcher = r"(?:" + regex_matcher + \
                        r")|(?:\bmodel\.layers\.[\d]{1,}\.(?:" + regex_components + \
                        r")\.(?:" + match_linear_modules + r"))"

    # Check if regex is wrong since model does not have vision parts
    check = any(re.search(regex_matcher, name, flags=re.DOTALL) for name in linear_modules)
    if not check:
        regex_matcher = \
            r".*?(?:" + regex_components + \
            r").*?" + match_linear_modules + ".*?"

    # Final check to confirm if matches exist
    check = any(re.search(regex_matcher, name, flags=re.DOTALL) for name in linear_modules)
    if not check and target_modules is not None:
        raise RuntimeError(
                f"No layers to finetune? You most likely specified target_modules = {target_modules} incorrectly!"
        )
    elif not check:
        raise RuntimeError(
                f"No layers to finetune for {model.config._name_or_path}. Please file a bug report!"
        )

    return regex_matcher


class DeepSpeedCompatibleModelParameterManager:
    """
    Enhanced parameter manager specifically designed for DeepSpeed Zero-2 compatibility.
    This version addresses common issues with empty tensor lists and parameter initialization.
    """


    def __init__(
            self,
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            **kwargs
            ):
        assert model is not None, "Model cannot be None"

        # Get target modules using regex pattern
        try:
            self.target_modules = get_peft_regex(
                    model,
                    finetune_vision_layers=finetune_vision_layers,
                    finetune_language_layers=finetune_language_layers,
                    finetune_attention_modules=finetune_attention_modules,
                    finetune_mlp_modules=finetune_mlp_modules,
            )
            print(f"✅ Target modules pattern: {self.target_modules}")
        except Exception as e:
            print(f"⚠️ Regex pattern failed, falling back to common modules: {e}")
            # Fallback to common target modules
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def return_target_modules(self):
        """Return the target modules for LoRA."""
        return self.target_modules

    @staticmethod
    def freeze_all_parameters(model):
        """Freeze all model parameters."""
        frozen_count = 0
        for param in model.parameters():
            param.requires_grad = False
            frozen_count += 1
        print(f"🧊 Froze {frozen_count} parameters")

    @staticmethod
    def verify_trainable_parameters(model):
        """Verify that model has trainable parameters and no empty tensors."""
        trainable_params = []
        empty_tensors = []
        total_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()

            if param.requires_grad:
                trainable_params.append((name, param))

                # Check for empty tensors
                if hasattr(param, 'shape') and any(dim == 0 for dim in param.shape):
                    empty_tensors.append((name, param.shape))
                    print(f"❌ EMPTY TENSOR DETECTED: {name} with shape {param.shape}")

        trainable_count = sum(p.numel() for _, p in trainable_params)

        print(f"📊 Parameter verification:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_count:,}")
        print(f"   Trainable percentage: {100 * trainable_count / total_params:.2f}%")
        print(f"   Empty tensors found: {len(empty_tensors)}")

        if len(empty_tensors) > 0:
            raise RuntimeError(f"❌ Found {len(empty_tensors)} empty tensors that will cause DeepSpeed errors!")

        if len(trainable_params) == 0:
            raise RuntimeError("❌ No trainable parameters found!")

        return trainable_params

    def apply_lora_with_deepspeed_safety(
            self,
            model,
            r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            bias: str = "none",
            task_type: TaskType = TaskType.CAUSAL_LM,
            init_method: str = "gaussian",
            modules_to_save: List[str] = None,
            **kwargs
    ):
        """
        Apply LoRA with DeepSpeed Zero-2 safety checks and optimizations.
        """
        print("🔄 Applying LoRA with DeepSpeed safety checks...")

        # Step 1: Freeze all parameters
        self.freeze_all_parameters(model)

        # Step 2: Get target modules
        target_modules = self.return_target_modules()
        print(f"🎯 Target modules: {target_modules}")

        # Step 3: Create LoRA config with DeepSpeed-compatible settings
        peft_config = LoraConfig(
                task_type=task_type,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                target_modules=target_modules,
                init_lora_weights=init_method,
                modules_to_save=modules_to_save,
                # DeepSpeed-specific optimizations
                use_rslora=False,  # Disable for better compatibility
                use_dora=False,  # Disable for better compatibility
        )

        # Step 4: Apply PEFT
        try:
            model = get_peft_model(model, peft_config)
            print("✅ LoRA applied successfully")
        except Exception as e:
            print(f"❌ Error applying LoRA: {e}")
            raise

        # Step 5: Verify the model state
        trainable_params = self.verify_trainable_parameters(model)

        # Step 6: Print trainable parameters summary
        if hasattr(model, 'print_trainable_parameters'):
            print("📋 PEFT Summary:")
            model.print_trainable_parameters()

        print(f"✅ LoRA configuration completed with {len(trainable_params)} trainable parameter groups")
        return model

    def apply_deepspeed_optimized_strategies(self, model, strategy: str, **kwargs):
        """Apply different LoRA strategies optimized for DeepSpeed."""

        strategy_configs = {
                "lora_gaussian"    : {
                        "init_method" : "gaussian",
                        "r"           : kwargs.get("r", 16),
                        "lora_alpha"  : kwargs.get("lora_alpha", 32),
                        "lora_dropout": kwargs.get("lora_dropout", 0.05)
                },
                "lora_standard"    : {
                        "init_method" : True,  # Kaiming uniform
                        "r"           : kwargs.get("r", 16),
                        "lora_alpha"  : kwargs.get("lora_alpha", 32),
                        "lora_dropout": kwargs.get("lora_dropout", 0.05)
                },
                "lora_pissa"       : {
                        "init_method" : "pissa",
                        "r"           : kwargs.get("r", 8),  # Lower rank for PiSSA
                        "lora_alpha"  : kwargs.get("lora_alpha", 16),
                        "lora_dropout": kwargs.get("lora_dropout", 0.05)
                },
                "lora_conservative": {
                        "init_method" : "gaussian",
                        "r"           : kwargs.get("r", 8),  # Very conservative
                        "lora_alpha"  : kwargs.get("lora_alpha", 16),
                        "lora_dropout": kwargs.get("lora_dropout", 0.1)
                }
        }

        if strategy not in strategy_configs:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategy_configs.keys())}")

        config = strategy_configs[strategy]
        config.update(kwargs)  # Override with user parameters

        print(f"🚀 Applying strategy: {strategy}")
        print(f"📋 Configuration: {config}")

        return self.apply_lora_with_deepspeed_safety(model, **config)


def configure_model_for_training(
        model,
        strategy: str = "lora_gaussian",
        **kwargs
):
    """
    DeepSpeed-compatible model configuration function.

    Available strategies:
    - "lora_gaussian": LoRA with Gaussian initialization (recommended for DeepSpeed)
    - "lora_standard": LoRA with Kaiming uniform initialization
    - "lora_pissa": LoRA with PiSSA initialization (use with caution with DeepSpeed)
    - "lora_conservative": Very conservative LoRA settings for maximum compatibility
    """

    print(f"🔧 Configuring model with strategy: {strategy}")

    # Create manager with model
    manager = DeepSpeedCompatibleModelParameterManager(
            model,
            finetune_vision_layers=kwargs.get("finetune_vision_layers", False),
            finetune_language_layers=kwargs.get("finetune_language_layers", True),
            finetune_attention_modules=kwargs.get("finetune_attention_modules", True),
            finetune_mlp_modules=kwargs.get("finetune_mlp_modules", True),
    )

    # Apply the strategy
    try:
        configured_model = manager.apply_deepspeed_optimized_strategies(model, strategy, **kwargs)
        print("✅ Model configuration completed successfully")
        return configured_model
    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        print("💡 Try using 'lora_conservative' strategy for maximum compatibility")
        raise


# Test function for debugging
def test_model_configuration():
    """Test function to verify the configuration works."""
    try:
        from transformers import AutoModelForCausalLM

        print("🧪 Testing model configuration...")

        # Load a small model for testing
        model_name = "microsoft/DialoGPT-small"
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print(f"📥 Loaded model: {model_name}")

        # Test conservative configuration
        configured_model = configure_model_for_training(
                model,
                strategy="lora_conservative",
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"]
        )

        print("✅ Test completed successfully!")
        return configured_model

    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    print("🚀 DeepSpeed-Compatible PEFT Configuration loaded!")

    # Run test if requested
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_model_configuration()
    else:
        print("💡 To run test: python fixed_peft.py --test")