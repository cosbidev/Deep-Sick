import re

import torch
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Optional
from collections import Counter

def get_peft_regex(
    model,
    finetune_vision_layers     : bool = True,
    finetune_language_layers   : bool = True,
    finetune_attention_modules : bool = True,
    finetune_mlp_modules       : bool = True,
    target_modules             : List[str] = None,
    vision_tags                : Optional[List[str]] = None,
    language_tags              : Optional[List[str]] = None,
    attention_tags             : Optional[List[str]] = None,
    mlp_tags                   : Optional[List[str]] = None

) -> str:
    """
    # Unsloth Zoo - Utilities for Unsloth
    # Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
    #
    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU Lesser General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    #
    # You should have received a copy of the GNU Lesser General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.
    Create a regex pattern to apply LoRA to only select layers of a model.
    """

    if vision_tags is None:
        vision_tags = ["vision", "image", "visual", "patch"]
    if language_tags is None:
        language_tags = ["language", "text"]
    if attention_tags is None:
        attention_tags = ["self_attn", "attention", "attn"]
    if mlp_tags is None:
        mlp_tags = ["mlp", "feed_forward", "ffn", "dense"]


    # All Unsloth Zoo code licensed under LGPLv3
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
        projection_modules  = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
        pass
    else:
        assert(type(target_modules) is list)
        only_linear_modules = list(target_modules)


    # Create regex matcher
    regex_model_parts = []
    if finetune_vision_layers:     regex_model_parts += vision_tags
    if finetune_language_layers:   regex_model_parts += language_tags
    regex_components  = []
    if finetune_attention_modules: regex_components  += attention_tags
    if finetune_mlp_modules:       regex_components  += mlp_tags

    regex_model_parts = "|".join(regex_model_parts)
    regex_components  = "|".join(regex_components)

    match_linear_modules = r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"
    regex_matcher = \
        r".*?(?:"  + regex_model_parts + \
        r").*?(?:" + regex_components + \
        r").*?"    + match_linear_modules + ".*?"

    # Also account for model.layers.0.self_attn/mlp type modules like Qwen
    if finetune_language_layers:
        regex_matcher = r"(?:" + regex_matcher + \
        r")|(?:\bmodel\.layers\.[\d]{1,}\.(?:" + regex_components + \
        r")\.(?:" + match_linear_modules + r"))"

    # Check if regex is wrong since model does not have vision parts
    check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
    if not check:
        regex_matcher = \
            r".*?(?:" + regex_components + \
            r").*?"   + match_linear_modules + ".*?"

    # Final check to confirm if matches exist
    check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
    if not check and target_modules is not None:
        raise RuntimeError(
            f"Unsloth: No layers to finetune? You most likely specified target_modules = {target_modules} incorrectly!"
        )
    elif not check:
        raise RuntimeError(
            f"Unsloth: No layers to finetune for {model.config._name_or_path}. Please file a bug report!"
        )

    return regex_matcher


class ModelParameterManager:
    """
    Gestore per configurare e applicare diverse strategie di fine-tuning ai modelli.
    Supporta LoRA con diverse inizializzazioni e adapter.
    """
    
    def __init__(self,
                 model,
                 finetune_vision_layers=True,
                 finetune_language_layers=True,
                 finetune_attention_modules=True,
                 finetune_mlp_modules=True,
                 **kwargs
                 ):
        assert model is not None, "Model cannot be None"
        # UNSLOTH CODE
        self.target_modules = get_peft_regex(
                model,
                finetune_vision_layers     = finetune_vision_layers,
                finetune_language_layers   = finetune_language_layers,
                finetune_attention_modules = finetune_attention_modules,
                finetune_mlp_modules       = finetune_mlp_modules,
            )



    def return_target_modules(self):
        """
        Restituisce i moduli target per LoRA.
        """
        return self.target_modules


    @staticmethod
    def freeze_all_parameters(model):
        """Congela tutti i parametri del modello."""
        for param in model.parameters():
            param.requires_grad = False
        print("All model parameters frozen.")

    @staticmethod
    def unfreeze_multimodal_projector(model):
        """Sblocca i parametri del multimodal projector."""
        try:
            for name, param in model.model.multi_modal_projector.named_parameters():
                param.requires_grad = True
                print(f"Unfroze multimodal projector: {name}")
        except AttributeError:
            print("Warning: No multimodal_projector found in model.")


    @staticmethod
    def unfreeze_specific_modules(model, module_names: List[str]):
        """Sblocca moduli specifici del modello."""
        for name, param in model.named_parameters():
            for module_name in module_names:
                if module_name in name:
                    param.requires_grad = True
                    print(f"Unfroze: {name}")


    def apply_lora_standard(
        self,
        model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
        freeze_multimodal: bool = True,
        modules_to_save: List[str] = None,
        **kwargs
    ):
        """
        Applica LoRA standard con inizializzazione Kaiming-uniform.

        """
        target_modules = self.return_target_modules()
        self.freeze_all_parameters(model)

        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            modules_to_save=modules_to_save,

        )
        
        model = get_peft_model(model, peft_config)
        print("Applied standard LoRA (Kaiming-uniform initialization)")
        print(model.print_trainable_parameters())
        return model
    
    def apply_lora_gaussian(
        self,
        model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
        freeze_multimodal: bool = True,
        modules_to_save: List[str] = None,
        **kwargs
    ):
        """
        Applica LoRA con inizializzazione Gaussiana.
        """
        target_modules = self.return_target_modules()
        self.freeze_all_parameters(model)
        

        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            init_lora_weights="gaussian",
            modules_to_save=modules_to_save
        )
        
        model = get_peft_model(model, peft_config)
        print("Applied LoRA with Gaussian initialization")
        print(model.print_trainable_parameters())
        return model
    
    def apply_lora_pissa(
        self,
        model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
        freeze_multimodal: bool = True,
        fast_svd: bool = False,
        niter: int = 4,
        modules_to_save: List[str] = None,
        **kwargs
    ):
        """
        Applica LoRA con inizializzazione PiSSA (Principal Singular Subspace Analysis).
        PiSSA converge più rapidamente di LoRA e ottiene performance superiori.
        """

        target_modules = self.return_target_modules()
        self.freeze_all_parameters(model)


        if fast_svd:
            init_method = f"pissa_niter_{niter}"
        else:
            init_method = "pissa"
        
        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            init_lora_weights=init_method,
        )
        
        model = get_peft_model(model, peft_config)
        print(f"Applied LoRA with PiSSA initialization ({'fast' if fast_svd else 'standard'} SVD)")
        print(model.print_trainable_parameters())
        return model


    def apply_lora_olora(
        self,
        model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
        freeze_multimodal: bool = True,
        **kwargs
    ):
        """
        Applica LoRA con inizializzazione OLoRA (QR decomposition).
        OLoRA migliora la stabilità e accelera la convergenza.
        """
        target_modules = self.return_target_modules()
        self.freeze_all_parameters(model)
        


        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            init_lora_weights="olora",

        )
        
        model = get_peft_model(model, peft_config)
        print("Applied LoRA with OLoRA initialization (QR decomposition)")
        print(model.print_trainable_parameters())
        return model
    
    def apply_lora_dora(
        self,
        model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
        freeze_multimodal: bool = True,
        use_rslora: bool = False,
        **kwargs
    ):
        """
        Applica DoRA (Weight-Decomposed Low-Rank Adaptation).
        DoRA decompone gli aggiornamenti in magnitude e direction, migliorando le performance
        soprattutto a rank bassi.
        """
        target_modules = self.return_target_modules()
        self.freeze_all_parameters(model)
        

        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            use_dora=True,
            use_rslora=use_rslora,
        )
        
        model = get_peft_model(model, peft_config)
        print("Applied DoRA (Weight-Decomposed Low-Rank Adaptation)")
        if use_rslora:
            print("- With rank-stabilized LoRA (rsLoRA)")
        print(model.print_trainable_parameters())
        return model
    
    def apply_qora_style(
        self,
        model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
        freeze_multimodal: bool = True,
        modules_to_save: List[str] = None,
        init_method: str = "gaussian",
        **kwargs
    ):
        """
        Applica LoRA in stile QLoRA, targetando tutti i moduli lineari.
        """
        self.freeze_all_parameters(model)


        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules="all-linear",
            init_lora_weights=init_method,
        )
        
        model = get_peft_model(model, peft_config)
        print(f"Applied QLoRA-style LoRA (all-linear, {init_method} init)")
        print(model.print_trainable_parameters())
        return model
    
    def apply_manual_adapter(
        self,
        model,
        adapter_layers: List[str] = None,
        freeze_multimodal: bool = True,
        **kwargs
    ):
        """
        Applica adapter manuale sbloccando specifici layer per il training.
        Questo è un approccio alternativo agli adapter tradizionali usando solo PyTorch.
        
        Args:
            adapter_layers: Lista di nomi di layer da sbloccare per il training
            freeze_multimodal: Se sbloccare il multimodal projector
        """
        self.freeze_all_parameters(model)

        total_unfrozen = 0
        for name, param in model.named_parameters():
            for layer_name in adapter_layers:
                if layer_name in name:
                    param.requires_grad = True
                    total_unfrozen += param.numel()
                    print(f"Unfroze adapter layer: {name}")
        
        print(f"Applied manual adapter approach")
        print(f"Unfroze {total_unfrozen:,} parameters in specified layers")
        return model

    
    def print_model_summary(self, model):
        """Stampa un riassunto dei parametri trainabili del modello."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n=== Model Parameter Summary ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        if hasattr(model, 'print_trainable_parameters'):
            print("\n=== PEFT Summary ===")
            print(model.print_trainable_parameters())




# Esempio di utilizzo:
def configure_model_for_training(
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
    manager = ModelParameterManager()
    
    if strategy == "lora_standard":
        return manager.apply_lora_standard(model, **kwargs)
    elif strategy == "lora_gaussian":
        return manager.apply_lora_gaussian(model, **kwargs)
    elif strategy == "lora_pissa":
        return manager.apply_lora_pissa(model, fast_svd=False, **kwargs)
    elif strategy == "lora_pissa_fast":
        return manager.apply_lora_pissa(model, fast_svd=True, **kwargs)
    elif strategy == "lora_olora":
        return manager.apply_lora_olora(model, **kwargs)
    elif strategy == "dora":
        return manager.apply_lora_dora(model, use_rslora=False, **kwargs)
    elif strategy == "dora_rslora":
        return manager.apply_lora_dora(model, use_rslora=True, **kwargs)
    elif strategy == "qlora":
        return manager.apply_qora_style(model, **kwargs)
    elif strategy == "manual_adapter":
        return manager.apply_manual_adapter(model, **kwargs)
    elif strategy == "modules_to_save":
        return manager.apply_modules_to_save(model, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Esempio di utilizzo pratico:
if __name__ == "__main__":
    # Esempi di come usare le funzioni:
    
    # Per testare le funzioni, decommentare uno dei seguenti esempi:
    
    # # 1. Esempio con un modello fittizio per test
    # import torch
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # 
    # # Carica un modello (esempio con un modello piccolo per test)
    # model_name = "microsoft/DialoGPT-small"  # Piccolo per test rapidi
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # 
    # # Modo 1: Usando la classe direttamente
    # manager = ModelParameterManager()
    # model = manager.apply_lora_gaussian(model, r=8, lora_alpha=16)
    # manager.print_model_summary(model)
    
    # # 2. Esempio usando la funzione di convenienza
    # model = configure_model_for_training(
    #     model, 
    #     strategy="lora_pissa_fast", 
    #     r=16, 
    #     lora_alpha=32,
    #     lora_dropout=0.1
    # )
    
    # # 3. Esempio con adapter manuale
    # model = configure_model_for_training(
    #     model,
    #     strategy="manual_adapter",
    #     adapter_layers=["h.10", "h.11"]  # ultimi 2 layer per DialoGPT
    # )
    
    # # 4. Esempio LoRA con moduli aggiuntivi da salvare
    # model = configure_model_for_training(
    #     model,
    #     strategy="modules_to_save",
    #     modules_to_save=["lm_head"],
    #     r=8,
    #     init_method="pissa"
    # )
    
    # # 5. Stampa riassunto finale
    # manager = ModelParameterManager()
    # manager.print_model_summary(model)
    
    print("ModelParameterManager loaded successfully!")
    print("Uncomment the examples above to test the functions.")


def quick_test_example():
    """
    Funzione di test rapido che mostra come usare il manager.
    Chiamare questa funzione per testare le funzionalità.
    """
    print("=== Quick Test Example ===")
    print("Per testare le funzioni:")
    print("1. Carica un modello con transformers:")
    print("   model = AutoModelForCausalLM.from_pretrained('model_name')")
    print("2. Configura per il training:")
    print("   model = configure_model_for_training(model, strategy='lora_pissa')")
    print("3. Stampa il riassunto:")
    print("   manager = ModelParameterManager()")
    print("   manager.print_model_summary(model)")
    
    return "Test instructions displayed!"


def demo_model_configuration():
    """
    Funzione demo che mostra come configurare un modello per il training.
    Questa funzione può essere chiamata per vedere il manager in azione.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("=== Demo: Model Configuration ===")
        print("Caricamento di un modello piccolo per demo...")
        
        # Usa un modello molto piccolo per la demo
        model_name = "microsoft/DialoGPT-small"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"Modello caricato: {model_name}")
        
        # Crea il manager
        manager = ModelParameterManager()
        
        # Mostra stato iniziale
        print("\n=== Stato Iniziale ===")
        manager.print_model_summary(model)
        
        # Test 1: LoRA Gaussiano
        print("\n=== Test 1: LoRA Gaussiano ===")
        model_lora = configure_model_for_training(
            model, 
            strategy="lora_gaussian", 
            r=8, 
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"]  # moduli corretti per DialoGPT
        )
        
        # Test 2: Stampa riassunto finale
        print("\n=== Riassunto Finale ===")
        manager.print_model_summary(model_lora)
        
        print("\n✅ Demo completata con successo!")
        return model_lora
        
    except ImportError:
        print("❌ transformers non installato. Installare con: pip install transformers")
        return None
    except Exception as e:
        print(f"❌ Errore durante la demo: {e}")
        return None


def demo_all_strategies():
    """
    Demo che mostra tutte le strategie disponibili.
    """
    strategies = [
        "lora_standard",
        "lora_gaussian", 
        "lora_pissa",
        "lora_olora",
        "dora"
    ]
    
    print("=== Strategie Disponibili ===")
    for strategy in strategies:
        print(f"- {strategy}")
    
    print("\nPer testare una strategia:")
    print("model = configure_model_for_training(model, strategy='lora_pissa')")
    
    return strategies


# Chiamata di esempio per mostrare che il modulo funziona
if __name__ == "__main__":
    print("🚀 ModelParameterManager caricato!")
    
    # Mostra le strategie disponibili
    demo_all_strategies()
    
    # Esegui demo solo se richiesto esplicitamente
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("\n" + "="*50)
        print("Eseguendo demo completa...")
        demo_model_configuration()
    else:
        print("\nPer eseguire la demo completa, lancia:")
        print("python peft.py --demo")
        
    result = quick_test_example()