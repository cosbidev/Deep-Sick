import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import HfArgumentParser, AutoConfig, AutoModelForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel
from src.models import GemmaInference


# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")
cache_dir = os.path.join(os.getcwd(), "hf_cache")
for var in ["HF_DATASETS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"]:
    os.environ[var] = cache_dir
hf_token = os.environ.get("HF_TOKEN", "")


# -------------------------------------------------------------------
# Argument classes
# -------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/path/to/your/model",
        metadata={"help": "Path or identifier of pretrained model"}
    )
    caching_local: bool = field(default=True)
    model_class_name: Optional[str] = None
    mm_tunable_parts: Optional[str] = None


@dataclass
class DataArguments:
    dataset_name: Optional[str] = None
    dataset_dir: Optional[str] = None
    data_path: str = "data"
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = None
    data_debug: bool = False
    preprocessing_num_workers: Optional[int] = None
    cache_dir: Optional[str] = CACHE_DIR


@dataclass
class CustomInferenceArguments:
    output_dir: str = "./results"
    num_train_epochs: float = 3.0
    per_device_test_batch_size: int = 1
    seed: Optional[int] = None
    model_max_length: int = 2048
    attn_implementation: str = "flash_attention_2"
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    peft_strategy: str = "lora_gaussian"
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 512
    report_to: str = "wandb"
    debug: bool = False


# -------------------------------------------------------------------
# Flexible parsing
# -------------------------------------------------------------------
def parse_args_flexible():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomInferenceArguments))
    import sys
    if len(sys.argv) > 1:
        try:
            return parser.parse_args_into_dataclasses(return_remaining_strings=True)
        except Exception as e:
            print(f"Argument parsing failed: {e}\n→ Using defaults.")
    return ModelArguments(), DataArguments(), CustomInferenceArguments(), []


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
def test():
    model_args, data_args, test_args, _ = parse_args_flexible()
    rexrank_dir = "modules/ReXrank/data"

    # Load benchmark data
    datasets = {
        "chexpert-public": os.path.join(rexrank_dir, "chexpert_plus/ReXRank_CheXpertPlus.json"),
        "openi": os.path.join(rexrank_dir, "iu_xray/ReXRank_IUXray_test.json"),
        "mimic-cxr": os.path.join(rexrank_dir, "mimic-cxr/ReXRank_MIMICCXR_test.json"),
    }
    data_dict = {k: json.load(open(v)) for k, v in datasets.items()}

    save_dir = os.path.join(test_args.output_dir, "predictions", "FindingsGeneration")
    accelerator = Accelerator()

    # Attention backend sanity check
    if test_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise RuntimeError("sdpa attention requires torch>=2.1.2")

    # Config
    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
    cfg_pretrained.text_config.hidden_size = 2560

    # Load model
    print("🔄 Loading base model...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=data_args.cache_dir,
    #     torch_dtype=torch.bfloat16,
    #     config=cfg_pretrained,
    # )
    if 'gemma' in model_args.model_name_or_path.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR,
                    config=cfg_pretrained,

            )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    model = PeftModel.from_pretrained(model, model_args.model_name_or_path)

    # Load merged weights
    ckpt_path = os.path.join(model_args.model_name_or_path, "non_lora_state_dict.bin")
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    print("✅ Model ready.")
    model = GemmaInference(
        device=f"cuda:{accelerator.process_index}",
        test_args=test_args,
        model_instance=model,
        processor=processor,
        tokenizer=processor.tokenizer,
    )

    accelerator.wait_for_everyone()

    to_be_replaced = {"chexpert-public": {"valid": "valid-512"}}
    results = []

    # Iterate over datasets
    for dataset_name, dataset in data_dict.items():
        if accelerator.is_main_process:
            print(f"{dataset_name}: {len(dataset)} samples")

        for sample_idx, (pid, sample) in tqdm.tqdm(enumerate(dataset.items()), total=len(dataset)):
            if sample.get("section_findings") != sample.get("section_findings"):  # NaN check
                sample["section_findings"] = sample.get("section_impression")
                if sample["section_findings"] != sample["section_findings"]:
                    continue

            image_path = sample["key_image_path"].replace(
                *list(to_be_replaced.get(dataset_name, {}).items())[0]
            ) if dataset_name in to_be_replaced else sample["key_image_path"]

            text = model.generate(
                os.path.join(data_args.data_path, dataset_name, image_path),
                "Evaluate the chest X-rays and describe any evolving patterns, changes, or developments in the findings",
                num_beams=1,
                temperature=0.9,
                max_new_tokens=2048,
            )

            results.append({
                "sample_idx": sample_idx,
                "patient_id": pid,
                "image_path": sample["key_image_path"],
                "section_findings": sample["section_findings"],
                "candidate_findings": text,
            })

        # Sync results
        results = gather_object([results])
        if accelerator.is_main_process:
            merged = [s for r in results for s in r]
            merged.sort(key=lambda x: x["sample_idx"])
            save_path = os.path.join(save_dir, f"{dataset_name}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            json.dump(merged, open(save_path, "w"), ensure_ascii=False, indent=2)


# -------------------------------------------------------------------
# Metrics (stub)
# -------------------------------------------------------------------
def compute_scores():
    clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip().lower()
    result_path = "evaluation_chexbench/results/axis_3/axis_3_text_generation/predictions/FindingsGeneration/CheXagent.json"
    if not os.path.isfile(result_path):
        return
    data = json.load(open(result_path))
    candidates = [clean(s["candidate_findings"]) for s in data]
    references = [clean(s["section_findings"]) for s in data if s["section_findings"]]
    assert len(candidates) == len(references)


# -------------------------------------------------------------------
if __name__ == "__main__":
    test()
    compute_scores()
