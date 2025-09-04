import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
#from f1chexbert import F1CheXbert
from transformers import HfArgumentParser, AutoConfig, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from src.models import GemmaInference

# Environment setup
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")
hf_token = os.environ.get("HF_TOKEN", "")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
            default="/path/to/your/model",  # Change this to your model path or identifier
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    caching_local: bool = field(default=True, metadata={"help": "Whether to cache the model locally."})
    model_class_name: Optional[str] = field(
            default=None,
            metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"}
    )

    mm_tunable_parts: Optional[str] = field(
            default=None,
            metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", etc.'}
    )




@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
            default=None,
            metadata={"help": "The name of the dataset to use (via the datasets library - or - local function for parquet chexinstruct)."}
    )
    dataset_dir: Optional[str] = field(
            default=None,
            metadata={"help": "Path to a directory containing the dataset files in .parquet format."}
    )
    data_path: str = field(
            default="data",
            metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)

    data_debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode with fewer epochs and smaller batch size."})
    preprocessing_num_workers: Optional[int] = field(
            default=None,
            metadata={"help": "The number of processes to use for the preprocessing."}
    )
    cache_dir: Optional[str] = field(default=CACHE_DIR, metadata={"help": "Path to a directory where the model will be cached."})

@dataclass
class CustomInferenceArguments:
    # Basic training arguments
    output_dir: str = field(default="./results", metadata={"help": "Output directory for model predictions and checkpoints."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    per_device_test_batch_size: int = field(default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed that will be set at the beginning of training."})

    # Model configuration
    model_max_length: int = field(
            default=2048,
            metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    # LoRA/PEFT configuration
    lora_enable: bool = field(default=True, metadata={"help": "Whether to enable LoRA training."})
    lora_r: int = field(default=64, metadata={"help": "Rank for LoRA layers."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_weight_path: str = field(default="", metadata={"help": "Path to LoRA weights."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias."})
    peft_strategy: str = field(default="lora_gaussian", metadata={"help": "PEFT strategy to use."})

    # Inference configuration
    num_beams: int = field(default=1, metadata={"help": "Number of beams for beam search."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for sampling."})
    top_p: float = field(default=0.9, metadata={"help": "Top-p (nucleus) sampling."})
    max_new_tokens: int = field(default=512, metadata={"help": "Maximum number of new tokens to generate."})

    # W&B integration
    report_to: str = field(default="wandb")
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode with fewer epochs and smaller batch size."})

def parse_args_flexible():
    """
    Flexible argument parsing that handles missing arguments gracefully.
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomInferenceArguments))

    # Check if we're running from command line with arguments
    import sys
    if len(sys.argv) > 1:
        try:
            return parser.parse_args_into_dataclasses(return_remaining_strings=True)
        except Exception as e:
            print(f"Error parsing command line arguments: {e}")
            print("Using default arguments instead.")
    # Use defaults if no command line args or parsing failed
    return ModelArguments(), DataArguments(), CustomInferenceArguments(), []



def test():
    """Findings Generation for CheXagent supporting multi-gpu inference"""
    model_args, data_args, test_args, remaining = parse_args_flexible()
    # constant
    rexrank_dir = "modules/ReXrank/data"


    # Load chexinstruct data


    chexpert_plus_test_data = os.path.join(rexrank_dir, "chexpert_plus/ReXRank_CheXpertPlus.json")
    iu_xray_test_data = os.path.join(rexrank_dir, "iu_xray/ReXRank_IUXray_test.json")
    mimic_cxr_test_data = os.path.join(rexrank_dir, "mimic-cxr/ReXRank_MIMICCXR_test.json")

    # Open
    chexpert_data = json.load(open(chexpert_plus_test_data))
    iu_xray_data = json.load(open(iu_xray_test_data))
    mimic_cxr_data = json.load(open(mimic_cxr_test_data))


    data_path = data_args.data_path
    save_dir = os.path.join(test_args.output_dir, "predictions", "Findings Generation")
    # load benchmark



    accelerator = Accelerator()

    """Setup model configuration and load the model."""
    assert model_args.model_name_or_path, "You need to specify a model name or path"

    # Attention implementation check
    if test_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
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

    print("🔄 Setting up collator...")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                              )

    model = PeftModel.from_pretrained(
        model,
        model_args.model_name_or_path,   # directory where adapter is saved
    )

    from safetensors.torch import load_file
    # 2. Load safetensors weights
    #state_dict = load_file("/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/reports/finetune_gemma_findings_lora_llm_projlora64_alpha64_vanilla/epoch_1/model.safetensors")
    #state_dict = {k:v for k, v in state_dict.items()}

    #missing, unexpected = model.load_state_dict(state_dict, strict=False)


    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"


    print("✅ Model loaded successfully.")
    # load the model
    model = GemmaInference(device=f"cuda:{accelerator.process_index}",
                           test_args=test_args,
                           model_instance=model,
                           processor=processor,
                           tokenizer=processor.tokenizer)
    accelerator.wait_for_everyone()
    data_dict = {"chexpert-public": chexpert_data, 'openi':iu_xray_data, 'mimic-cxr':mimic_cxr_data}
    to_be_replaced = {
        "chexpert-public": {'valid': 'valid-512'},
    }

    # inference
    results = []
    for dataset_name, dataset in data_dict.items():
        if accelerator.is_main_process:
            print(f"Processing dataset with {len(dataset)} samples...")

        # filter out samples without section_indication or section_findings

        if accelerator.is_main_process:
            print(f"Filtered dataset size: {len(dataset)}")
        for sample_idx, (patient_id, sample) in tqdm.tqdm(enumerate(dataset.items()), total=len(dataset)):
            if sample['section_findings'] != sample['section_findings']:
                sample['section_findings'] = sample['section_impression']
                if sample['section_impression'] != sample['section_impression']:
                    continue

            text = model.generate(
                    os.path.join(data_path, dataset_name, sample["key_image_path"].replace(list(to_be_replaced[dataset_name].keys())[0],list(to_be_replaced[dataset_name].values())[0])),
                    f'Evaluate the chest X-rays',
                    num_beams=3,
                    temperature=0.5,
                    max_new_tokens=1500,
            )


            results.append({
                    "sample_idx"        : sample_idx,
                    "patient_id"        : patient_id,
                    "image_path"        : sample["key_image_path"],
                    "section_findings"  : sample["section_findings"],
                    "candidate_findings": text,
            })


        # gather results from multiple processes
        results = [results]
        results = gather_object(results)
        if accelerator.is_main_process:
            to_save = [sample for result in results for sample in result]
            to_save = sorted(to_save, key=lambda x: x["sample_idx"])
            save_path = f'{save_dir}/predictions/Findings Generation/{model}.json'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            json.dump(to_save, open(save_path, "wt"), ensure_ascii=False, indent=2)



def compute_scores():
    clean_text = lambda x: re.sub("\s+", " ", re.sub("\[.*?\]", "", x).replace("**", "")).strip().lower()
    root_dir = f"evaluation_chexbench/results/axis_3/axis_3_text_generation/predictions/Findings Generation/"
    result_path = f"{root_dir}/CheXagent.json"
    data = json.load(open(result_path))

    candidates = [clean_text(sample["candidate_findings"]) for sample in data]
    references = [clean_text(sample["section_findings"]) for sample in data]
    text_pairs = [(cand, refer) for cand, refer in zip(candidates, references) if refer]
    candidates, references = [pair[0] for pair in text_pairs], [pair[1] for pair in text_pairs]
    assert len(candidates) == len(references)
    #scores = F1CheXbert()(references, candidates)
    #print(scores)


if __name__ == '__main__':

    test()
    compute_scores()


