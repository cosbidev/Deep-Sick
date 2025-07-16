#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning script for GEMMA3 or other causal language models using HuggingFace Transformers and Accelerate.

This script is adapted from the official Transformers language modeling tutorial to support finetuning
GEMMA3 and similar models on a text file or dataset, without using the HuggingFace Trainer.

For more details and checkpoints, see:
https://huggingface.co/models?filter=text-generation
"""
# You can adapt this script for your own causal language modeling tasks. See comments for pointers.
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
import os
from accelerate.logging import get_logger
from src.dataset import load_parquet_image_dataset, save_dataset_as_parquet
from src.models import get_collator

os.environ["HF_TOKEN"] = "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"
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["HF_TOKEN"] = "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"
CACHE_DIR = os.path.join(os.getcwd(), "hf_models_cache")
logger = get_logger(__name__)
hf_token = os.environ.get("HF_TOKEN", "")


def parse_args():

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
            "--dataset_dir", type=str, default=None, help="Path to a directory containing the dataset files in .parquet format."
    )
    parser.add_argument(
            "--model_name_or_path",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            required=True,
            default='google/gemma-3-4b-it',
    )

    parser.add_argument(
            "--preprocessing_num_workers",
            type=int,
            default=None,
            help="The number of processes to use for the preprocessing.",
    )

    args = parser.parse_args()

    return args


# @hydra.main(version_base="v1.3", config_path="../../configs/PEFT_runs", config_name="config") # TODO # to use hydra for configuration management
def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    # ----------------------------------- Accelerator Initialization -----------------------------------

    # ----------------------------------- Dataset Loading -----------------------------------
    if args.dataset_dir is not None:
        # if os.path.exists(args.dataset_dir + '_tok'):
        #     args.dataset_dir = args.dataset_dir + '_tok'
        # Loading a dataset from a local directory.
        raw_datasets = load_parquet_image_dataset(
                dataset_dir=args.dataset_dir,
                split_list=["train", "val"],
                cache_dir=cache_dir # Specify the split you want to use ['test', 'val', 'train']
        )



    if args.dataset_name is None and args.dataset_dir is None:
        raise ValueError(
                "You need to specify either a dataset name or a dataset directory. "
                "Use --dataset_name for a HF dataset or --dataset_dir to specify the dataset folder in local (parquet)."
        )

    column_names = raw_datasets["train"].column_names
    if args.model_name_or_path:
        # Get the text column names for the training and evaluation datasets
        collator = get_collator(model_id=args.model_name_or_path,
                                padding_side="right",
                                token=hf_token)
        # If the model is a GEMMA3 model, we use the processor and tokenizer from the collator
        processor = collator.processor
    else:
        raise ValueError(
                "You need to specify a model name or a path to a pretrained model."
        )

    def preprocess(examples):
        return collator.get_tokenize_function()(examples)

    tokenized_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            batch_size=512,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False,  # ← important
            keep_in_memory=True

    )
    save_dataset_as_parquet(dataset_dict=tokenized_datasets,
                            output_dir='data_chexinstruct/hf_parquet_gemma_format/gemma_findings_tok',
                            name_file='tokenized'
                            )



if __name__ == "__main__":
    main()