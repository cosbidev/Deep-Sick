import os
import torch
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset

import torchvision.transforms as T

__all__ = [
    "load_parquet_image_dataset",
    "save_dataset_as_parquet",
    "image_preproc",
    "SimpleCollator",
    "get_text_column",
    "train_on_responses_only",
    "load_dataset",
]



def load_parquet_image_dataset(dataset_dir: str, skip_test=True) -> DatasetDict:
    """
    Loads a DatasetDict from a directory of split-based parquet files,
    and casts the 'image' column to Sequence[Image].
    """

    split_files = {
        split_file.replace("data_", "").replace(".parquet", ""): os.path.join(dataset_dir, split_file)
        for split_file in os.listdir(dataset_dir)
        if (split_file.endswith(".parquet")) and ("test" not in split_file if skip_test else True)
    }

    dataset_dict = load_dataset("parquet", data_files=split_files)

    return dataset_dict
def save_dataset_as_parquet(dataset_dict, output_dir):
    """
    Save a DatasetDict to Parquet format without copying image files.

    Args:
        dataset_dict (DatasetDict): Hugging Face dataset with an "image" column (list or str).
        output_dir (str): Destination directory where Parquet files will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)


    for split in dataset_dict:
        parquet_path = os.path.join(output_dir, f"data_{split}.parquet")
        dataset_dict[split].to_parquet(parquet_path)
        print(f"✅ Saved split '{split}' to: {parquet_path}")



def image_preproc(img_size: int=224) -> T.Compose:
    """
    Returns a torchvision transform for image preprocessing.
    Args:
        img_size (int): Size to which the image will be resized.

    Returns:
        Transforms.Compose: A composed transform that resizes, crops, and converts images to tensors.

    """
    return T.Compose(
                [T.Resize(img_size + 16), # Resize to a larger size to ensure center crop works well
             T.CenterCrop(img_size), # Center crop to the desired size
             T.ToTensor() # Convert PIL Image or numpy.ndarray to tensor
             ]
    )



def get_text_column(dataset):
    """Find the text column in dataset"""
    text_candidates = ["text", "content", "sentence", "review", "comment", "document"]

    for col in text_candidates:
        if col in dataset.features:
            return col

    # Look for any string column
    for col_name, feature in dataset.features.items():
        if hasattr(feature, 'dtype') and feature.dtype == "string":
            return col_name

    raise ValueError("No text column found in dataset")


class SimpleCollator:
    """Simple collator that can be pickled"""

    def __init__(self, tokenizer, has_text=True, has_images=False):
        self.tokenizer = tokenizer
        self.has_text = has_text
        self.has_images = has_images

    def __call__(self, batch):
        if self.has_text and not self.has_images:
            # Text only
            return self.tokenizer.pad(batch, return_tensors="pt", padding=True)
        elif self.has_images and not self.has_text:
            # Images only
            return {"pixel_values": torch.stack([item["pixel_values"] for item in batch])}
        else:
            # Mixed or other cases
            result = {}
            if self.has_text:
                text_keys = ["input_ids", "attention_mask", "token_type_ids"]
                text_batch = [{k: item[k] for k in text_keys if k in item} for item in batch]
                result.update(self.tokenizer.pad(text_batch, return_tensors="pt", padding=True))

            if self.has_images:
                result["pixel_values"] = torch.stack([torch.Tensor(item["pixel_values"]) for item in batch])

            return result


def _longest_common_sublist(lists):
    """
    Finds the longest common sublist among multiple lists.

    Parameters:
    lists (List[List[int]]): A list of lists.

    Returns:
    List[int]: The longest common sublist. If multiple sublists have the same maximum length,
               one of them is returned. If there's no common sublist, an empty list is returned.
    """
    if not lists:
        return []

    # Find the minimum length among all lists
    min_len = min(len(lst) for lst in lists)
    if min_len == 0:
        return []

    def has_common_sublist(length):
        """
        Checks if there's a common sublist of the given length across all lists.

        Returns:
        (bool, List): Tuple of whether such a sublist exists and the sublist itself.
        """
        common = set()
        first = lists[0]
        # Generate all possible sublists of the given length from the first list
        for i in range(len(first) - length + 1):
            sub = tuple(first[i:i + length])
            common.add(sub)
        pass

        # Iterate over the remaining lists and retain only the common sublists
        for lst in lists[1:]:
            current = set()
            for i in range(len(lst) - length + 1):
                sub = tuple(lst[i:i + length])
                if sub in common:
                    current.add(sub)
            common = current
            if not common:
                return False, []
        pass

        # If common is not empty, return one of the common sublists
        return True, list(common.pop())

    pass

    left, right = 1, min_len
    result = []

    while left <= right:
        mid = left + (right - left) // 2
        exists, sublist = has_common_sublist(mid)
        if exists:
            result = sublist  # Update result with the latest found sublist
            left = mid + 1  # Try to find a longer sublist
        else:
            right = mid - 1  # Try with a shorter length
    pass

    return result


pass


def _find_common_token_ids(component, tokenizer, force_match=False):
    """
    \n### User:\n\n
    \n\n### User:\n\n
    etc
    we need to find the middle most repeatted part.
    Tokenizers can tokenize newlines or spaces as 1 token!
    """
    right_text = ""
    if component.endswith(" "):
        right_text = " "
    elif component.endswith("\n"):
        right_text = "\n"
    left_text = ""
    if component.startswith(" "):
        left_text = " "
    elif component.startswith("\n"):
        left_text = "\n"
    stripped = component.strip()

    # Add current pieces and also newlines
    all_input_ids = []
    if not force_match:
        for left in range(3):
            for right in range(3):
                x = left * left_text + stripped + right * right_text
                x = tokenizer(x, add_special_tokens=False).input_ids
                all_input_ids.append(x)

                x = left * "\n" + stripped + right * "\n"
                x = tokenizer(x, add_special_tokens=False).input_ids
                all_input_ids.append(x)
            pass
        pass
    else:
        x = tokenizer(component, add_special_tokens=False).input_ids
        all_input_ids.append(x)
    pass

    # Old longest common substring is replaced with actual longest common list of numbers
    # substring = _old_longest_common_substring([str(x + [0]) for x in all_input_ids])
    # substring = substring.split(", ")[:-1]
    # substring = [int(x) for x in substring if x.isdigit()]
    substring = _longest_common_sublist([x + [0] for x in all_input_ids])

    # If substring is simply [0], this might be just the original single token
    # Fixes https://github.com/unslothai/unsloth/issues/1290
    # Mistral [INST] [/INST] singular tokens breaks since we output [0] but we need [3] [4]
    if substring == [0] and len(all_input_ids[0]) == 1:
        single_token = all_input_ids[0][0]
        # Confirm single token in every single possible match
        if all(single_token in x for x in all_input_ids):
            substring = [single_token]
    pass

    # Also if substring is original input_ids + [0], then leave it as the original one
    # This happens when no newlines / spaces are used in chat template
    # Eg Phi-4 does not use newlines or spaces
    if (len(set(str(x) for x in all_input_ids)) == 1) and \
            (len(all_input_ids[0]) + 1 == len(substring)) and \
            (all_input_ids[0] == substring[:-1]):

        # Use original un-changed substring
        substring = all_input_ids[0]
    pass

    # Also get rest of tokenized string
    original = tokenizer(component, add_special_tokens=False).input_ids
    # Get optional left and right
    for j in range(len(original)):
        if original[j: j + len(substring)] == substring:
            break
    optional_left = original[:j]
    optional_right = original[j + len(substring):]
    return substring, optional_left, optional_right


pass


def train_on_responses_only(
        trainer,
        instruction_part=None,
        response_part=None,
        force_match=True,  # Match newlines as well!
        tokenizer=None,  # Optional
        return_function=False,  # Useful for iterating over lists
        num_proc=None,
):
    """
    Trains only on responses and not on the instruction by masking out
    the labels with -100 for the instruction part.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if tokenizer is None and trainer is not None:
        tokenizer = trainer.processing_class if hasattr(trainer, "processing_class") else trainer.tokenizer
    # Get non vision tokenizer
    if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if not hasattr(tokenizer, "_unsloth_input_part") or \
            not hasattr(tokenizer, "_unsloth_output_part"):

        if instruction_part is None or response_part is None:
            raise ValueError("Unsloth: instruction_part and response_part must be given!")
        pass
    elif (instruction_part is not None or response_part is not None) and \
            (hasattr(tokenizer, "_unsloth_input_part") or hasattr(tokenizer, "_unsloth_output_part")):

        raise ValueError("Unsloth: Your tokenizer already has instruction and response parts set - do not give custom ones!")
    else:
        instruction_part = tokenizer._unsloth_input_part
        response_part = tokenizer._unsloth_output_part
    pass

    # Get most common tokens since tokenizers can tokenize stuff differently!
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer, force_match)
    A_must, A_left, A_right = _find_common_token_ids(response_part, tokenizer, force_match)

    # Store some temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right
    torch_Tensor = torch.Tensor
    torch_int64 = torch.int64

    def _train_on_responses_only(examples):
        input_ids_ = examples["input_ids"]
        use_tensors = False
        if type(input_ids_) is torch_Tensor:
            use_tensors = True
            input_ids_ = input_ids_.tolist()
        if "labels" in examples:
            labels_ = examples["labels"].tolist()
            assert (len(labels_) == len(input_ids_))
        else:
            labels_ = [None] * len(input_ids_)

        all_labels = []
        for input_ids, old_labels in zip(input_ids_, labels_):
            n = len(input_ids)
            labels = [-100] * n

            use_old_labels = False
            if old_labels is not None:
                use_old_labels = True
                assert (n == len(old_labels))
            n_minus_1 = n - 1
            j = 0
            while j < n:
                # Find <assistant>
                if (input_ids[j] == A_first) and \
                        (input_ids[j: (k := j + len_A_must)] == A_must):

                    # Now backtrack to get previous optional tokens
                    for optional_left in A_left_reversed:
                        if j < 1:
                            break
                        if optional_left == input_ids[j - 1]:
                            j -= 1
                        else:
                            break
                    pass
                    # And forwards look as well
                    for optional_right in A_right_forward:
                        if k >= n_minus_1:
                            break
                        if optional_right == input_ids[k + 1]:
                            k += 1
                        else:
                            break
                    pass
                    # assistant_j = j
                    assistant_k = k

                    j = assistant_k
                    # Given <assistant>, now find next user
                    while j < n:
                        # Find <user>
                        # Also accept last final item if assistant is the last turn
                        if (j == n_minus_1) or \
                                ((input_ids[j] == Q_first) and \
                                 (input_ids[j: (k := j + len_Q_must)] == Q_must)):

                            # Now backtrack to get previous optional tokens
                            for optional_left in Q_left_reversed:
                                if j < 1:
                                    break
                                if optional_left == input_ids[j - 1]:
                                    j -= 1
                                else:
                                    break
                            pass
                            # And forwards look as well
                            for optional_right in Q_right_forward:
                                if k >= n_minus_1:
                                    break
                                if optional_right == input_ids[k + 1]:
                                    k += 1
                                else:
                                    break
                            pass
                            user_j = j
                            # Account for last item
                            if user_j != n_minus_1:
                                # user_k = k
                                # j = user_k
                                j = k
                            else:
                                user_j = n
                                k = n
                            pass

                            if not use_old_labels:
                                # Now copy input_ids to labels
                                labels[assistant_k: user_j] = input_ids[assistant_k: user_j]
                                # print(assistant_j, assistant_k, user_j, user_k)
                            else:
                                # Copy over from old labels!
                                labels[assistant_k: user_j] = old_labels[assistant_k: user_j]
                            break
                        pass
                        j += 1
                    pass
                pass
                j += 1
            pass
            all_labels.append(labels)
        pass
        return {"labels": torch.tensor(all_labels, dtype=torch.int64) if use_tensors else all_labels}

    pass
    if return_function:
        return _train_on_responses_only

    from multiprocessing import cpu_count
    if num_proc is None or type(num_proc) is not int:
        num_proc = cpu_count()

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        if not hasattr(trainer.train_dataset, "map"):
            raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
        if isinstance(trainer.train_dataset, IterableDataset):
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batch_size=trainer.train_dataset._ex_iterable.batch_size, batched=True)
        else:
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched=True, num_proc=num_proc)
    pass

    if hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
        # Eval datasets could be a dict!
        if type(trainer.eval_dataset) is dict:
            for key, value in trainer.eval_dataset.items():
                if not hasattr(value, "map"):
                    raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
                if isinstance(value, IterableDataset):
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batch_size=value._ex_iterable.batch_size, batched=True)
                else:
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batched=True, num_proc=num_proc)
        else:
            if not hasattr(trainer.eval_dataset, "map"):
                raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
            if isinstance(trainer.eval_dataset, IterableDataset):
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batch_size=trainer.eval_dataset._ex_iterable.batch_size, batched=True)
            else:
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched=True, num_proc=num_proc)
        pass
    pass

    # Edit data collator as well if not DataCollatorForSeq2Seq
    from transformers import DataCollatorForSeq2Seq
    if hasattr(trainer, "data_collator") and \
            not isinstance(trainer.data_collator, DataCollatorForSeq2Seq):
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # Check if all labels randomnly got masked to nothing - maybe wrong chat template?
    from .training_utils import fix_zero_training_loss
    fix_zero_training_loss(None, tokenizer, trainer.train_dataset)
    return trainer


pass
