import math
from itertools import chain
import os
import numpy as np
import torch
import logging
from torch.utils.data import Sampler
from transformers import MODEL_MAPPING,  SchedulerType
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import torch.distributed as dist


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)






def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# New Code #
def evaluate(training_args, model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks






def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # Ensure all samples have non-zero length
    assert all(l != 0 for l in lengths), "Should not have zero length."
    # If all samples belong to a single modality (either all multimodal or all language-only), apply default grouping
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    # Separate indices and lengths of multimodal and language-only samples
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # Perform length-based grouping and random shuffling for multimodal samples
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    # Perform length-based grouping and random shuffling for language-only samples
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # Prepare for optional merging of final smaller batches from each modality
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    # Combine all full megabatches from both modalities
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    # Shuffle the megabatches to avoid ordering bias
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    # Apply the random shuffling to the megabatches
    megabatches = [megabatches[i] for i in megabatch_indices]

    # Optionally add the remaining samples as a final megabatch
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]



class BlueprintGroupedSampler(Sampler):
    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths,
            n_images,
            generator=None,
            seed: int = 42
    ):
        super().__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.global_batch_size = batch_size * world_size
        self.generator = generator or torch.Generator().manual_seed(seed)

        # Safe rank detection
        try:
            import torch.distributed as dist
            self.rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            self.rank = int(os.environ.get("LOCAL_RANK", 0))

        self.indices = self._build_index_sequence(lengths, n_images)

    def _build_index_sequence(self, lengths, n_images):
        """
        Costruisce una sequenza di indici dove ogni batch ha ESATTAMENTE lo stesso numero
        totale di immagini per ogni GPU per garantire bilanciamento perfetto.
        """
        n_images_array = np.array(n_images)

        # Separa gli indici per numero di immagini
        indices_2img = np.where(n_images_array == 2)[0]
        indices_1img = np.where(n_images_array == 1)[0]

        # Shuffle deterministico degli indici per ogni categoria
        rng_2 = torch.Generator().manual_seed(42)
        rng_1 = torch.Generator().manual_seed(43)

        perm_2 = torch.randperm(len(indices_2img), generator=rng_2).numpy()
        perm_1 = torch.randperm(len(indices_1img), generator=rng_1).numpy()

        indices_2img = indices_2img[perm_2]
        indices_1img = indices_1img[perm_1]


        # Calcola strategia per garantire lo stesso numero di immagini per GPU
        if self.batch_size == 4:
            # Strategia fissa: 3 campioni da 2 img + 1 da 1 img = 7 immagini per GPU
            samples_2img_per_gpu = 3
            samples_1img_per_gpu = 1
            images_per_gpu = 7
        elif self.batch_size == 2:
            # Strategia fissa: 1 campione da 2 img + 0 da 1 img = 2 immagini per GPU
            samples_2img_per_gpu = 1
            samples_1img_per_gpu = 0
            images_per_gpu = 2
        elif self.batch_size == 1:
            # Strategia fissa: 1 campione da 2 img + 0 da 1 img = 2 immagini per GPU
            samples_2img_per_gpu = 2
            samples_1img_per_gpu = 1
            images_per_gpu = 2
        else:
            # Strategia di fallback
            samples_2img_per_gpu = min(self.batch_size, self.batch_size * 2 // 3)
            samples_1img_per_gpu = self.batch_size - samples_2img_per_gpu
            images_per_gpu = samples_2img_per_gpu * 2 + samples_1img_per_gpu

        # Calcola quante batch complete possiamo creare
        max_batches_2img = len(indices_2img) // (samples_2img_per_gpu * self.world_size)
        max_batches_1img = len(indices_1img) // (samples_1img_per_gpu * self.world_size) if samples_1img_per_gpu > 0 else float('inf')
        max_batches = min(max_batches_2img, max_batches_1img)

        print(f"[BlueprintSampler] Strategia: {samples_2img_per_gpu}x2img + {samples_1img_per_gpu}x1img = {images_per_gpu} img/GPU")
        print(f"[BlueprintSampler] Max batches possibili: {max_batches}")
        print(f"[BlueprintSampler] Campioni utilizzati: {max_batches * self.world_size * self.batch_size}")

        batches = []

        # Crea batch con strategia fissa
        for batch_idx in range(max_batches):
            global_batch = []

            for gpu_rank in range(self.world_size):
                gpu_batch = []

                # Aggiungi campioni da 2 immagini
                for _ in range(samples_2img_per_gpu):
                    idx = batch_idx * self.world_size * samples_2img_per_gpu + gpu_rank * samples_2img_per_gpu + len(gpu_batch)
                    if idx < len(indices_2img):
                        gpu_batch.append(indices_2img[idx])

                # Aggiungi campioni da 1 immagine
                for _ in range(samples_1img_per_gpu):
                    idx = batch_idx * self.world_size * samples_1img_per_gpu + gpu_rank * samples_1img_per_gpu + (len(gpu_batch) - samples_2img_per_gpu)
                    if idx < len(indices_1img):
                        gpu_batch.append(indices_1img[idx])

                global_batch.extend(gpu_batch)

            batches.append(global_batch)

        # Aggiungi batch rimanenti solo con campioni da 1 immagine (se batch_size lo permette)
        if samples_1img_per_gpu == 0 and len(indices_1img) > 0:
            # Modalità fallback per batch_size=2: usa solo campioni da 1 immagine
            remaining_1img = len(indices_1img)
            additional_batches = remaining_1img // (self.batch_size * self.world_size)

            for batch_idx in range(additional_batches):
                global_batch = []
                for gpu_rank in range(self.world_size):
                    gpu_batch = []
                    for sample_idx in range(self.batch_size):
                        idx = batch_idx * self.world_size * self.batch_size + gpu_rank * self.batch_size + sample_idx
                        if idx < len(indices_1img):
                            gpu_batch.append(indices_1img[idx])
                    global_batch.extend(gpu_batch)

                if len(global_batch) == self.global_batch_size:
                    batches.append(global_batch)

        # Shuffle delle batch per evitare pattern
        if len(batches) > 1:
            batch_indices = torch.randperm(len(batches), generator=torch.Generator().manual_seed(123)).tolist()
            batches = [batches[i] for i in batch_indices]

        # Flatten tutte le batch
        indices_sequence = []
        for batch in batches:
            indices_sequence.extend(batch)


        return indices_sequence

    def __iter__(self):
        """
        Restituisce gli indici per il rank corrente usando operazioni vettoriali.
        """
        # Verifica che gli indici siano divisibili per il batch size globale
        total_samples = len(self.indices)
        if total_samples % self.global_batch_size != 0:
            # Tronca per avere un numero esatto di batch complete
            total_samples = (total_samples // self.global_batch_size) * self.global_batch_size
            self.indices = self.indices[:total_samples]

        # Converti in numpy array per operazioni vettoriali più efficienti
        indices_array = np.array(self.indices)

        # Reshape in batch globali: (num_batches, global_batch_size)
        num_batches = len(indices_array) // self.global_batch_size
        reshaped = indices_array.reshape(num_batches, self.global_batch_size)

        # Estrai slice per il rank corrente da ogni batch
        rank_start = self.rank * self.batch_size
        rank_end = (self.rank + 1) * self.batch_size
        rank_batches = reshaped[:, rank_start:rank_end]

        # Flatten per ottenere la sequenza finale
        rank_indices = rank_batches.flatten()

        return iter(rank_indices.tolist())

    def __len__(self):
        """
        Restituisce il numero totale di campioni per il rank corrente.
        """
        total_samples = len(self.indices)
        # Assicurati che sia divisibile per il batch size globale
        total_samples = (total_samples // self.global_batch_size) * self.global_batch_size
        return total_samples // self.world_size



