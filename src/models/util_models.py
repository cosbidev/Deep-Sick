from tqdm import tqdm
def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def est_flops(n_params, batch_tokens):
    """Estimate FLOPs for forward + backward pass"""
    # Forward: 2 * n_params * batch_tokens (matrix multiply approximation)
    # Backward: 4 * n_params * batch_tokens (gradient computation)
    return 6 * n_params * batch_tokens

def est_vlm_flops(n_params, batch_size, image_tokens, text_tokens):
    """Estimate FLOPs for VLM forward + backward pass"""
    # VLM combines vision and language processing
    # Vision forward: 2 * vision_params * batch_size * image_tokens
    # Language forward: 2 * language_params * batch_size * text_tokens
    # Cross-attention: 2 * cross_attn_params * batch_size * image_tokens * text_tokens
    # Backward: ~4x forward cost
    total_tokens = image_tokens + text_tokens + (image_tokens * text_tokens / 1000)  # Cross-attention approx
    return 6 * n_params * batch_size * total_tokens



def count_tokens_worker(args):
    """Worker function for parallel token counting"""
    tokenizer_name, samples, text_column = args

    # Import here to avoid pickling issues
    from transformers import AutoTokenizer

    # Load tokenizer in worker process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_total_tokens = 0
    local_valid_samples = 0

    for ex in samples:
        text = ex.get(text_column)
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            continue
        try:
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
            local_total_tokens += len(tokens)
            local_valid_samples += 1
        except Exception:
            continue

    return local_total_tokens, local_valid_samples


def count_tokens(tokenizer, dataset, text_column="text", num_workers=None):
    """Count total tokens in dataset using parallel processing"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 1)  # Use up to 1 workers

    dataset_size = len(dataset)
    if dataset_size == 0:
        return 0, 0, 0

    # For small datasets, use single process
    if num_workers == 1:
        return count_tokens_sequential(tokenizer, dataset, text_column)

    print(f"Counting tokens using {num_workers} workers...")

    # Split dataset into chunks for parallel processing
    chunk_size = max(1, dataset_size // num_workers)
    chunks = []

    for i in range(0, dataset_size, chunk_size):
        end_idx = min(i + chunk_size, dataset_size)
        chunk = [dataset[j] for j in range(i, end_idx)]
        chunks.append((tokenizer.name_or_path, chunk, text_column))

    total_tokens = 0
    valid_samples = 0

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(count_tokens_worker, chunk): i
                           for i, chunk in enumerate(chunks)}

        # Collect results with progress bar
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                try:
                    chunk_tokens, chunk_valid = future.result()
                    total_tokens += chunk_tokens
                    valid_samples += chunk_valid
                    pbar.update(1)
                except Exception as e:
                    print(f"Warning: Failed to process chunk: {e}")
                    pbar.update(1)

    if valid_samples == 0:
        raise ValueError(f"No valid text found in column '{text_column}'")

    avg_tokens = total_tokens / valid_samples if valid_samples > 0 else 0
    print(f"Dataset analysis (parallel processing):")
    print(f"  - Total samples: {dataset_size}")
    print(f"  - Valid samples: {valid_samples}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average tokens per sample: {avg_tokens:.2f}")
    print(f"  - Workers used: {num_workers}")

    return total_tokens, valid_samples, avg_tokens


def count_tokens_sequential(tokenizer, dataset, text_column="text"):
    """Sequential token counting for small datasets or fallback"""
    total_tokens = 0
    valid_samples = 0

    for ex in tqdm(dataset, desc="Counting tokens (sequential)"):
        text = ex.get(text_column)
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            continue
        try:
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
            total_tokens += len(tokens)
            valid_samples += 1
        except Exception as e:
            print(f"Warning: Failed to tokenize sample: {e}")
            continue

    if valid_samples == 0:
        raise ValueError(f"No valid text found in column '{text_column}'")

    avg_tokens = total_tokens / valid_samples if valid_samples > 0 else 0
    print(f"Dataset analysis (sequential):")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Valid samples: {valid_samples}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average tokens per sample: {avg_tokens:.2f}")

    return total_tokens, valid_samples, avg_tokens

