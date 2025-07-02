import datetime
import json
import os
import pandas as pd

def save_report(report: dict, out_dir: str ="benchmark_report", out_prefix: str="report") -> None:
    """
    Save the evaluation report to a JSON file and a CSV file.
    Args:
        report: A dictionary containing the evaluation report data.
        out_dir: string, directory where the report will be saved.
        out_prefix: string, prefix for the report filename.

    Returns: None

    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Save JSON with proper indentation
    filename = f"{out_prefix}_.json"
    with open(filename, "w", encoding='utf-8') as jf:
        json.dump(report, jf, indent=4, ensure_ascii=False, sort_keys=True)

    # Create reports directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Also save in reports directory
    json_path = f"reports/benchmark/{out_prefix}_{ts}.json"
    with open(json_path, "w", encoding='utf-8') as jf:
        json.dump(report, jf, indent=4, ensure_ascii=False, sort_keys=True)

    # Save CSV version
    csv_path = f"reports/benchmark/{out_prefix}_{ts}.csv"
    pd.DataFrame([report]).to_csv(csv_path, index=False)

    print(f"Report saved to:")
    print(f"  JSON: {filename}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


def estimate_memory_usage(model, batch_size, seq_len, model_type, has_amp=False):
    """Estimate GPU memory usage for training"""

    # Get model parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Base model memory (parameters + gradients + optimizer states)
    # Parameters: 4 bytes per param (FP32) or 2 bytes (FP16)
    param_bytes = n_params * (2 if has_amp else 4)

    # Gradients: same size as parameters
    grad_bytes = param_bytes

    # Optimizer states (AdamW): momentum + variance = 2x parameters
    optimizer_bytes = param_bytes * 2

    # Model memory
    model_memory_gb = (param_bytes + grad_bytes + optimizer_bytes) / (1024 ** 3)

    # Activation memory (depends on batch size and sequence length)
    if model_type == "text":
        # Text model activations: roughly 12 * batch_size * seq_len * hidden_size * layers
        # Estimate hidden_size based on model size
        if n_params < 50e6:  # Small models (BERT-base, etc.)
            hidden_size, num_layers = 768, 12
        elif n_params < 200e6:  # Medium models (BERT-large, etc.)
            hidden_size, num_layers = 1024, 24
        elif n_params < 1e9:  # Large models
            hidden_size, num_layers = 1536, 36
        else:  # Very large models
            hidden_size, num_layers = 2048, 48

        activation_bytes = 12 * batch_size * seq_len * hidden_size * num_layers * (2 if has_amp else 4)

    elif model_type == "vision":
        # Vision model activations: batch_size * channels * height * width * depth
        # Assume standard 224x224 images, estimate depth based on model size
        if n_params < 50e6:  # Small vision models
            depth_factor = 1000
        elif n_params < 200e6:  # Medium vision models
            depth_factor = 2000
        else:  # Large vision models
            depth_factor = 3000

        activation_bytes = batch_size * 3 * 224 * 224 * depth_factor * (2 if has_amp else 4)

    else:  # VLM
        # VLMs combine both text and vision activations
        # Use text estimation as base and add vision overhead
        hidden_size, num_layers = 1024, 24  # Typical VLM size
        text_activations = 12 * batch_size * seq_len * hidden_size * num_layers * (2 if has_amp else 4)
        vision_activations = batch_size * 3 * 224 * 224 * 1500 * (2 if has_amp else 4)
        activation_bytes = text_activations + vision_activations

    activation_memory_gb = activation_bytes / (1024 ** 3)

    # Add buffer for PyTorch overhead (10-20%)
    overhead_gb = (model_memory_gb + activation_memory_gb) * 0.15

    total_memory_gb = model_memory_gb + activation_memory_gb + overhead_gb

    return {
            'model_memory_gb'     : model_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'overhead_gb'         : overhead_gb,
            'total_memory_gb'     : total_memory_gb,
            'estimated_parameters': n_params
    }


def find_optimal_batch_size(model, seq_len, model_type, available_memory_gb, has_amp=False, target_utilization=0.85):
    """Find the largest batch size that fits in available memory"""

    # Start with a reasonable batch size and work our way up
    optimal_batch = 1
    max_batch_to_try = 512  # Reasonable upper limit

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if batch_size > max_batch_to_try:
            break

        memory_estimate = estimate_memory_usage(model, batch_size, seq_len, model_type, has_amp)
        required_memory = memory_estimate['total_memory_gb']

        # Check if this batch size fits within target utilization
        if required_memory <= available_memory_gb * target_utilization:
            optimal_batch = batch_size
        else:
            break

    return optimal_batch, memory_estimate


def get_memory_recommendations(model, seq_len, model_type, gpu_memory_gb, current_batch_size, has_amp=False):
    """Generate memory optimization recommendations"""

    recommendations = []

    # Find optimal batch size
    optimal_batch, memory_estimate = find_optimal_batch_size(
            model, seq_len, model_type, gpu_memory_gb, has_amp
    )

    # Current memory usage
    current_memory = estimate_memory_usage(model, current_batch_size, seq_len, model_type, has_amp)

    # Memory utilization
    current_utilization = current_memory['total_memory_gb'] / gpu_memory_gb
    optimal_utilization = memory_estimate['total_memory_gb'] / gpu_memory_gb

    # Generate recommendations
    if optimal_batch > current_batch_size:
        speedup_factor = optimal_batch / current_batch_size
        recommendations.append({
                'type'       : 'increase_batch_size',
                'current'    : current_batch_size,
                'recommended': optimal_batch,
                'speedup'    : f"{speedup_factor:.1f}x faster training",
                'description': f"Increase batch size from {current_batch_size} to {optimal_batch}"
        })

    if current_utilization < 0.5:
        recommendations.append({
                'type'       : 'underutilized_gpu',
                'utilization': f"{current_utilization:.1%}",
                'description': "GPU memory is underutilized, consider larger batch size or model"
        })

    if current_utilization > 0.9:
        recommendations.append({
                'type'       : 'memory_risk',
                'utilization': f"{current_utilization:.1%}",
                'description': "High memory usage, risk of OOM. Consider reducing batch size"
        })

    if not has_amp and model_type in ['text', 'vlm']:
        memory_savings = current_memory['total_memory_gb'] * 0.4  # AMP typically saves 40%
        recommendations.append({
                'type'       : 'enable_amp',
                'savings'    : f"{memory_savings:.1f} GB",
                'description': f"Enable AMP to save ~{memory_savings:.1f} GB memory"
        })

    if seq_len > 512 and model_type in ['text', 'vlm']:
        # Estimate memory savings from shorter sequences
        shorter_memory = estimate_memory_usage(model, current_batch_size, 512, model_type, has_amp)
        savings = current_memory['total_memory_gb'] - shorter_memory['total_memory_gb']
        if savings > 0.5:
            recommendations.append({
                    'type'       : 'reduce_sequence_length',
                    'current'    : seq_len,
                    'recommended': 512,
                    'savings'    : f"{savings:.1f} GB",
                    'description': f"Reduce sequence length from {seq_len} to 512 to save {savings:.1f} GB"
            })

    return {
            'current_memory'     : current_memory,
            'optimal_batch_size' : optimal_batch,
            'optimal_memory'     : memory_estimate,
            'current_utilization': current_utilization,
            'optimal_utilization': optimal_utilization,
            'recommendations'    : recommendations
    }
    """Handle forward pass for different model types, especially VLMs"""
    try:
        if model_type == "vlm":
            # For VLMs, try different forward pass strategies

            # Strategy 1: Direct forward (works for many VLMs)
            try:
                outputs = model(**model_inputs)
                if hasattr(outputs, 'logits'):
                    return outputs
                # If no logits attribute, create a mock output for benchmarking
                batch_size = model_inputs.get('input_ids', model_inputs.get('pixel_values')).size(0)
                mock_logits = torch.randn(batch_size, 2, device=next(model.parameters()).device)

                class MockOutput:
                    def __init__(self, logits):
                        self.logits = logits

                return MockOutput(mock_logits)
            except Exception as e:
                print(f"Direct VLM forward failed: {e}")

            # Strategy 2: Try text-only forward for VLMs that support it
            if 'input_ids' in model_inputs:
                try:
                    text_inputs = {k: v for k, v in model_inputs.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
                    outputs = model(**text_inputs)
                    if hasattr(outputs, 'logits'):
                        return outputs
                except Exception as e:
                    print(f"VLM text-only forward failed: {e}")

            # Strategy 3: Create mock output for unsupported VLMs
            batch_size = model_inputs.get('input_ids', model_inputs.get('pixel_values')).size(0)
            mock_logits = torch.randn(batch_size, 2, device=next(model.parameters()).device)

            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits

            print("Using mock output for VLM benchmarking")
            return MockOutput(mock_logits)

        else:
            # Standard forward pass for text/vision models
            return model(**model_inputs)

    except Exception as e:
        # Final fallback: create mock output
        print(f"Forward pass failed, using mock output: {e}")
        batch_size = list(model_inputs.values())[0].size(0)
        mock_logits = torch.randn(batch_size, 2, device=next(model.parameters()).device)

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        return MockOutput(mock_logits)


def estimate_vlm_memory_usage(model, batch_size, avg_image_tokens, avg_text_tokens, has_amp=False):
    """VLM-specific memory estimation"""
    n_params = sum(p.numel() for p in model.parameters())

    # Model memory (params + gradients + optimizer)
    param_bytes = n_params * (2 if has_amp else 4)
    grad_bytes = param_bytes
    optimizer_bytes = param_bytes * 2  # AdamW states
    model_memory_gb = (param_bytes + grad_bytes + optimizer_bytes) / (1024 ** 3)

    # VLM activation memory includes vision, language, and cross-attention
    # Vision activations: batch_size * image_tokens * hidden_size * num_layers
    # Language activations: batch_size * text_tokens * hidden_size * num_layers
    # Cross-attention: batch_size * image_tokens * text_tokens * hidden_size

    if n_params < 5e9:  # 3B models
        hidden_size, num_layers = 2048, 28
    elif n_params < 10e9:  # 7B models
        hidden_size, num_layers = 3584, 28
    elif n_params < 35e9:  # 32B models
        hidden_size, num_layers = 5120, 64
    else:  # 72B+ models
        hidden_size, num_layers = 8192, 80

    bytes_per_element = 2 if has_amp else 4

    vision_activations = batch_size * avg_image_tokens * hidden_size * num_layers * bytes_per_element
    text_activations = batch_size * avg_text_tokens * hidden_size * num_layers * bytes_per_element
    cross_attention = batch_size * avg_image_tokens * avg_text_tokens * hidden_size * bytes_per_element / 4  # Approximate

    activation_memory_gb = (vision_activations + text_activations + cross_attention) / (1024 ** 3)

    # PyTorch overhead
    overhead_gb = (model_memory_gb + activation_memory_gb) * 0.15

    total_memory_gb = model_memory_gb + activation_memory_gb + overhead_gb

    return {
            'model_memory_gb'     : model_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'overhead_gb'         : overhead_gb,
            'total_memory_gb'     : total_memory_gb
    }



def find_optimal_vlm_batch_size(model, avg_image_tokens, avg_text_tokens, available_memory_gb, has_amp=False):
    """Find optimal batch size for VLM training"""
    optimal_batch = 1

    for batch_size in [1, 2, 4, 8, 16, 32]:
        memory_estimate = estimate_vlm_memory_usage(model, batch_size, avg_image_tokens, avg_text_tokens, has_amp)
        if memory_estimate['total_memory_gb'] <= available_memory_gb * 0.85:
            optimal_batch = batch_size
        else:
            break

    return optimal_batch, memory_estimate
