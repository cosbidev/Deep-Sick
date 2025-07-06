#!/usr/bin/env python3
"""
Simple orchestrator script for testing memory usage with LaTeX OCR dataset.
Works with both DeepSpeed and simple (single GPU) modes.
"""

import subprocess
import sys
import argparse
from itertools import count



def run_memory_test(model_type="gemma", max_batch_size=None, use_deepspeed=False):
    """
    Run memory usage tests for the specified model type.

    Args:
        model_type: Either "gemma" or "qwen_vl"
        max_batch_size: Maximum batch size to test (None for no limit)
        use_deepspeed: Whether to use DeepSpeed or simple mode
    """

    suffix = "_deepspeed" if use_deepspeed else "_simple"
    results_file = f"./results_{model_type}{suffix}.csv"

    # Clear previous results
    with open(results_file, "w") as f:
        # Write CSV header
        f.write("batch_size,")
        max_gpus = 8 if use_deepspeed else 1
        for gpu in range(max_gpus):
            f.write(f"gpu{gpu}_active_mib,gpu{gpu}_reserved_mib,")
        f.write("iterations_per_second\n")

    mode_str = "DeepSpeed" if use_deepspeed else "Simple"
    print(f"Starting {mode_str} memory usage tests for {model_type} model...")
    print(f"Results will be saved to: {results_file}")

    for batch_size in count(1):
        if max_batch_size and batch_size > max_batch_size:
            print(f"Reached maximum batch size limit: {max_batch_size}")
            break

        print(f"\n{'=' * 50}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'=' * 50}")

        succeeded = False
        tries = 0
        max_tries = 3

        while not succeeded and tries < max_tries:
            tries += 1
            print(f"Attempt {tries}/{max_tries}")

            try:
                if use_deepspeed:
                    # Run with DeepSpeed
                    cmd = [
                            "deepspeed",
                            "latex_ocr_simple_test.py",
                            model_type,
                            str(batch_size),
                            "true"
                    ]
                else:
                    # Run without DeepSpeed
                    cmd = [
                            "python",
                            "latex_ocr_simple_test.py",
                            model_type,
                            str(batch_size),
                            "false"
                    ]

                print(f"Running command: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                succeeded = True
                print(f"✓ Batch size {batch_size} completed successfully")

            except subprocess.CalledProcessError as exc:
                print(f"✗ Error occurred: {exc}")
                if tries < max_tries:
                    print(f"Retrying... ({tries}/{max_tries})")
                else:
                    print(f"Failed after {max_tries} attempts")

        # Check if we hit an OOM error
        try:
            with open(results_file, "r") as f:
                lines = f.readlines()
                if lines and "OOM" in lines[-1]:
                    print(f"\n🚫 Hit Out of Memory (OOM) at batch size {batch_size}")
                    print("Memory testing complete!")
                    break
        except FileNotFoundError:
            pass

        if not succeeded:
            print(f"\n🚫 Too many failures at batch size {batch_size}")
            print("Stopping memory testing.")
            break

    print(f"\n✅ Memory testing finished! Results saved to: {results_file}")


def analyze_results(model_type="gemma", use_deepspeed=False):
    """
    Analyze the results from the memory testing.
    """
    suffix = "_deepspeed" if use_deepspeed else "_simple"
    results_file = f"./results_{model_type}{suffix}.csv"

    try:
        with open(results_file, "r") as f:
            lines = f.readlines()

        mode_str = "DeepSpeed" if use_deepspeed else "Simple"
        print(f"\n📊 Analysis of {model_type} results ({mode_str} mode):")
        print("=" * 60)

        # Skip header
        data_lines = lines[1:]

        max_successful_batch = 0
        best_throughput = 0
        best_throughput_batch = 0

        for line in data_lines:
            if "OOM" in line:
                batch_size = int(line.split(",")[0])
                print(f"❌ OOM at batch size: {batch_size}")
                break
            else:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue

                batch_size = int(parts[0])
                max_successful_batch = batch_size

                # Get throughput (last column)
                try:
                    throughput = float(parts[-1])
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_throughput_batch = batch_size
                except (ValueError, IndexError):
                    throughput = 0.0

                # Calculate total memory usage across all GPUs
                total_active = 0
                total_reserved = 0
                gpu_count = 0

                # Skip batch_size column, then read GPU memory pairs
                for i in range(1, len(parts) - 1, 2):
                    if i + 1 < len(parts) - 1:  # Make sure we have both active and reserved
                        try:
                            total_active += int(parts[i])
                            total_reserved += int(parts[i + 1])
                            gpu_count += 1
                        except (ValueError, IndexError):
                            pass

                print(f"✅ Batch {batch_size}: {total_active}MB active, {total_reserved}MB reserved, {throughput:.2f} it/s")

        print(f"\n📈 Summary:")
        print(f"  • Maximum successful batch size: {max_successful_batch}")
        print(f"  • Best throughput: {best_throughput:.2f} it/s at batch size {best_throughput_batch}")
        print(f"  • Model: {model_type}")
        print(f"  • Mode: {mode_str}")

    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("Run the memory test first!")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Memory usage testing for LaTeX OCR models")
    parser.add_argument("--model", choices=["gemma", "qwen_vl"], default="gemma",
                        help="Model type to test (default: gemma)")
    parser.add_argument("--max-batch-size", type=int, default=None,
                        help="Maximum batch size to test")
    parser.add_argument("--deepspeed", action="store_true",
                        help="Use DeepSpeed for distributed training")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results instead of running tests")

    args = parser.parse_args()

    if args.analyze:
        analyze_results(args.model, args.deepspeed)
    else:
        run_memory_test(args.model, args.max_batch_size, args.deepspeed)



if __name__ == "__main__":
    main()