"""
Verify and Filter Buggy Code Dataset

Read buggy code from JSONL, verify each sample using eval_kernel_against_ref(),
and output only the samples that compile but produce incorrect results.

Usage:
uv run python scripts/verify_and_filter.py \
    --input torch_bugcode_bench_raw.jsonl \
    --output torch_bugcode_bench.jsonl \
    --backend cuda \
    --gpu-arch Ada \
    --precision fp32
"""

import argparse
import json
import os
import sys

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernelbench.buggy_dataset import BuggySample, append_buggy_sample
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.utils import set_gpu_arch


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw_dataset(jsonl_path: str) -> list[BuggySample]:
    """Load buggy samples from JSONL file."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(BuggySample.from_json(line))
    return samples


def verify_single_sample(
    sample: BuggySample,
    dataset,
    backend: str,
    precision: str,
    verbose: bool = False,
) -> tuple[bool, bool, dict]:
    """
    Verify a single buggy sample.

    Returns:
        (compiled, verified, metadata)
        - compiled: whether the code compiled successfully
        - verified: whether compiled=True and correctness=False (truly buggy)
        - metadata: eval result metadata
    """
    # Get reference code from dataset
    try:
        problem = dataset.get_problem_by_id(sample.problem_id)
        ref_arch_src = problem.code
    except Exception as e:
        return False, False, {"error": f"Failed to get reference code: {e}"}

    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=sample.buggy_code,
            num_correct_trials=3,
            measure_performance=False,
            backend=backend,
            precision=get_torch_dtype_from_string(precision),
            verbose=verbose,
        )

        if eval_result is None:
            return False, False, {"error": "eval_result is None"}

        compiled = eval_result.compiled
        correctness = eval_result.correctness

        # A valid buggy sample: compiles but is NOT correct
        verified = compiled and not correctness

        return compiled, verified, {
            "compiled": compiled,
            "correctness": correctness,
            "metadata": eval_result.metadata,
        }

    except Exception as e:
        return False, False, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Verify and filter buggy code dataset"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL path (raw buggy dataset)"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL path (verified dataset)"
    )
    parser.add_argument(
        "--backend", default="cuda", help="Backend (cuda, triton, etc.)"
    )
    parser.add_argument(
        "--gpu-arch", default="Ada", help="GPU architecture"
    )
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "fp16", "bf16"], help="Precision"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--dataset-level", type=int, default=1, help="KernelBench level for reference dataset"
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = args.input if os.path.isabs(args.input) else os.path.join(REPO_TOP_DIR, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_TOP_DIR, args.output)

    # Set GPU architecture
    if torch.cuda.is_available():
        set_gpu_arch([args.gpu_arch])
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ERROR: No GPU available. This script requires CUDA.")
        sys.exit(1)

    # Load raw dataset
    print(f"Loading raw dataset from: {input_path}")
    samples = load_raw_dataset(input_path)
    print(f"Loaded {len(samples)} samples")

    # Get reference dataset (we need this to get the original code for each problem)
    # Since problems can be from different levels, we construct datasets for all levels
    datasets = {}
    for level in [1, 2, 3]:
        try:
            datasets[level] = construct_kernelbench_dataset(level=level, source="local")
        except:
            pass

    # Verify each sample
    verified_count = 0
    compiled_count = 0
    error_count = 0

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Verifying problem {sample.problem_id} (level {sample.level})...")

        if sample.level not in datasets:
            print(f"  [SKIP] No dataset for level {sample.level}")
            continue

        compiled, verified, meta = verify_single_sample(
            sample,
            datasets[sample.level],
            args.backend,
            args.precision,
            args.verbose,
        )

        if verified:
            # Append to output
            append_buggy_sample(sample, output_path)
            verified_count += 1
            compiled_count += 1
            print(f"  [OK] Verified buggy: {sample.bug_type}")
        elif compiled:
            # Compiles but is correct (not buggy)
            compiled_count += 1
            print(f"  [FAIL] Compiles but is CORRECT (not buggy)")
            if args.verbose:
                print(f"       Details: {meta}")
        else:
            # Compilation error
            error_count += 1
            print(f"  [ERROR] Compilation failed")
            if args.verbose:
                print(f"       Details: {meta}")

    print(f"\n{'='*60}")
    print(f"Verification complete!")
    print(f"Verified (compiled + incorrect): {verified_count}")
    print(f"Compiled but correct: {compiled_count - verified_count}")
    print(f"Failed to compile: {error_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
