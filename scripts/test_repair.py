"""
Test Repair Capability on Buggy Code Dataset

Use LLM to repair buggy code and verify if the修复 is correct.

Usage:
uv run python scripts/test_repair.py \
    --api-key "xxx" \
    --api-base "https://uni-api.cstcloud.cn/v1" \
    --model "gpt-oss-120b" \
    --input torch_bugcode_bench.jsonl \
    --output repair_results.jsonl \
    --backend cuda \
    --num-samples 10

 uv run python scripts/test_repair.py       --api-key "token-abc123"       --api-base "http://0.0.0.0:8002/v1"       --model base       --input torch_bugcode_bench.jsonl       --output repair_results.jsonl       --backend cuda       --num-samples 10
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernelbench.buggy_dataset import load_buggy_dataset, BuggySample
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.utils import query_llm, set_gpu_arch


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REPAIR_SYSTEM_PROMPT = """You are a GPU kernel programming expert. The following code has a bug that causes incorrect output. Your task is to find and fix the bug.

CRITICAL:
1. The buggy code compiles and runs, but produces WRONG results
2. Find the subtle bug and fix it
3. The class name must be "ModelNew" (NOT "Model")
4. Keep the same function signatures for get_inputs() and get_init_inputs()
5. Output ONLY the corrected code, no explanation"""

REPAIR_USER_TEMPLATE = """Here is a buggy GPU kernel code:

```python
{buggy_code}
```

Your task: Fix the bug in this code. The code should compile and produce CORRECT output.

IMPORTANT:
- Class name must be "ModelNew"
- Keep get_inputs() and get_init_inputs() unchanged
- Output the complete corrected code"""


def build_repair_prompt(buggy_code: str) -> tuple[str, str]:
    """Build system and user prompts for repair."""
    return REPAIR_SYSTEM_PROMPT, REPAIR_USER_TEMPLATE.format(buggy_code=buggy_code)


def extract_code_from_response(response: str) -> str:
    """Extract code from LLM response, handling markdown code blocks."""
    json_str = response.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])
    elif json_str.startswith("```python"):
        lines = json_str.split("\n")
        json_str = "\n".join(lines[1:-1])
    return json_str


def test_single_repair(
    sample: BuggySample,
    dataset,
    api_key: str,
    api_base: str,
    model: str,
    backend: str,
    precision: str,
    max_retries: int = 3,
    retry_interval: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Test repair for a single sample."""
    # Build prompt
    system_prompt, user_prompt = build_repair_prompt(sample.buggy_code)

    if verbose:
        print(f"  [LLM] Calling repair API...")

    # Call LLM
    try:
        response = query_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,  # Use deterministic for repair
            max_tokens=8192,
            api_key=api_key,
            api_base=api_base,
            model=model,
            max_retries=max_retries,
            retry_interval=retry_interval,
        )
    except Exception as e:
        if verbose:
            print(f"  [ERROR] LLM call failed: {e}")
        return {
            "problem_id": sample.problem_id,
            "level": sample.level,
            "bug_type": sample.bug_type,
            "buggy_code": sample.buggy_code,
            "fixed_code": None,
            "fix_success": False,
            "error": str(e),
        }

    # Extract fixed code
    fixed_code = extract_code_from_response(response)

    if verbose:
        print(f"  [EVAL] Verifying fix...")

    # Get reference code
    try:
        problem = dataset.get_problem_by_id(sample.problem_id)
        ref_arch_src = problem.code
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Failed to get reference: {e}")
        return {
            "problem_id": sample.problem_id,
            "level": sample.level,
            "bug_type": sample.bug_type,
            "buggy_code": sample.buggy_code,
            "fixed_code": fixed_code,
            "fix_success": False,
            "error": f"Failed to get reference: {e}",
        }

    # Verify the fix
    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=fixed_code,
            num_correct_trials=3,
            measure_performance=False,
            backend=backend,
            precision=get_torch_dtype_from_string(precision),
            verbose=False,
        )

        if eval_result is None:
            fix_success = False
            error = "eval_result is None"
        else:
            fix_success = eval_result.compiled and eval_result.correctness
            error = None if fix_success else f"compiled={eval_result.compiled}, correct={eval_result.correctness}"

    except Exception as e:
        fix_success = False
        error = str(e)

    if verbose:
        status = "FIXED" if fix_success else "FAILED"
        print(f"  [{status}] fix_success={fix_success}")

    return {
        "problem_id": sample.problem_id,
        "level": sample.level,
        "bug_type": sample.bug_type,
        "buggy_code": sample.buggy_code,
        "fixed_code": fixed_code,
        "fix_success": fix_success,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description="Test repair capability on buggy dataset")
    parser.add_argument("--api-key", required=True, help="API key for LLM service")
    parser.add_argument("--api-base", required=True, help="API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--input", required=True, help="Input JSONL (buggy dataset)")
    parser.add_argument("--output", default="repair_results.jsonl", help="Output JSONL")
    parser.add_argument("--backend", default="cuda", help="Backend")
    parser.add_argument("--gpu-arch", default="Ada", help="GPU architecture")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to test (default: all)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max LLM retries")
    parser.add_argument("--retry-interval", type=float, default=1.0, help="Retry interval")
    parser.add_argument("--api-query-interval", type=float, default=1.0, help="Interval between API queries")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU verification (for testing without GPU)")

    args = parser.parse_args()

    # Resolve paths
    input_path = args.input if os.path.isabs(args.input) else os.path.join(REPO_TOP_DIR, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_TOP_DIR, args.output)

    # Set GPU
    if not args.skip_gpu and torch.cuda.is_available():
        set_gpu_arch([args.gpu_arch])
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif args.skip_gpu:
        print("Skipping GPU check")
    else:
        print("ERROR: No GPU available")
        sys.exit(1)

    # Load dataset
    print(f"Loading buggy dataset: {input_path}")
    all_samples = load_buggy_dataset(input_path)
    print(f"Loaded {len(all_samples)} samples")

    # Limit samples if requested
    samples = all_samples[:args.num_samples] if args.num_samples else all_samples
    print(f"Testing {len(samples)} samples")

    # Load reference datasets
    datasets = {}
    for level in [1, 2, 3]:
        try:
            datasets[level] = construct_kernelbench_dataset(level=level, source="local")
        except:
            pass

    # Test each sample
    results = []
    fix_count = 0

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Problem {sample.problem_id} (Level {sample.level}, {sample.bug_type})")

        if sample.level not in datasets:
            print(f"  [SKIP] No dataset for level {sample.level}")
            continue

        if args.skip_gpu:
            # Skip actual GPU verification
            print(f"  [SKIP] GPU verification skipped")
            results.append({
                "problem_id": sample.problem_id,
                "level": sample.level,
                "bug_type": sample.bug_type,
                "buggy_code": sample.buggy_code,
                "fixed_code": "SKIPPED",
                "fix_success": False,
                "error": "GPU verification skipped",
            })
        else:
            result = test_single_repair(
                sample,
                datasets[sample.level],
                args.api_key,
                args.api_base,
                args.model,
                args.backend,
                args.precision,
                args.max_retries,
                args.retry_interval,
                args.verbose,
            )
            results.append(result)

            if result["fix_success"]:
                fix_count += 1

        # Rate limiting
        if i < len(samples) - 1:
            time.sleep(args.api_query_interval)

    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Compute statistics
    print(f"\n{'='*60}")
    print(f"Test Complete!")
    print(f"Total: {len(results)}")
    print(f"Fixed: {fix_count}")
    print(f"Fix Rate: {fix_count/len(results):.2%}")

    # Per bug type
    print(f"\nPer Bug Type:")
    by_bug_type = defaultdict(lambda: {"total": 0, "fixed": 0})
    for r in results:
        by_bug_type[r["bug_type"]]["total"] += 1
        if r["fix_success"]:
            by_bug_type[r["bug_type"]]["fixed"] += 1

    for bug_type, stats in sorted(by_bug_type.items()):
        rate = stats["fixed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {bug_type}: {stats['fixed']}/{stats['total']} ({rate:.1%})")

    # Per level
    print(f"\nPer Level:")
    by_level = defaultdict(lambda: {"total": 0, "fixed": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        if r["fix_success"]:
            by_level[r["level"]]["fixed"] += 1

    for level, stats in sorted(by_level.items()):
        rate = stats["fixed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  Level {level}: {stats['fixed']}/{stats['total']} ({rate:.1%})")


if __name__ == "__main__":
    main()
