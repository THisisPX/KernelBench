"""
Generate Buggy Code Repair Dataset

Generate a dataset of buggy GPU kernel code for code repair training/evaluation.

Usage:
uv run python scripts/generate_buggy_dataset.py \
    --api-key "YOUR_API_KEY" \
    --api-base "https://uni-api.cstcloud.cn/v1" \
    --model "gpt-oss-120b" \
    --dataset-src local \
    --num-samples 50 \
    --run-name buggy_dataset_v1 \
    --backend cuda \
    --max-retries 3 \
    --retry-interval 1.0 \
    --api-query-interval 1.0
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

# Disable LiteLLM remote model cost map fetch (causes timeout in China)
os.environ["LITELLM_FAIL_ON_MODEL_MISMATCH"] = "False"
os.environ["LITELLM_MODEL_BUMP_TRIGGER"] = "None"

import torch
import tomli

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernelbench.buggy_dataset import (
    BuggySample,
    BUG_TYPE_WEIGHTS,
    BUG_TYPES,
    append_buggy_sample,
    sample_bug_type,
)
from kernelbench.dataset import construct_kernelbench_dataset, get_representative_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.utils import query_llm, set_gpu_arch


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class GenerationConfig:
    api_key: str
    api_base: str
    model: str
    dataset_src: str = "local"
    num_samples: int = 50
    run_name: str = "buggy_dataset"
    backend: str = "cuda"
    max_retries: int = 3
    retry_interval: float = 1.0
    api_query_interval: float = 1.0
    level: int = None  # If None, sample from all levels
    gpu_arch: str = "Ada"
    precision: str = "fp32"
    verbose: bool = True
    skip_verification: bool = True  # Default True: generation only, no GPU needed


def load_prompt_templates() -> dict:
    """Load buggy generation prompt templates from TOML."""
    prompts_path = os.path.join(
        REPO_TOP_DIR, "src/kernelbench/prompts/buggy_generation_prompts.toml"
    )
    with open(prompts_path, "rb") as f:
        return tomli.load(f)


def build_bug_type_description(bug_type: str) -> str:
    """Get a description hint for the bug type."""
    prompts = load_prompt_templates()
    bug_types_section = prompts.get("bug_types", {})
    if bug_type in bug_types_section:
        examples = bug_types_section[bug_type].get("examples", [])
        if examples:
            return random.choice(examples)
    return ""


def build_prompts(ref_arch_src: str, bug_type: str, bug_description_hint: str) -> tuple[str, str]:
    """Build system and user prompts for buggy code generation."""
    prompts = load_prompt_templates()
    templates = prompts.get("templates", {})

    system_template = templates.get("system_prompt", "")
    user_template = templates.get("user_prompt", "")

    # Format bug type list for system prompt
    bug_type_list = ", ".join(BUG_TYPES)

    # Build system prompt
    system_prompt = system_template.format(bug_type_list=bug_type_list)

    # Build user prompt
    user_prompt = user_template.format(
        ref_arch_src=ref_arch_src,
        bug_type=bug_type,
        bug_description_hint=bug_description_hint,
    )

    return system_prompt, user_prompt


def sample_problems(
    dataset_src: str, num_samples: int, level: int = None
) -> list:
    """Sample problems from KernelBench dataset."""
    if level is not None:
        # Sample from specific level
        if dataset_src == "local":
            dataset = get_representative_dataset(level, source="local")
        else:
            dataset = get_representative_dataset(level, source="huggingface")
        problems = list(dataset)
        if len(problems) > num_samples:
            problems = random.sample(problems, num_samples)
        return problems

    # Sample across all levels
    all_problems = []
    levels = [1, 2, 3] if level is None else [level]

    if num_samples <= len(levels):
        # If we want fewer samples than levels, just pick one level and sample from it
        lvl = random.choice(levels)
        if dataset_src == "local":
            dataset = get_representative_dataset(lvl, source="local")
        else:
            dataset = get_representative_dataset(lvl, source="huggingface")
        problems = list(dataset)
        if len(problems) >= num_samples:
            problems = random.sample(problems, num_samples)
        return problems

    # Sample across all levels (15-20 per level)
    per_level = num_samples // len(levels)
    remainder = num_samples % len(levels)

    for i, lvl in enumerate(levels):
        if dataset_src == "local":
            dataset = get_representative_dataset(lvl, source="local")
        else:
            dataset = get_representative_dataset(lvl, source="huggingface")
        problems = list(dataset)
        # Sample up to per_level from each level
        sample_count = per_level + (1 if i < remainder else 0)
        if len(problems) > sample_count:
            problems = random.sample(problems, sample_count)
        elif len(problems) > 0:
            problems = random.sample(problems, min(sample_count, len(problems)))
        all_problems.extend(problems)

    return all_problems


def generate_buggy_code(
    ref_arch_src: str,
    bug_type: str,
    config: GenerationConfig,
) -> dict:
    """Generate buggy code using LLM API."""
    # Build description hint
    bug_description_hint = build_bug_type_description(bug_type)

    # Build prompts
    system_prompt, user_prompt = build_prompts(ref_arch_src, bug_type, bug_description_hint)

    if config.verbose:
        print(f"[Generate] Bug type: {bug_type}")
        print(f"[Generate] Description hint: {bug_description_hint}")

    # Call LLM API
    response = query_llm(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=8192,
        api_key=config.api_key,
        api_base=config.api_base,
        model=config.model,
        max_retries=config.max_retries,
        retry_interval=config.retry_interval,
    )

    # Parse JSON response
    try:
        # Try to extract JSON from response
        json_str = response.strip()
        # Handle case where response might have markdown code blocks
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1])  # Remove first and last line
        elif json_str.startswith("```json"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1])

        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        if config.verbose:
            print(f"[Generate] JSON parsing error: {e}")
            print(f"[Generate] Raw response: {response[:500]}...")
        raise


def verify_buggy_sample(
    ref_arch_src: str,
    buggy_code: str,
    config: GenerationConfig,
) -> tuple[bool, bool, dict]:
    """
    Verify that buggy code compiles but produces incorrect results.

    Returns:
        (compiled, verified, metadata)
        - compiled: whether the code compiled successfully
        - verified: whether compiled=True and correctness=False (truly buggy)
        - metadata: eval result metadata
    """
    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=buggy_code,
            num_correct_trials=3,
            measure_performance=False,
            backend=config.backend,
            precision=get_torch_dtype_from_string(config.precision),
            verbose=config.verbose,
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


def generate_single_sample(
    problem,
    config: GenerationConfig,
) -> BuggySample | None:
    """Generate and verify a single buggy sample."""
    ref_arch_src = problem.code
    problem_id = problem.problem_id
    problem_name = problem.name
    level = problem.level

    # Sample bug type
    bug_type = sample_bug_type()

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"[Problem {problem_id}] Level {level}: {problem_name}")
        print(f"[Problem {problem_id}] Bug type: {bug_type}")

    # Generate buggy code
    try:
        result = generate_buggy_code(ref_arch_src, bug_type, config)
        buggy_code = result.get("buggy_code", "")
        bug_description = result.get("bug_description", "")
        expected_behavior = result.get("expected_behavior", "")

        if not buggy_code:
            if config.verbose:
                print(f"[Problem {problem_id}] ERROR: Empty buggy_code in response")
            return None

    except Exception as e:
        if config.verbose:
            print(f"[Problem {problem_id}] ERROR: Generation failed: {e}")
        return None

    # Verify the buggy code (skip if no GPU available)
    if config.skip_verification:
        if config.verbose:
            print(f"[Problem {problem_id}] Skipping verification (--skip-verification enabled)")
        verified = True  # Trust the LLM generated buggy code
    else:
        compiled, verified, verification_meta = verify_buggy_sample(
            ref_arch_src, buggy_code, config
        )

        if config.verbose:
            print(f"[Problem {problem_id}] Compilation: {compiled}, Verified buggy: {verified}")
            if not verified:
                print(f"[Problem {problem_id}] Verification details: {verification_meta}")

        if not verified:
            return None

    # Create BuggySample
    sample = BuggySample(
        problem_id=problem_id,
        level=level,
        problem_name=problem_name,
        buggy_code=buggy_code,
        bug_type=bug_type,
        bug_description=bug_description,
        expected_behavior=expected_behavior,
        backend=config.backend,
        generation_model=config.model,
    )

    return sample


def main():
    parser = argparse.ArgumentParser(
        description="Generate Buggy Code Repair Dataset"
    )
    parser.add_argument(
        "--api-key", required=True, help="API key for LLM service"
    )
    parser.add_argument(
        "--api-base",
        required=True,
        help="API base URL (e.g., https://uni-api.cstcloud.cn/v1)",
    )
    parser.add_argument(
        "--model", required=True, help="Model name (e.g., gpt-oss-120b)"
    )
    parser.add_argument(
        "--dataset-src",
        default="local",
        choices=["local", "huggingface"],
        help="Dataset source",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50, help="Number of samples to generate"
    )
    parser.add_argument(
        "--run-name", default="buggy_dataset", help="Name for this run"
    )
    parser.add_argument(
        "--backend", default="cuda", help="Backend (cuda, triton, etc.)"
    )
    parser.add_argument(
        "--level", type=int, default=None, help="Specific level (1, 2, or 3). If None, sample from all."
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries for API calls"
    )
    parser.add_argument(
        "--retry-interval", type=float, default=1.0, help="Retry interval in seconds"
    )
    parser.add_argument(
        "--api-query-interval", type=float, default=1.0, help="Interval between API queries"
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
        "--skip-verification", action="store_true", default=True, help="Skip GPU verification (default: True, generation only)"
    )
    parser.add_argument(
        "--no-skip-verification", dest="skip_verification", action="store_false", help="Enable GPU verification (requires GPU)"
    )
    parser.add_argument(
        "--output", default=None, help="Output JSONL path (default: {run_name}_raw.jsonl)"
    )

    args = parser.parse_args()

    config = GenerationConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        dataset_src=args.dataset_src,
        num_samples=args.num_samples,
        run_name=args.run_name,
        backend=args.backend,
        max_retries=args.max_retries,
        retry_interval=args.retry_interval,
        api_query_interval=args.api_query_interval,
        level=args.level,
        gpu_arch=args.gpu_arch,
        precision=args.precision,
        verbose=args.verbose,
        skip_verification=args.skip_verification,
    )

    # Set GPU architecture
    if torch.cuda.is_available():
        set_gpu_arch([config.gpu_arch])

    # Output file
    if args.output:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_TOP_DIR, args.output)
    else:
        output_path = os.path.join(REPO_TOP_DIR, f"{config.run_name}_raw.jsonl")

    print(f"Output: {output_path}")

    # Sample problems
    print(f"Sampling {config.num_samples} problems from KernelBench...")
    problems = sample_problems(config.dataset_src, config.num_samples, config.level)
    print(f"Sampled {len(problems)} problems")

    # Generate and verify samples
    generated = 0
    failed = 0

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Processing...")

        sample = generate_single_sample(problem, config)

        if sample is not None:
            # Append to JSONL
            append_buggy_sample(sample, output_path)
            generated += 1
            print(f"[OK] Generated and verified: {sample.bug_type}")
        else:
            failed += 1
            print(f"[FAIL] Generation or verification failed")

        # Rate limiting
        if i < len(problems) - 1:
            time.sleep(config.api_query_interval)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Generated: {generated}")
    print(f"Failed: {failed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
