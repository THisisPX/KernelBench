"""
Generate Full Buggy Dataset for torch-repair

Generate buggy code for all 250 KernelBench problems with diverse bug types.

Usage:
# Generate batch 1 (problems 1-50, 4 bugs each = 200 samples)
uv run python scripts/generate_full_dataset.py \
    --api-key "xxx" --api-base "https://uni-api.cstcloud.cn/v1" --model "gpt-oss-120b" \
    --problem-ids 1-50 --bugs-per-problem 4 \
    --output torch_repair_batch1.jsonl

# After all batches:
cat torch_repair_batch*.jsonl > torch_repair_raw.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernelbench.buggy_dataset import (
    BuggySample,
    BUG_TYPES,
    append_buggy_sample,
)
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import query_llm, set_gpu_arch
import tomli


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class GenerationConfig:
    api_key: str
    api_base: str
    model: str
    dataset_src: str = "local"
    backend: str = "cuda"
    max_retries: int = 3
    retry_interval: float = 1.0
    api_query_interval: float = 1.0
    gpu_arch: str = "Ada"
    precision: str = "fp32"
    verbose: bool = True


def get_bug_types_for_problem(problem_id: int, bugs_per_problem: int = 4) -> list:
    """
    Assign diverse bug types to a problem.
    Uses round-robin with shuffling to ensure diversity.
    """
    # Shuffle bug types for this problem based on problem_id as seed
    random.seed(problem_id * 12345)
    shuffled = BUG_TYPES.copy()
    random.shuffle(shuffled)

    # Take bugs_per_problem consecutive bug types (wrapping around)
    start_idx = problem_id % len(BUG_TYPES)
    result = []
    for i in range(bugs_per_problem):
        idx = (start_idx + i) % len(BUG_TYPES)
        result.append(shuffled[idx])

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for bt in result:
        if bt not in seen:
            seen.add(bt)
            unique.append(bt)

    # If still not enough, fill with remaining bug types
    if len(unique) < bugs_per_problem:
        for bt in shuffled:
            if bt not in seen:
                unique.append(bt)
                if len(unique) >= bugs_per_problem:
                    break

    return unique[:bugs_per_problem]


def load_prompt_templates() -> dict:
    """Load buggy generation prompt templates from TOML."""
    prompts_path = os.path.join(
        REPO_TOP_DIR, "src/kernelbench/prompts/buggy_generation_prompts.toml"
    )
    with open(prompts_path, "rb") as f:
        return tomli.load(f)


def get_bug_description_hint(bug_type: str) -> str:
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

    bug_type_list = ", ".join(BUG_TYPES)
    system_prompt = system_template.format(bug_type_list=bug_type_list)
    user_prompt = user_template.format(
        ref_arch_src=ref_arch_src,
        bug_type=bug_type,
        bug_description_hint=bug_description_hint,
    )

    return system_prompt, user_prompt


def generate_single_buggy(
    problem,
    bug_type: str,
    config: GenerationConfig,
) -> BuggySample | None:
    """Generate a single buggy sample for a problem with a specific bug type."""
    ref_arch_src = problem.code
    problem_id = problem.problem_id
    problem_name = problem.name
    level = problem.level

    bug_description_hint = get_bug_description_hint(bug_type)

    if config.verbose:
        print(f"  Bug type: {bug_type} - {bug_description_hint[:50]}...")

    # Build prompts
    system_prompt, user_prompt = build_prompts(ref_arch_src, bug_type, bug_description_hint)

    # Call LLM
    try:
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
    except Exception as e:
        if config.verbose:
            print(f"  [ERROR] LLM call failed: {e}")
        return None

    # Parse JSON response
    try:
        json_str = response.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1])
        elif json_str.startswith("```json"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1])

        result = json.loads(json_str)
        buggy_code = result.get("buggy_code", "")
        bug_description = result.get("bug_description", "")
        expected_behavior = result.get("expected_behavior", "")

        if not buggy_code:
            return None

    except json.JSONDecodeError as e:
        if config.verbose:
            print(f"  [ERROR] JSON parse failed: {e}")
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


def parse_problem_ids(problem_ids_str: str) -> list[int]:
    """Parse problem IDs from string like '1-50' or '1,3,5' or '1-10,20,30-40'."""
    result = []
    parts = problem_ids_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description="Generate full buggy dataset")
    parser.add_argument("--api-key", required=True, help="API key for LLM service")
    parser.add_argument("--api-base", required=True, help="API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--problem-ids", required=True, help="Problem IDs, e.g., '1-50' or '1,3,5' or '1-10,20,30-40'")
    parser.add_argument("--level", type=int, default=None, help="Specific level to generate for (1, 2, 3, or 4). If not specified, generates across all levels.")
    parser.add_argument("--bugs-per-problem", type=int, default=4, help="Number of bug types per problem")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--dataset-src", default="local", choices=["local", "huggingface"])
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    parser.add_argument("--api-query-interval", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = GenerationConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        dataset_src=args.dataset_src,
        backend=args.backend,
        max_retries=args.max_retries,
        retry_interval=args.retry_interval,
        api_query_interval=args.api_query_interval,
        verbose=args.verbose,
    )

    # Parse problem IDs
    problem_ids = parse_problem_ids(args.problem_ids)
    print(f"Generating for {len(problem_ids)} problems, {args.bugs_per_problem} bugs each")
    print(f"Expected output: {len(problem_ids) * args.bugs_per_problem} samples")

    # Load datasets - either specific level or all levels
    datasets = {}
    levels_to_load = [args.level] if args.level else [1, 2, 3]
    for level in levels_to_load:
        try:
            datasets[level] = construct_kernelbench_dataset(level=level, source=args.dataset_src)
        except Exception as e:
            if args.verbose:
                print(f"Warning: Could not load level {level}: {e}")

    # Generate samples
    total_generated = 0
    total_failed = 0

    for pid in problem_ids:
        # Find which level contains this problem
        problem = None
        problem_level = None
        for level, dataset in datasets.items():
            if pid in dataset.get_problem_ids():
                problem = dataset.get_problem_by_id(pid)
                problem_level = level
                break

        if problem is None:
            print(f"Problem {pid}: NOT FOUND in any level, skipping")
            continue

        # Get bug types for this problem
        bug_types = get_bug_types_for_problem(pid, args.bugs_per_problem)
        print(f"\n[{pid}] Level {problem_level}: {problem.name}")
        print(f"  Assigned bug types: {bug_types}")

        for i, bug_type in enumerate(bug_types):
            print(f"  [{i+1}/{len(bug_types)}] Generating {bug_type}...")

            sample = generate_single_buggy(problem, bug_type, config)

            if sample is not None:
                append_buggy_sample(sample, args.output)
                total_generated += 1
                print(f"    [OK] Generated")
            else:
                total_failed += 1
                print(f"    [FAIL] Failed")

            # Rate limiting
            time.sleep(config.api_query_interval)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Generated: {total_generated}")
    print(f"Failed: {total_failed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
