"""
Split Buggy Dataset into Train/Test Sets

Usage:
uv run python scripts/split_train_test.py \
    --input torch_repair_verified.jsonl \
    --output-prefix torch_repair \
    --train-ratio 0.8
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernelbench.buggy_dataset import load_buggy_dataset, BuggySample


def main():
    parser = argparse.ArgumentParser(description="Split buggy dataset into train/test")
    parser.add_argument("--input", required=True, help="Input JSONL (verified buggy dataset)")
    parser.add_argument("--output-prefix", required=True, help="Output prefix, e.g., 'torch_repair'")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading: {args.input}")
    samples = load_buggy_dataset(args.input)
    print(f"Loaded {len(samples)} samples")

    # Shuffle with seed
    random.seed(args.seed)
    random.shuffle(samples)

    # Split
    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    # Save train
    train_path = f"{args.output_prefix}_train.jsonl"
    test_path = f"{args.output_prefix}_test.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(sample.to_json() + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(sample.to_json() + "\n")

    print(f"\n{'='*60}")
    print(f"Train: {len(train_samples)} ({len(train_samples)/len(samples):.1%}) -> {train_path}")
    print(f"Test:  {len(test_samples)} ({len(test_samples)/len(samples):.1%}) -> {test_path}")

    # Statistics
    if args.verbose:
        print(f"\n--- Train Bug Type Distribution ---")
        train_bug_types = Counter(s.bug_type for s in train_samples)
        for bt, cnt in train_bug_types.most_common():
            print(f"  {bt}: {cnt}")

        print(f"\n--- Test Bug Type Distribution ---")
        test_bug_types = Counter(s.bug_type for s in test_samples)
        for bt, cnt in test_bug_types.most_common():
            print(f"  {bt}: {cnt}")

        print(f"\n--- Train Level Distribution ---")
        train_levels = Counter(s.level for s in train_samples)
        for lvl, cnt in sorted(train_levels.items()):
            print(f"  Level {lvl}: {cnt}")

        print(f"\n--- Test Level Distribution ---")
        test_levels = Counter(s.level for s in test_samples)
        for lvl, cnt in sorted(test_levels.items()):
            print(f"  Level {lvl}: {cnt}")


if __name__ == "__main__":
    main()
