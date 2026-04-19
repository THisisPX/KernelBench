"""
Buggy Code Repair Dataset

Utilities for generating and managing a dataset of buggy GPU kernel code
for code repair training/evaluation.
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import Optional


# Bug type taxonomy with weights for stratified sampling
BUG_TYPE_WEIGHTS = [
    ("arithmetic_incorrectness", 0.30),
    ("indexing_error", 0.20),
    ("memory_access_violation", 0.15),
    ("dtype_mismatch", 0.10),
    ("boundary_condition", 0.10),
    ("algorithm_wrong", 0.10),
    ("operator_fusion_error", 0.03),
    ("initialization_error", 0.02),
]

BUG_TYPES = [bt for bt, _ in BUG_TYPE_WEIGHTS]


@dataclass
class BuggySample:
    """Single sample in the buggy code repair dataset."""
    problem_id: int
    level: int
    problem_name: str
    buggy_code: str
    bug_type: str
    bug_description: str
    expected_behavior: str
    backend: str
    generation_model: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "BuggySample":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "BuggySample":
        return cls.from_dict(json.loads(s))


def sample_bug_type() -> str:
    """Sample a bug type based on predefined weights."""
    r = random.random()
    cumulative = 0
    for bug_type, weight in BUG_TYPE_WEIGHTS:
        cumulative += weight
        if r < cumulative:
            return bug_type
    return "arithmetic_incorrectness"  # fallback


def get_bug_type_list() -> str:
    """Get formatted list of bug types for prompt injection."""
    return ", ".join(BUG_TYPES)


def load_buggy_dataset(jsonl_path: str) -> list[BuggySample]:
    """Load a buggy dataset from JSONL file."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(BuggySample.from_json(line))
    return samples


def save_buggy_dataset(samples: list[BuggySample], jsonl_path: str):
    """Save a buggy dataset to JSONL file."""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")


def append_buggy_sample(sample: BuggySample, jsonl_path: str):
    """Append a single sample to a JSONL file."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(sample.to_json() + "\n")
