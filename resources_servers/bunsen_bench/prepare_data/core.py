#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare bunsen-bench data for the Gym exact-match environment.

This package ports the observable data-prep flow from the internal
`bunsen-bench` repo into the Gym environment:

- load Bunsen samples from HuggingFace by config
- render the canonical chemistry prompt shape
- emit a `test.jsonl` handoff artifact plus metadata

The package also keeps a small raw-row converter for ad hoc local JSONL files
and regenerates the checked-in example dataset for smoke tests.
"""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from datasets import load_dataset


try:
    from huggingface_hub import dataset_info as hf_dataset_info
except ImportError:  # pragma: no cover - `datasets` normally brings this in.
    hf_dataset_info = None


BUNSEN_HF_REPO_ENV = "BUNSEN_HF_REPO"
BUNSEN_HF_CONFIG_ENV = "BUNSEN_HF_CONFIG"
HF_TOKEN_ENV = "HF_TOKEN"

DEFAULT_HF_REPO = "nvidia/bunsen-bench-internal"
DEFAULT_HF_CONFIG = "mcq"
DEFAULT_HF_SPLIT = "test"
DEFAULT_CHOICE_SHUFFLE_SEED = 1337
DEFAULT_SYSTEM_PROMPT = "Answer the bunsen-bench task exactly. Return only the final answer."
DEFAULT_OUTPUT_FILENAME = "test.jsonl"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_EXAMPLE_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"

DEFAULT_BUNSEN_SAMPLES = [
    {
        "uuid": "bunsen-001",
        "problem": "Which apparatus is best for heating a small liquid sample directly over a flame?",
        "choices": ["Beaker", "Test tube", "Graduated cylinder", "Watch glass"],
        "expected_answer": "Test tube",
        "question_format": "mcq",
        "subset_for_metrics": "lab-equipment",
        "source": "bunsen-bench-stub",
        "source_subset": "example",
    },
    {
        "uuid": "bunsen-002",
        "problem": "What is the chemical symbol for sodium?",
        "expected_answer": "Na",
        "question_format": "open_ended",
        "subset_for_metrics": "chemistry",
        "source": "bunsen-bench-stub",
        "source_subset": "example",
    },
    {
        "uuid": "bunsen-003",
        "problem": "Which item should be worn to protect the eyes during a chemistry lab?",
        "choices": ["Safety goggles", "Wool gloves", "Sandals", "Paper mask"],
        "expected_answer": "Safety goggles",
        "question_format": "mcq",
        "subset_for_metrics": "lab-safety",
        "source": "bunsen-bench-stub",
        "source_subset": "example",
    },
    {
        "uuid": "bunsen-004",
        "problem": "Classify pure water at room temperature as acidic, basic, or neutral.",
        "expected_answer": "neutral",
        "question_format": "open_ended",
        "subset_for_metrics": "chemistry",
        "source": "bunsen-bench-stub",
        "source_subset": "example",
    },
    {
        "uuid": "bunsen-005",
        "problem": "Which flame condition is generally hottest in a properly adjusted Bunsen burner?",
        "choices": [
            "Yellow luminous flame",
            "Blue non-luminous flame",
            "Orange smoky flame",
            "Invisible pilot flame",
        ],
        "expected_answer": "Blue non-luminous flame",
        "question_format": "mcq",
        "subset_for_metrics": "lab-technique",
        "source": "bunsen-bench-stub",
        "source_subset": "example",
    },
]


def build_bunsen_prompt(
    problem_text: str,
    choices: list[str] | None = None,
    *,
    shuffle_choices: bool = False,
    shuffle_seed: int | None = None,
) -> tuple[str, list[str]]:
    """Render the visible prompt shape used by internal bunsen-bench."""

    rendered_choices = list(choices or [])
    if rendered_choices and shuffle_choices:
        if shuffle_seed is None:
            raise ValueError("shuffle_seed must be provided when shuffle_choices=True")
        rng = random.Random(shuffle_seed)
        rng.shuffle(rendered_choices)

    if rendered_choices:
        formatted_choices = "\n".join(f"<choice>{choice}</choice>" for choice in rendered_choices)
        return (
            f"{problem_text}\n"
            "<choices>\n"
            f"{formatted_choices}\n"
            "</choices>\n"
            "Respond with the correct choice in <choice></choice> tags, exactly as it is written above.",
            rendered_choices,
        )

    return f"{problem_text}\nRespond with the final answer in <answer></answer> tags.", []


def _normalize_bunsen_sample(
    sample: dict[str, Any],
    index: int,
    *,
    dataset_config: str,
    shuffle_mcq_choices: bool,
    choice_shuffle_seed: int,
) -> dict[str, Any]:
    problem_text = sample["problem"]
    expected_answer = sample["expected_answer"]
    question_format = sample.get("question_format", "mcq")

    choices = list(sample.get("choices") or [])
    is_mcq = question_format == "mcq" or (question_format != "open_ended" and bool(choices))
    if not is_mcq:
        choices = []
        question_format = "open_ended"

    prompt, rendered_choices = build_bunsen_prompt(
        problem_text,
        choices,
        shuffle_choices=shuffle_mcq_choices and bool(choices),
        shuffle_seed=choice_shuffle_seed if choices else None,
    )

    task_id = sample.get("uuid") or sample.get("task_id") or f"bunsen-{index:06d}"
    subset_for_metrics = sample.get("subset_for_metrics") or sample.get("bct_field", "")
    metadata = {
        key: deepcopy(value) for key, value in sample.items() if key not in {"problem", "expected_answer", "choices"}
    }
    metadata.update(
        {
            "problem": problem_text,
            "choices": rendered_choices,
            "question_format": question_format,
            "dataset_config": sample.get("dataset_config", dataset_config),
            "subset_for_metrics": subset_for_metrics,
            "source": sample.get("source", ""),
            "source_subset": sample.get("source_subset", ""),
            "dataset_sha": sample.get("dataset_sha", ""),
            "dataset_last_updated": sample.get("dataset_last_updated", ""),
            "uuid": sample.get("uuid", task_id),
        }
    )

    return {
        "task_id": task_id,
        "prompt": prompt,
        "expected_answer": expected_answer,
        "metadata": metadata,
        "choices": rendered_choices,
        "question_format": question_format,
        "subset_for_metrics": subset_for_metrics,
        "source": metadata["source"],
        "source_subset": metadata["source_subset"],
        "dataset_config": metadata["dataset_config"],
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
        },
    }


def convert_bunsen_samples(
    samples: list[dict[str, Any]],
    *,
    dataset_config: str = DEFAULT_HF_CONFIG,
    shuffle_mcq_choices: bool = True,
    choice_shuffle_seed: int = DEFAULT_CHOICE_SHUFFLE_SEED,
) -> list[dict[str, Any]]:
    return [
        _normalize_bunsen_sample(
            sample,
            index,
            dataset_config=dataset_config,
            shuffle_mcq_choices=shuffle_mcq_choices,
            choice_shuffle_seed=choice_shuffle_seed,
        )
        for index, sample in enumerate(samples, start=1)
    ]


def _gym_row_from_raw_row(
    row: dict[str, Any],
    index: int,
    *,
    system_prompt: str,
    prompt_field: str,
    answer_field: str,
    task_id_field: str,
    metadata_field: str,
) -> dict[str, Any]:
    prompt = row[prompt_field]
    expected_answer = row[answer_field]
    task_id = row.get(task_id_field) or f"bunsen-{index:03d}"
    metadata = row.get(metadata_field) or {}

    if not isinstance(metadata, dict):
        raise TypeError(f"Expected '{metadata_field}' to be an object, got {type(metadata).__name__}")

    return {
        "task_id": task_id,
        "prompt": prompt,
        "expected_answer": expected_answer,
        "metadata": metadata,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        },
    }


def convert_rows(
    rows: list[dict[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_field: str = "prompt",
    answer_field: str = "expected_answer",
    task_id_field: str = "task_id",
    metadata_field: str = "metadata",
) -> list[dict[str, Any]]:
    return [
        _gym_row_from_raw_row(
            row,
            index,
            system_prompt=system_prompt,
            prompt_field=prompt_field,
            answer_field=answer_field,
            task_id_field=task_id_field,
            metadata_field=metadata_field,
        )
        for index, row in enumerate(rows, start=1)
    ]


def _load_jsonl_rows(input_path: str | Path) -> list[dict[str, Any]]:
    with Path(input_path).open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows)


def _default_prepare_output_dir(config_name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "prepare" / config_name


def _get_dataset_info(repo_id: str) -> tuple[str, str]:
    if hf_dataset_info is None:
        return "", ""

    try:
        info = hf_dataset_info(repo_id)
    except Exception:
        return "", ""

    dataset_sha = info.sha or ""
    dataset_last_updated = info.last_modified.isoformat() if info.last_modified else ""
    return dataset_sha, dataset_last_updated


def prepare_raw_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_field: str = "prompt",
    answer_field: str = "expected_answer",
    task_id_field: str = "task_id",
    metadata_field: str = "metadata",
) -> int:
    raw_rows = _load_jsonl_rows(input_path)
    converted_rows = convert_rows(
        raw_rows,
        system_prompt=system_prompt,
        prompt_field=prompt_field,
        answer_field=answer_field,
        task_id_field=task_id_field,
        metadata_field=metadata_field,
    )
    return write_jsonl(converted_rows, output_path)


def prepare_data(*, output_path: str | Path = DEFAULT_EXAMPLE_OUTPUT_PATH) -> int:
    example_rows = convert_bunsen_samples(deepcopy(DEFAULT_BUNSEN_SAMPLES), dataset_config="example")
    return write_jsonl(example_rows, output_path)


def prepare_huggingface_data(
    *,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    config_name: str | None = None,
    split: str = DEFAULT_HF_SPLIT,
    shuffle_mcq_choices: bool = True,
    choice_shuffle_seed: int = DEFAULT_CHOICE_SHUFFLE_SEED,
    dry_run: bool = False,
) -> tuple[Path, int]:
    resolved_repo_id = repo_id or os.environ.get(BUNSEN_HF_REPO_ENV, DEFAULT_HF_REPO)
    resolved_config_name = config_name or os.environ.get(BUNSEN_HF_CONFIG_ENV, DEFAULT_HF_CONFIG)
    output_dirpath = Path(output_dir) if output_dir else _default_prepare_output_dir(resolved_config_name)
    output_path = output_dirpath / DEFAULT_OUTPUT_FILENAME

    dataset_sha, dataset_last_updated = _get_dataset_info(resolved_repo_id)
    dataset = load_dataset(
        resolved_repo_id,
        name=resolved_config_name,
        split=split,
        token=os.environ.get(HF_TOKEN_ENV),
    )

    normalized_samples = []
    for row in dataset:
        sample = dict(row)
        sample["dataset_config"] = resolved_config_name
        sample["dataset_sha"] = dataset_sha
        sample["dataset_last_updated"] = dataset_last_updated
        normalized_samples.append(sample)

    converted_rows = convert_bunsen_samples(
        normalized_samples,
        dataset_config=resolved_config_name,
        shuffle_mcq_choices=shuffle_mcq_choices,
        choice_shuffle_seed=choice_shuffle_seed,
    )

    if not dry_run:
        write_jsonl(converted_rows, output_path)
        output_dirpath.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset_repo": resolved_repo_id,
            "dataset_config": resolved_config_name,
            "dataset_sha": dataset_sha,
            "dataset_last_updated": dataset_last_updated,
            "split": split,
            "shuffle_mcq_choices": shuffle_mcq_choices,
            "choice_shuffle_seed": choice_shuffle_seed if shuffle_mcq_choices else None,
            "num_samples": len(converted_rows),
        }
        metadata_path = output_dirpath / DEFAULT_METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return output_path, len(converted_rows)
