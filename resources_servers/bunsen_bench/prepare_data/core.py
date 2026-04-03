#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare bunsen-bench data for the Gym exact-match environment.

This package ports the upstream-sourcing part of the internal bunsen-bench
pipeline into the Gym environment:

- load and transform the original source datasets from HuggingFace
- optionally derive the open-ended split from an annotation JSONL
- render the canonical chemistry prompt shape
- emit a `test.jsonl` handoff artifact plus metadata

The package also keeps a small raw-row converter for ad hoc local JSONL files
and regenerates the checked-in example dataset for smoke tests.
"""

from __future__ import annotations

import hashlib
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


BUNSEN_HF_CONFIG_ENV = "BUNSEN_HF_CONFIG"
BUNSEN_OPEN_ENDED_ANNOTATION_PATH_ENV = "BUNSEN_OPEN_ENDED_ANNOTATION_PATH"
HF_TOKEN_ENV = "HF_TOKEN"

DEFAULT_HF_CONFIG = "mcq"
OPEN_ENDED_CONFIG = "open_ended"
DEFAULT_OPEN_ENDED_ANNOTATION_PATH = "annotation-open-ended.jsonl"
DEFAULT_CHOICE_SHUFFLE_SEED = 1337
DEFAULT_SYSTEM_PROMPT = "Answer the bunsen-bench task exactly. Return only the final answer."
DEFAULT_OUTPUT_FILENAME = "test.jsonl"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_EXAMPLE_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"

CHEMBENCH_SUBSETS = [
    "analytical_chemistry",
    "chemical_preference",
    "general_chemistry",
    "inorganic_chemistry",
    "materials_science",
    "organic_chemistry",
    "physical_chemistry",
    "technical_chemistry",
    "toxicity_and_safety",
]

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


def _get_hf_token() -> str | None:
    return os.environ.get(HF_TOKEN_ENV)


def _compute_problem_hash(problem_text: str) -> str:
    return hashlib.sha256(problem_text.encode("utf-8")).hexdigest()


def _stable_source_uuid(dataset_name: str, source_subset: str, source_id: str, problem_text: str) -> str:
    stable_seed = "|".join([dataset_name, source_subset, source_id, _compute_problem_hash(problem_text)])
    return f"bunsen-{hashlib.sha256(stable_seed.encode('utf-8')).hexdigest()[:16]}"


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
    is_mcq = question_format == "mcq" or (question_format != OPEN_ENDED_CONFIG and bool(choices))
    if not is_mcq:
        choices = []
        question_format = OPEN_ENDED_CONFIG

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
        info = hf_dataset_info(repo_id, token=_get_hf_token())
    except Exception:
        return "", ""

    dataset_sha = info.sha or ""
    dataset_last_updated = info.last_modified.isoformat() if getattr(info, "last_modified", None) else ""
    return dataset_sha, dataset_last_updated


def _load_upstream_dataset(dataset_name: str, dataset_config: str | None, split: str):
    if dataset_config is None:
        return load_dataset(dataset_name, split=split, token=_get_hf_token())
    return load_dataset(dataset_name, dataset_config, split=split, token=_get_hf_token())


def _build_source_metadata_entry(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    count: int,
    dataset_sha: str,
    dataset_last_updated: str,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config or "",
        "split": split,
        "count": count,
        "dataset_sha": dataset_sha,
        "dataset_last_updated": dataset_last_updated,
    }


def _build_source_sample(
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    source_id: str,
    problem: str,
    expected_answer: str,
    choices: list[str] | None,
    source_meta: dict[str, Any] | None = None,
    source_subset: str | None = None,
    dataset_sha: str = "",
    dataset_last_updated: str = "",
    question_format: str = "mcq",
    dataset_config_name: str = DEFAULT_HF_CONFIG,
    subset_for_metrics: str = "",
    expected_answers: list[str] | None = None,
    open_ended_eligible: bool = False,
    open_ended_eligibility_reason: str = "",
) -> dict[str, Any]:
    normalized_problem = problem.strip()
    normalized_expected_answer = expected_answer.strip()
    normalized_choices = [choice.strip() for choice in (choices or [])]
    normalized_expected_answers = [answer.strip() for answer in (expected_answers or []) if answer and answer.strip()]
    if not normalized_expected_answers and normalized_expected_answer:
        normalized_expected_answers = [normalized_expected_answer]

    resolved_source_subset = source_subset if source_subset is not None else (dataset_config or "")
    sample_uuid = _stable_source_uuid(dataset_name, resolved_source_subset, str(source_id), normalized_problem)

    return {
        "uuid": sample_uuid,
        "source_id": str(source_id),
        "source": dataset_name,
        "source_split": dataset_split,
        "source_subset": resolved_source_subset,
        "dataset_sha": dataset_sha,
        "dataset_last_updated": dataset_last_updated,
        "source_meta": deepcopy(source_meta or {}),
        "problem": normalized_problem,
        "problem_hash": _compute_problem_hash(normalized_problem),
        "has_choices": bool(normalized_choices),
        "choices": normalized_choices,
        "expected_answer": normalized_expected_answer,
        "expected_answers": normalized_expected_answers,
        "question_format": question_format,
        "dataset_config": dataset_config_name,
        "subset_for_metrics": subset_for_metrics,
        "open_ended_eligible": open_ended_eligible,
        "open_ended_eligibility_reason": open_ended_eligibility_reason,
    }


def _deduplicate_by_problem_hash(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_hashes: set[str] = set()
    unique_samples: list[dict[str, Any]] = []
    for sample in samples:
        problem_hash = sample["problem_hash"]
        if problem_hash in seen_hashes:
            continue
        seen_hashes.add(problem_hash)
        unique_samples.append(sample)
    return unique_samples


def _load_mmlu_pro_chemistry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = "TIGER-Lab/MMLU-Pro"
    dataset_split = "test"
    dataset_sha, dataset_last_updated = _get_dataset_info(dataset_name)
    rows = _load_upstream_dataset(dataset_name, None, dataset_split)

    samples: list[dict[str, Any]] = []
    for row in rows:
        if row.get("category") != "chemistry":
            continue
        options = list(row["options"])
        samples.append(
            _build_source_sample(
                dataset_name=dataset_name,
                dataset_config=None,
                dataset_split=dataset_split,
                source_id=str(row["question_id"]),
                problem=row["question"],
                expected_answer=options[row["answer_index"]],
                choices=options,
                source_meta={
                    "cot_content": row.get("cot_content"),
                    "src": row.get("src"),
                    "answer_letter": row.get("answer"),
                },
                dataset_sha=dataset_sha,
                dataset_last_updated=dataset_last_updated,
            )
        )

    metadata = _build_source_metadata_entry(
        dataset_name=dataset_name,
        dataset_config=None,
        split=dataset_split,
        count=len(samples),
        dataset_sha=dataset_sha,
        dataset_last_updated=dataset_last_updated,
    )
    return samples, metadata


def _load_mmlu_redux_subset(dataset_config: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = "edinburgh-dawg/mmlu-redux-2.0"
    dataset_split = "test"
    dataset_sha, dataset_last_updated = _get_dataset_info(dataset_name)
    rows = _load_upstream_dataset(dataset_name, dataset_config, dataset_split)

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("error_type") != "ok":
            continue
        choices = list(row["choices"])
        samples.append(
            _build_source_sample(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
                source_id=str(index),
                problem=row["question"],
                expected_answer=choices[row["answer"]],
                choices=choices,
                source_meta={
                    "error_type": row.get("error_type"),
                    "original_source": row.get("source"),
                    "correct_answer": row.get("correct_answer"),
                    "potential_reason": row.get("potential_reason"),
                },
                dataset_sha=dataset_sha,
                dataset_last_updated=dataset_last_updated,
            )
        )

    metadata = _build_source_metadata_entry(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        count=len(samples),
        dataset_sha=dataset_sha,
        dataset_last_updated=dataset_last_updated,
    )
    return samples, metadata


def _load_supergpqa_chemistry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = "m-a-p/SuperGPQA"
    dataset_split = "train"
    dataset_sha, dataset_last_updated = _get_dataset_info(dataset_name)
    rows = _load_upstream_dataset(dataset_name, None, dataset_split)

    samples: list[dict[str, Any]] = []
    for row in rows:
        if row.get("field") != "Chemistry":
            continue
        samples.append(
            _build_source_sample(
                dataset_name=dataset_name,
                dataset_config=None,
                dataset_split=dataset_split,
                source_id=str(row["uuid"]),
                problem=row["question"],
                expected_answer=row["answer"],
                choices=list(row["options"]),
                source_meta={
                    "discipline": row.get("discipline"),
                    "subfield": row.get("subfield"),
                    "difficulty": row.get("difficulty"),
                    "is_calculation": row.get("is_calculation"),
                    "answer_letter": row.get("answer_letter"),
                },
                dataset_sha=dataset_sha,
                dataset_last_updated=dataset_last_updated,
            )
        )

    metadata = _build_source_metadata_entry(
        dataset_name=dataset_name,
        dataset_config=None,
        split=dataset_split,
        count=len(samples),
        dataset_sha=dataset_sha,
        dataset_last_updated=dataset_last_updated,
    )
    return samples, metadata


def _load_gpqa_diamond_chemistry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = "Idavidrein/gpqa"
    dataset_config = "gpqa_diamond"
    dataset_split = "train"
    dataset_sha, dataset_last_updated = _get_dataset_info(dataset_name)
    rows = _load_upstream_dataset(dataset_name, dataset_config, dataset_split)

    samples: list[dict[str, Any]] = []
    for row in rows:
        if row.get("High-level domain") != "Chemistry":
            continue
        correct_answer = row["Correct Answer"]
        samples.append(
            _build_source_sample(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
                source_id=str(row["Record ID"]),
                problem=row["Question"],
                expected_answer=correct_answer,
                choices=[
                    correct_answer,
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ],
                source_meta={
                    "subdomain": row.get("Subdomain"),
                    "difficulty_estimate": row.get("Writer's Difficulty Estimate"),
                    "explanation": row.get("Explanation"),
                    "expert_validator_accuracy": row.get("Expert Validator Accuracy"),
                    "non_expert_validator_accuracy": row.get("Non-Expert Validator Accuracy"),
                },
                dataset_sha=dataset_sha,
                dataset_last_updated=dataset_last_updated,
            )
        )

    metadata = _build_source_metadata_entry(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        count=len(samples),
        dataset_sha=dataset_sha,
        dataset_last_updated=dataset_last_updated,
    )
    return samples, metadata


def _load_chembench_mcq() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_name = "jablonkagroup/ChemBench"
    dataset_split = "train"
    dataset_sha, dataset_last_updated = _get_dataset_info(dataset_name)

    samples: list[dict[str, Any]] = []
    metadata_entries: list[dict[str, Any]] = []
    for subset in CHEMBENCH_SUBSETS:
        rows = _load_upstream_dataset(dataset_name, subset, dataset_split)
        subset_count = 0
        for row in rows:
            if row.get("preferred_score") != "multiple_choice_grade":
                continue
            example = row["examples"][0]
            target_scores = json.loads(example["target_scores"])
            choices = list(target_scores.keys())
            expected_answer = next(
                (option for option, score in target_scores.items() if score == 1.0),
                choices[0],
            )
            samples.append(
                _build_source_sample(
                    dataset_name=dataset_name,
                    dataset_config=subset,
                    dataset_split=dataset_split,
                    source_id=str(row["uuid"]),
                    problem=example["input"],
                    expected_answer=expected_answer,
                    choices=choices,
                    source_meta={
                        "name": row.get("name"),
                        "description": row.get("description"),
                        "keywords": row.get("keywords"),
                        "subfield": row.get("subfield"),
                        "in_humansubset_w_tool": row.get("in_humansubset_w_tool"),
                        "in_humansubset_wo_tool": row.get("in_humansubset_wo_tool"),
                    },
                    dataset_sha=dataset_sha,
                    dataset_last_updated=dataset_last_updated,
                    source_subset=subset,
                )
            )
            subset_count += 1

        metadata_entries.append(
            _build_source_metadata_entry(
                dataset_name=dataset_name,
                dataset_config=subset,
                split=dataset_split,
                count=subset_count,
                dataset_sha=dataset_sha,
                dataset_last_updated=dataset_last_updated,
            )
        )

    return samples, metadata_entries


def _load_mcq_source_samples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mcq_samples: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []

    for loader in (
        _load_mmlu_redux_high_school_chemistry,
        _load_mmlu_redux_college_chemistry,
        _load_mmlu_pro_chemistry,
        _load_supergpqa_chemistry,
        _load_gpqa_diamond_chemistry,
    ):
        samples, metadata = loader()
        mcq_samples.extend(samples)
        source_metadata.append(metadata)

    chembench_samples, chembench_metadata = _load_chembench_mcq()
    mcq_samples.extend(chembench_samples)
    source_metadata.extend(chembench_metadata)

    return _deduplicate_by_problem_hash(mcq_samples), source_metadata


def _load_mmlu_redux_high_school_chemistry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _load_mmlu_redux_subset("high_school_chemistry")


def _load_mmlu_redux_college_chemistry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _load_mmlu_redux_subset("college_chemistry")


def _load_annotation_map(annotation_path: str | Path) -> dict[str, bool]:
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Open-ended annotation file not found at {path}. "
            f"Set {BUNSEN_OPEN_ENDED_ANNOTATION_PATH_ENV} or pass --annotation-path."
        )

    annotations: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            problem_hash = record.get("problem_hash")
            if not isinstance(problem_hash, str) or not problem_hash:
                continue
            annotations[problem_hash] = bool(record.get("open_ended_good", False))

    return annotations


def load_source_samples(
    *,
    config_name: str = DEFAULT_HF_CONFIG,
    annotation_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config_name not in {DEFAULT_HF_CONFIG, OPEN_ENDED_CONFIG}:
        raise ValueError(f"Unsupported bunsen config '{config_name}'. Expected 'mcq' or '{OPEN_ENDED_CONFIG}'.")

    mcq_samples, source_metadata = _load_mcq_source_samples()
    metadata: dict[str, Any] = {
        "source_mode": "original_upstream_datasets",
        "source_datasets": source_metadata,
        "source_sample_count_before_dedup": sum(entry["count"] for entry in source_metadata),
        "source_sample_count_after_dedup": len(mcq_samples),
    }

    if config_name == DEFAULT_HF_CONFIG:
        return mcq_samples, metadata

    resolved_annotation_path = Path(
        annotation_path or os.environ.get(BUNSEN_OPEN_ENDED_ANNOTATION_PATH_ENV, DEFAULT_OPEN_ENDED_ANNOTATION_PATH)
    )
    annotations = _load_annotation_map(resolved_annotation_path)
    good_hashes = {problem_hash for problem_hash, is_good in annotations.items() if is_good}

    open_ended_samples: list[dict[str, Any]] = []
    for sample in mcq_samples:
        if sample["problem_hash"] not in good_hashes:
            continue
        derived_sample = deepcopy(sample)
        source_meta = deepcopy(derived_sample.get("source_meta") or {})
        source_meta["original_choices"] = derived_sample.get("choices", [])
        derived_sample.update(
            {
                "source_meta": source_meta,
                "has_choices": False,
                "choices": [],
                "question_format": OPEN_ENDED_CONFIG,
                "dataset_config": OPEN_ENDED_CONFIG,
                "expected_answers": derived_sample.get("expected_answers") or [derived_sample["expected_answer"]],
                "open_ended_eligible": True,
                "open_ended_eligibility_reason": "annotated_good",
            }
        )
        open_ended_samples.append(derived_sample)

    metadata.update(
        {
            "annotation_path": str(resolved_annotation_path),
            "annotation_positive_count": len(good_hashes),
        }
    )
    return open_ended_samples, metadata


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


def prepare_source_data(
    *,
    output_dir: str | Path | None = None,
    config_name: str | None = None,
    annotation_path: str | Path | None = None,
    shuffle_mcq_choices: bool = True,
    choice_shuffle_seed: int = DEFAULT_CHOICE_SHUFFLE_SEED,
    dry_run: bool = False,
) -> tuple[Path, int]:
    resolved_config_name = config_name or os.environ.get(BUNSEN_HF_CONFIG_ENV, DEFAULT_HF_CONFIG)
    output_dirpath = Path(output_dir) if output_dir else _default_prepare_output_dir(resolved_config_name)
    output_path = output_dirpath / DEFAULT_OUTPUT_FILENAME

    source_samples, source_metadata = load_source_samples(
        config_name=resolved_config_name,
        annotation_path=annotation_path,
    )
    converted_rows = convert_bunsen_samples(
        source_samples,
        dataset_config=resolved_config_name,
        shuffle_mcq_choices=shuffle_mcq_choices,
        choice_shuffle_seed=choice_shuffle_seed,
    )

    if not dry_run:
        output_dirpath.mkdir(parents=True, exist_ok=True)
        write_jsonl(converted_rows, output_path)
        metadata = {
            "dataset_config": resolved_config_name,
            "num_samples": len(converted_rows),
            "shuffle_mcq_choices": shuffle_mcq_choices,
            "choice_shuffle_seed": choice_shuffle_seed if shuffle_mcq_choices else None,
            **source_metadata,
        }
        metadata_path = output_dirpath / DEFAULT_METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return output_path, len(converted_rows)
