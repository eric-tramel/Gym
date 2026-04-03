#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare raw bunsen-bench rows into Gym-compatible JSONL.

This follows the same role as the AIME preparation scripts: take a simpler raw
dataset format and emit NeMo Gym rows with `responses_create_params.input`
already populated.

Expected raw input JSONL shape:
    {
        "prompt": "Return only the chemical symbol for sodium.",
        "expected_answer": "Na",
        "task_id": "bunsen-002",            # optional
        "metadata": {"topic": "chemistry"}  # optional
    }

If `--input` is omitted, the script writes a small built-in example dataset that
matches the checked-in `data/example.jsonl`.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = "Answer the bunsen-bench task exactly. Return only the final answer."
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "data" / "example.jsonl"
DEFAULT_EXAMPLES = [
    {
        "task_id": "bunsen-001",
        "prompt": "Respond with exactly the single word: exothermic",
        "expected_answer": "exothermic",
        "metadata": {"category": "stub", "topic": "chemistry"},
    },
    {
        "task_id": "bunsen-002",
        "prompt": "Return only the chemical symbol for sodium.",
        "expected_answer": "Na",
        "metadata": {"category": "stub", "topic": "chemistry"},
    },
    {
        "task_id": "bunsen-003",
        "prompt": "Respond with exactly the phrase: safety goggles",
        "expected_answer": "safety goggles",
        "metadata": {"category": "stub", "topic": "lab-safety"},
    },
    {
        "task_id": "bunsen-004",
        "prompt": "Return only the pH classification of pure water at room temperature.",
        "expected_answer": "neutral",
        "metadata": {"category": "stub", "topic": "chemistry"},
    },
    {
        "task_id": "bunsen-005",
        "prompt": "Respond with exactly the apparatus name: test tube",
        "expected_answer": "test tube",
        "metadata": {"category": "stub", "topic": "lab-equipment"},
    },
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


def _load_jsonl_rows(input_path: str) -> list[dict[str, Any]]:
    with open(input_path, encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows)


def prepare_data(
    *,
    input_path: str | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_field: str = "prompt",
    answer_field: str = "expected_answer",
    task_id_field: str = "task_id",
    metadata_field: str = "metadata",
) -> int:
    raw_rows = _load_jsonl_rows(input_path) if input_path else deepcopy(DEFAULT_EXAMPLES)
    converted_rows = convert_rows(
        raw_rows,
        system_prompt=system_prompt,
        prompt_field=prompt_field,
        answer_field=answer_field,
        task_id_field=task_id_field,
        metadata_field=metadata_field,
    )
    return write_jsonl(converted_rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bunsen-bench data into Gym-compatible JSONL")
    parser.add_argument("--input", help="Optional raw JSONL input. If omitted, built-in example rows are used.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt to prepend.")
    parser.add_argument("--prompt-field", default="prompt", help="Field name containing the user prompt.")
    parser.add_argument("--answer-field", default="expected_answer", help="Field name containing the gold answer.")
    parser.add_argument("--task-id-field", default="task_id", help="Field name containing the task id.")
    parser.add_argument("--metadata-field", default="metadata", help="Field name containing metadata.")
    args = parser.parse_args()

    count = prepare_data(
        input_path=args.input,
        output_path=args.output,
        system_prompt=args.system_prompt,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        task_id_field=args.task_id_field,
        metadata_field=args.metadata_field,
    )
    print(f"Wrote {count} bunsen-bench examples to {args.output}")


if __name__ == "__main__":
    main()
