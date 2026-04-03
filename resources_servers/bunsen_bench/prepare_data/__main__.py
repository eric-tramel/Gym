#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

from .core import (
    BUNSEN_HF_CONFIG_ENV,
    BUNSEN_OPEN_ENDED_ANNOTATION_PATH_ENV,
    DEFAULT_CHOICE_SHUFFLE_SEED,
    DEFAULT_EXAMPLE_OUTPUT_PATH,
    DEFAULT_HF_CONFIG,
    DEFAULT_OPEN_ENDED_ANNOTATION_PATH,
    DEFAULT_SYSTEM_PROMPT,
    prepare_data,
    prepare_raw_jsonl,
    prepare_source_data,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare bunsen-bench data for the Gym environment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    examples_parser = subparsers.add_parser("examples", help="Regenerate the checked-in smoke-test dataset.")
    examples_parser.add_argument(
        "--output",
        default=str(DEFAULT_EXAMPLE_OUTPUT_PATH),
        help=f"Output JSONL path (default: {DEFAULT_EXAMPLE_OUTPUT_PATH})",
    )

    raw_parser = subparsers.add_parser("raw", help="Convert raw prompt/answer JSONL into Gym-compatible rows.")
    raw_parser.add_argument("--input", required=True, help="Input raw JSONL path.")
    raw_parser.add_argument("--output", required=True, help="Output Gym JSONL path.")
    raw_parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional system prompt to prepend.",
    )
    raw_parser.add_argument("--prompt-field", default="prompt", help="Field name containing the user prompt.")
    raw_parser.add_argument("--answer-field", default="expected_answer", help="Field name containing the gold answer.")
    raw_parser.add_argument("--task-id-field", default="task_id", help="Field name containing the task id.")
    raw_parser.add_argument("--metadata-field", default="metadata", help="Field name containing metadata.")

    source_parser = subparsers.add_parser(
        "source",
        help="Source bunsen-bench from the original upstream datasets and write test.jsonl + metadata.json.",
    )
    source_parser.add_argument(
        "--config",
        default=None,
        help=f"Dataset config (default: ${BUNSEN_HF_CONFIG_ENV} or {DEFAULT_HF_CONFIG})",
    )
    source_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated test.jsonl and metadata.json (default: data/prepare/<config>).",
    )
    source_parser.add_argument(
        "--annotation-path",
        default=None,
        help=(
            "Open-ended annotation JSONL path "
            f"(default: ${BUNSEN_OPEN_ENDED_ANNOTATION_PATH_ENV} or {DEFAULT_OPEN_ENDED_ANNOTATION_PATH})"
        ),
    )
    source_parser.add_argument(
        "--choice-shuffle-seed",
        type=int,
        default=DEFAULT_CHOICE_SHUFFLE_SEED,
        help=f"Deterministic seed for MCQ choice shuffling (default: {DEFAULT_CHOICE_SHUFFLE_SEED})",
    )
    source_parser.add_argument(
        "--no-shuffle-mcq-choices",
        action="store_true",
        help="Disable deterministic shuffling for MCQ choices.",
    )
    source_parser.add_argument("--dry-run", action="store_true", help="Load and convert without writing files.")

    args = parser.parse_args(argv)

    if args.command == "examples":
        count = prepare_data(output_path=args.output)
        print(f"Wrote {count} bunsen-bench example rows to {args.output}")
        return

    if args.command == "raw":
        count = prepare_raw_jsonl(
            input_path=args.input,
            output_path=args.output,
            system_prompt=args.system_prompt,
            prompt_field=args.prompt_field,
            answer_field=args.answer_field,
            task_id_field=args.task_id_field,
            metadata_field=args.metadata_field,
        )
        print(f"Wrote {count} converted rows to {args.output}")
        return

    output_path, count = prepare_source_data(
        output_dir=args.output_dir,
        config_name=args.config,
        annotation_path=args.annotation_path,
        shuffle_mcq_choices=not args.no_shuffle_mcq_choices,
        choice_shuffle_seed=args.choice_shuffle_seed,
        dry_run=args.dry_run,
    )
    output_dir = Path(output_path).parent
    action = "Prepared" if args.dry_run else "Wrote"
    print(f"{action} {count} bunsen-bench rows in {output_dir}")
    print(f"  test.jsonl: {output_path}")
    if not args.dry_run:
        print(f"  metadata.json: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
