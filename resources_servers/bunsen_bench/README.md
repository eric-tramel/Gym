# Description

`bunsen-bench` is a stub single-turn environment for bootstrapping a new NeMo-Gym benchmark. The current verifier uses a normalized exact-match check against the task's `expected_answer`, with support for the tagged `<choice>...</choice>` and `<answer>...</answer>` responses emitted by the internal bunsen-bench preparation flow.

Example inputs live in `resources_servers/bunsen_bench/data/example.jsonl`.

## Prepare data

The prep entrypoint now lives under `resources_servers/bunsen_bench/prepare_data/` and mirrors the internal bunsen-bench flow at the source-build layer: load the original upstream datasets, assemble the bunsen benchmark split, render the canonical chemistry prompt shape, and emit `test.jsonl` plus `metadata.json`.

To regenerate the built-in smoke-test dataset:

```bash
python -m resources_servers.bunsen_bench.prepare_data examples
```

To rebuild the MCQ split from the original upstream datasets into a Gym-ready handoff artifact:

```bash
python -m resources_servers.bunsen_bench.prepare_data source \
    --config mcq \
    --output-dir resources_servers/bunsen_bench/data/prepare/mcq
```

This writes:

- `resources_servers/bunsen_bench/data/prepare/mcq/test.jsonl`
- `resources_servers/bunsen_bench/data/prepare/mcq/metadata.json`

The source-build mode pulls from the original upstream datasets used by bunsen-bench, including:

- `TIGER-Lab/MMLU-Pro`
- `edinburgh-dawg/mmlu-redux-2.0`
- `m-a-p/SuperGPQA`
- `Idavidrein/gpqa`
- `jablonkagroup/ChemBench`

Config defaults:

- config: `mcq` or `$BUNSEN_HF_CONFIG`
- optional HuggingFace auth: `$HF_TOKEN`

For the annotation-derived open-ended split:

```bash
python -m resources_servers.bunsen_bench.prepare_data source \
    --config open_ended \
    --annotation-path /path/to/annotation-open-ended.jsonl \
    --output-dir resources_servers/bunsen_bench/data/prepare/open_ended
```

The annotation path defaults to `$BUNSEN_OPEN_ENDED_ANNOTATION_PATH` and otherwise `annotation-open-ended.jsonl`.

For local ad hoc JSONL conversion, a raw compatibility mode is still available:

```bash
python -m resources_servers.bunsen_bench.prepare_data raw \
    --input /path/to/raw.jsonl \
    --output resources_servers/bunsen_bench/data/train.jsonl
```

Raw rows should contain `prompt` and `expected_answer`, plus optional `task_id` and `metadata` fields.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
