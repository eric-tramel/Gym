# Description

`bunsen-bench` is a stub single-turn environment for bootstrapping a new NeMo-Gym benchmark. The current verifier uses a normalized exact-match check against the task's `expected_answer`, which makes the scaffold immediately runnable while leaving room for richer grading logic later.

Example inputs live in `resources_servers/bunsen_bench/data/example.jsonl`.

## Prepare data

To regenerate the built-in example dataset:

```bash
python resources_servers/bunsen_bench/prepare_data.py
```

To convert raw JSONL into Gym-compatible `bunsen-bench` rows:

```bash
python resources_servers/bunsen_bench/prepare_data.py \
    --input /path/to/raw.jsonl \
    --output resources_servers/bunsen_bench/data/train.jsonl
```

Raw rows should contain `prompt` and `expected_answer`, plus optional `task_id` and `metadata` fields.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
