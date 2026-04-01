# Description

`bunsen-bench` is a stub single-turn environment for bootstrapping a new NeMo-Gym benchmark. The current verifier uses a normalized exact-match check against the task's `expected_answer`, which makes the scaffold immediately runnable while leaving room for richer grading logic later.

Example inputs live in `resources_servers/bunsen_bench/data/example.jsonl`.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
