# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.bunsen_bench.app import (
    BunsenBenchResourcesServer,
    BunsenBenchResourcesServerConfig,
    BunsenBenchVerifyRequest,
)
from resources_servers.bunsen_bench.prepare_data import (
    BUNSEN_OPEN_ENDED_ANNOTATION_PATH_ENV,
    DEFAULT_BUNSEN_SAMPLES,
    DEFAULT_CHOICE_SHUFFLE_SEED,
    DEFAULT_HF_CONFIG,
    convert_bunsen_samples,
    convert_rows,
    load_source_samples,
    prepare_data,
    prepare_source_data,
)


class TestBunsenBenchResourcesServer:
    def _create_server(self) -> BunsenBenchResourcesServer:
        config = BunsenBenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return BunsenBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_request(
        self,
        model_output: str,
        expected_answer: str,
        task_id: str = "bunsen-001",
        prompt: str = "Respond with exactly the single word: exothermic",
    ) -> BunsenBenchVerifyRequest:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_test",
                    content=[NeMoGymResponseOutputText(annotations=[], text=model_output)],
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        return BunsenBenchVerifyRequest(
            task_id=task_id,
            prompt=prompt,
            expected_answer=expected_answer,
            metadata={"category": "test"},
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content=prompt)]
            ),
            response=response,
        )

    def test_sanity(self) -> None:
        self._create_server()

    def test_verify_matches_exact_answer(self) -> None:
        server = self._create_server()
        request = self._create_request(model_output="exothermic", expected_answer="exothermic")

        result = asyncio.run(server.verify(request))

        assert result.reward == 1.0
        assert result.matched is True
        assert result.model_answer == "exothermic"

    def test_verify_extracts_choice_tag_answer(self) -> None:
        server = self._create_server()
        request = self._create_request(
            model_output="<choice>Safety goggles</choice>", expected_answer="safety goggles"
        )

        result = asyncio.run(server.verify(request))

        assert result.reward == 1.0
        assert result.matched is True
        assert result.model_answer == "Safety goggles"

    def test_verify_extracts_open_ended_answer_tag(self) -> None:
        server = self._create_server()
        request = self._create_request(model_output="<answer>neutral</answer>", expected_answer="neutral")

        result = asyncio.run(server.verify(request))

        assert result.reward == 1.0
        assert result.matched is True
        assert result.model_answer == "neutral"

    def test_verify_normalizes_case_and_whitespace(self) -> None:
        server = self._create_server()
        request = self._create_request(
            model_output="  SAFETY   GOGGLES  ",
            expected_answer="safety goggles",
            task_id="bunsen-003",
            prompt="Respond with exactly the phrase: safety goggles",
        )

        result = asyncio.run(server.verify(request))

        assert result.reward == 1.0
        assert result.matched is True
        assert result.normalized_model_answer == "safety goggles"

    def test_verify_rejects_extra_explanation(self) -> None:
        server = self._create_server()
        request = self._create_request(
            model_output="neutral because pure water has pH 7",
            expected_answer="neutral",
            task_id="bunsen-004",
            prompt="Return only the pH classification of pure water at room temperature.",
        )

        result = asyncio.run(server.verify(request))

        assert result.reward == 0.0
        assert result.matched is False
        assert result.normalized_expected_answer == "neutral"

    def test_verify_handles_empty_output(self) -> None:
        server = self._create_server()
        request = self._create_request(
            model_output="",
            expected_answer="test tube",
            task_id="bunsen-005",
            prompt="Respond with exactly the apparatus name: test tube",
        )

        result = asyncio.run(server.verify(request))

        assert result.reward == 0.0
        assert result.matched is False
        assert result.model_answer == ""
        assert result.normalized_model_answer == ""


class TestPrepareData:
    @staticmethod
    def _source_sample(
        *,
        uuid: str,
        problem: str,
        expected_answer: str,
        choices: list[str] | None = None,
        question_format: str = "mcq",
    ) -> dict:
        return {
            "uuid": uuid,
            "source_id": uuid,
            "source": "test-source",
            "source_split": "test",
            "source_subset": "",
            "dataset_sha": "sha-test",
            "dataset_last_updated": "",
            "source_meta": {},
            "problem": problem,
            "problem_hash": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
            "has_choices": bool(choices),
            "choices": choices or [],
            "expected_answer": expected_answer,
            "expected_answers": [expected_answer],
            "question_format": question_format,
            "dataset_config": question_format,
            "subset_for_metrics": "",
            "open_ended_eligible": False,
            "open_ended_eligibility_reason": "",
        }

    def test_convert_rows_builds_gym_shape(self) -> None:
        rows = [
            {
                "task_id": "custom-1",
                "prompt": "Return only the chemical symbol for sodium.",
                "expected_answer": "Na",
                "metadata": {"topic": "chemistry"},
            }
        ]

        result = convert_rows(rows)

        assert result == [
            {
                "task_id": "custom-1",
                "prompt": "Return only the chemical symbol for sodium.",
                "expected_answer": "Na",
                "metadata": {"topic": "chemistry"},
                "responses_create_params": {
                    "input": [
                        {
                            "role": "system",
                            "content": "Answer the bunsen-bench task exactly. Return only the final answer.",
                        },
                        {"role": "user", "content": "Return only the chemical symbol for sodium."},
                    ]
                },
            }
        ]

    def test_convert_rows_generates_missing_task_id(self) -> None:
        rows = [{"prompt": "Respond with exactly the single word: exothermic", "expected_answer": "exothermic"}]

        result = convert_rows(rows)

        assert result[0]["task_id"] == "bunsen-001"
        assert result[0]["metadata"] == {}

    def test_convert_bunsen_samples_formats_mcq_rows(self) -> None:
        rows = convert_bunsen_samples([DEFAULT_BUNSEN_SAMPLES[0]], dataset_config="mcq")

        assert rows == [
            {
                "task_id": "bunsen-001",
                "prompt": (
                    "Which apparatus is best for heating a small liquid sample directly over a flame?\n"
                    "<choices>\n"
                    "<choice>Test tube</choice>\n"
                    "<choice>Beaker</choice>\n"
                    "<choice>Watch glass</choice>\n"
                    "<choice>Graduated cylinder</choice>\n"
                    "</choices>\n"
                    "Respond with the correct choice in <choice></choice> tags, exactly as it is written above."
                ),
                "expected_answer": "Test tube",
                "metadata": {
                    "uuid": "bunsen-001",
                    "question_format": "mcq",
                    "subset_for_metrics": "lab-equipment",
                    "source": "bunsen-bench-stub",
                    "source_subset": "example",
                    "problem": "Which apparatus is best for heating a small liquid sample directly over a flame?",
                    "choices": ["Test tube", "Beaker", "Watch glass", "Graduated cylinder"],
                    "dataset_config": "mcq",
                    "dataset_sha": "",
                    "dataset_last_updated": "",
                },
                "choices": ["Test tube", "Beaker", "Watch glass", "Graduated cylinder"],
                "question_format": "mcq",
                "subset_for_metrics": "lab-equipment",
                "source": "bunsen-bench-stub",
                "source_subset": "example",
                "dataset_config": "mcq",
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": (
                                "Which apparatus is best for heating a small liquid sample directly over a flame?\n"
                                "<choices>\n"
                                "<choice>Test tube</choice>\n"
                                "<choice>Beaker</choice>\n"
                                "<choice>Watch glass</choice>\n"
                                "<choice>Graduated cylinder</choice>\n"
                                "</choices>\n"
                                "Respond with the correct choice in <choice></choice> tags, exactly as it is written above."
                            ),
                        }
                    ]
                },
            }
        ]

    def test_load_source_samples_mcq_aggregates_original_datasets(self) -> None:
        mmlu_pro_rows = [
            {
                "question_id": 1,
                "category": "chemistry",
                "question": "MMLU-Pro chemistry question",
                "options": ["A", "B", "C", "D"],
                "answer_index": 2,
            },
            {
                "question_id": 2,
                "category": "biology",
                "question": "Ignore me",
                "options": ["A", "B"],
                "answer_index": 0,
            },
        ]
        mmlu_redux_hs_rows = [
            {
                "error_type": "ok",
                "question": "HS chemistry question",
                "choices": ["HS-A", "HS-B"],
                "answer": 1,
            }
        ]
        mmlu_redux_college_rows = [
            {
                "error_type": "not_ok",
                "question": "Bad college chemistry question",
                "choices": ["Bad-A", "Bad-B"],
                "answer": 0,
            },
            {
                "error_type": "ok",
                "question": "College chemistry question",
                "choices": ["C-A", "C-B"],
                "answer": 0,
            },
        ]
        supergpqa_rows = [
            {
                "uuid": "super-1",
                "field": "Chemistry",
                "question": "SuperGPQA chemistry question",
                "options": ["S-A", "S-B", "S-C"],
                "answer": "S-B",
            },
            {
                "uuid": "super-2",
                "field": "Physics",
                "question": "Ignore super physics question",
                "options": ["P-A", "P-B"],
                "answer": "P-A",
            },
        ]
        gpqa_rows = [
            {
                "Record ID": "gpqa-1",
                "High-level domain": "Chemistry",
                "Question": "GPQA chemistry question",
                "Correct Answer": "G-A",
                "Incorrect Answer 1": "G-B",
                "Incorrect Answer 2": "G-C",
                "Incorrect Answer 3": "G-D",
            },
            {
                "Record ID": "gpqa-2",
                "High-level domain": "Biology",
                "Question": "Ignore GPQA biology question",
                "Correct Answer": "B-A",
                "Incorrect Answer 1": "B-B",
                "Incorrect Answer 2": "B-C",
                "Incorrect Answer 3": "B-D",
            },
        ]
        chembench_rows = [
            {
                "uuid": "chembench-1",
                "preferred_score": "multiple_choice_grade",
                "examples": [
                    {
                        "input": "ChemBench chemistry question",
                        "target_scores": json.dumps({"CB-A": 0.0, "CB-B": 1.0, "CB-C": 0.0}),
                    }
                ],
            }
        ]

        def fake_load_dataset(dataset_name: str, dataset_config=None, split=None, token=None):
            if dataset_name == "TIGER-Lab/MMLU-Pro":
                assert dataset_config is None and split == "test"
                return mmlu_pro_rows
            if dataset_name == "edinburgh-dawg/mmlu-redux-2.0" and dataset_config == "high_school_chemistry":
                return mmlu_redux_hs_rows
            if dataset_name == "edinburgh-dawg/mmlu-redux-2.0" and dataset_config == "college_chemistry":
                return mmlu_redux_college_rows
            if dataset_name == "m-a-p/SuperGPQA":
                assert dataset_config is None and split == "train"
                return supergpqa_rows
            if dataset_name == "Idavidrein/gpqa":
                assert dataset_config == "gpqa_diamond" and split == "train"
                return gpqa_rows
            if dataset_name == "jablonkagroup/ChemBench":
                return chembench_rows if dataset_config == "general_chemistry" else []
            raise AssertionError(f"Unexpected dataset load: {(dataset_name, dataset_config, split, token)}")

        def fake_dataset_info(dataset_name: str, token=None):
            class _Info:
                sha = f"sha-{dataset_name.split('/')[-1]}"
                last_modified = None

            return _Info()

        with (
            patch("resources_servers.bunsen_bench.prepare_data.core.load_dataset", side_effect=fake_load_dataset),
            patch("resources_servers.bunsen_bench.prepare_data.core.hf_dataset_info", side_effect=fake_dataset_info),
        ):
            samples, metadata = load_source_samples(config_name="mcq")

        assert len(samples) == 6
        assert metadata["source_mode"] == "original_upstream_datasets"
        assert metadata["source_sample_count_before_dedup"] == 6
        assert metadata["source_sample_count_after_dedup"] == 6

        problems = {sample["problem"] for sample in samples}
        assert problems == {
            "HS chemistry question",
            "College chemistry question",
            "MMLU-Pro chemistry question",
            "SuperGPQA chemistry question",
            "GPQA chemistry question",
            "ChemBench chemistry question",
        }
        assert any(
            sample["source"] == "jablonkagroup/ChemBench" and sample["source_subset"] == "general_chemistry"
            for sample in samples
        )

    def test_load_source_samples_open_ended_filters_annotation_file(self, tmp_path: Path) -> None:
        good_sample = self._source_sample(
            uuid="mcq-good",
            problem="What is the symbol for sodium?",
            expected_answer="Na",
            choices=["Na", "K", "Ca", "Mg"],
        )
        bad_sample = self._source_sample(
            uuid="mcq-bad",
            problem="Which option names potassium?",
            expected_answer="K",
            choices=["Na", "K", "Ca", "Mg"],
        )
        annotation_path = tmp_path / "annotation-open-ended.jsonl"
        annotation_path.write_text(
            "\n".join(
                [
                    json.dumps({"problem_hash": good_sample["problem_hash"], "open_ended_good": True}),
                    json.dumps({"problem_hash": bad_sample["problem_hash"], "open_ended_good": False}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with patch(
            "resources_servers.bunsen_bench.prepare_data.core._load_mcq_source_samples",
            return_value=(
                [good_sample, bad_sample],
                [
                    {
                        "dataset_name": "test-source",
                        "dataset_config": "",
                        "split": "test",
                        "count": 2,
                        "dataset_sha": "",
                        "dataset_last_updated": "",
                    }
                ],
            ),
        ):
            samples, metadata = load_source_samples(config_name="open_ended", annotation_path=annotation_path)

        assert len(samples) == 1
        assert samples[0]["uuid"] == "mcq-good"
        assert samples[0]["choices"] == []
        assert samples[0]["question_format"] == "open_ended"
        assert samples[0]["dataset_config"] == "open_ended"
        assert samples[0]["source_meta"]["original_choices"] == ["Na", "K", "Ca", "Mg"]
        assert samples[0]["open_ended_eligible"] is True
        assert metadata["annotation_path"] == str(annotation_path)
        assert metadata["annotation_positive_count"] == 1

    def test_prepare_source_data_writes_test_jsonl_and_metadata(self, tmp_path: Path) -> None:
        fake_source_samples = [
            self._source_sample(
                uuid="src-1",
                problem="What is the chemical symbol for sodium?",
                expected_answer="Na",
                question_format="open_ended",
            ),
            self._source_sample(
                uuid="src-2",
                problem="Which device measures mass?",
                expected_answer="Balance",
                choices=["Thermometer", "Balance", "Pipette", "Bunsen burner"],
            ),
        ]

        with patch(
            "resources_servers.bunsen_bench.prepare_data.core.load_source_samples",
            return_value=(
                fake_source_samples,
                {
                    "source_mode": "original_upstream_datasets",
                    "source_datasets": [
                        {
                            "dataset_name": "TIGER-Lab/MMLU-Pro",
                            "dataset_config": "",
                            "split": "test",
                            "count": 2,
                            "dataset_sha": "abc123def456",
                            "dataset_last_updated": "",
                        }
                    ],
                    "source_sample_count_before_dedup": 2,
                    "source_sample_count_after_dedup": 2,
                },
            ),
        ):
            output_path, count = prepare_source_data(output_dir=tmp_path)

        assert count == 2
        assert output_path == tmp_path / "test.jsonl"

        generated_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

        assert generated_rows[0]["task_id"] == "src-1"
        assert generated_rows[0]["question_format"] == "open_ended"
        assert generated_rows[0]["dataset_config"] == "open_ended"
        assert generated_rows[0]["metadata"]["dataset_sha"] == "sha-test"

        assert generated_rows[1]["task_id"] == "src-2"
        assert generated_rows[1]["question_format"] == "mcq"
        assert generated_rows[1]["choices"] == ["Balance", "Thermometer", "Bunsen burner", "Pipette"]
        assert generated_rows[1]["metadata"]["choices"] == ["Balance", "Thermometer", "Bunsen burner", "Pipette"]

        assert metadata == {
            "dataset_config": DEFAULT_HF_CONFIG,
            "source_mode": "original_upstream_datasets",
            "source_datasets": [
                {
                    "dataset_name": "TIGER-Lab/MMLU-Pro",
                    "dataset_config": "",
                    "split": "test",
                    "count": 2,
                    "dataset_sha": "abc123def456",
                    "dataset_last_updated": "",
                }
            ],
            "source_sample_count_before_dedup": 2,
            "source_sample_count_after_dedup": 2,
            "shuffle_mcq_choices": True,
            "choice_shuffle_seed": DEFAULT_CHOICE_SHUFFLE_SEED,
            "num_samples": 2,
        }

    def test_load_source_samples_open_ended_uses_env_annotation_path(self, tmp_path: Path) -> None:
        sample = self._source_sample(
            uuid="mcq-good",
            problem="What is the symbol for sodium?",
            expected_answer="Na",
            choices=["Na", "K"],
        )
        annotation_path = tmp_path / "annotation-open-ended.jsonl"
        annotation_path.write_text(
            json.dumps({"problem_hash": sample["problem_hash"], "open_ended_good": True}) + "\n",
            encoding="utf-8",
        )

        with (
            patch(
                "resources_servers.bunsen_bench.prepare_data.core._load_mcq_source_samples",
                return_value=([sample], []),
            ),
            patch.dict(
                "os.environ",
                {BUNSEN_OPEN_ENDED_ANNOTATION_PATH_ENV: str(annotation_path)},
                clear=False,
            ),
        ):
            samples, _ = load_source_samples(config_name="open_ended")

        assert len(samples) == 1
        assert samples[0]["uuid"] == "mcq-good"

    def test_prepare_data_default_examples_match_checked_in_example_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "example.jsonl"

        prepare_data(output_path=output_path)

        generated = output_path.read_text(encoding="utf-8").strip().splitlines()
        checked_in = (
            Path("resources_servers/bunsen_bench/data/example.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )

        assert [json.loads(line) for line in generated] == [json.loads(line) for line in checked_in]
