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
    DEFAULT_BUNSEN_SAMPLES,
    DEFAULT_CHOICE_SHUFFLE_SEED,
    DEFAULT_HF_CONFIG,
    DEFAULT_HF_REPO,
    convert_bunsen_samples,
    convert_rows,
    prepare_data,
    prepare_huggingface_data,
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

    def test_prepare_huggingface_data_writes_test_jsonl_and_metadata(self, tmp_path: Path) -> None:
        fake_dataset = [
            {
                "uuid": "hf-1",
                "problem": "What is the chemical symbol for sodium?",
                "expected_answer": "Na",
                "question_format": "open_ended",
                "subset_for_metrics": "chemistry",
                "source": "hf-source",
                "source_subset": "validation",
            },
            {
                "uuid": "hf-2",
                "problem": "Which device measures mass?",
                "choices": ["Thermometer", "Balance", "Pipette", "Bunsen burner"],
                "expected_answer": "Balance",
                "question_format": "mcq",
                "subset_for_metrics": "lab-equipment",
                "source": "hf-source",
                "source_subset": "validation",
            },
        ]

        class _Info:
            sha = "abc123def456"
            last_modified = None

        with (
            patch("resources_servers.bunsen_bench.prepare_data.core.load_dataset", return_value=fake_dataset),
            patch("resources_servers.bunsen_bench.prepare_data.core.hf_dataset_info", return_value=_Info()),
        ):
            output_path, count = prepare_huggingface_data(output_dir=tmp_path)

        assert count == 2
        assert output_path == tmp_path / "test.jsonl"

        generated_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

        assert generated_rows[0]["task_id"] == "hf-1"
        assert generated_rows[0]["question_format"] == "open_ended"
        assert generated_rows[0]["dataset_config"] == DEFAULT_HF_CONFIG
        assert generated_rows[0]["metadata"]["dataset_sha"] == "abc123def456"

        assert generated_rows[1]["task_id"] == "hf-2"
        assert generated_rows[1]["question_format"] == "mcq"
        assert generated_rows[1]["choices"] == ["Balance", "Thermometer", "Bunsen burner", "Pipette"]
        assert generated_rows[1]["metadata"]["choices"] == ["Balance", "Thermometer", "Bunsen burner", "Pipette"]

        assert metadata == {
            "dataset_repo": DEFAULT_HF_REPO,
            "dataset_config": DEFAULT_HF_CONFIG,
            "dataset_sha": "abc123def456",
            "dataset_last_updated": "",
            "split": "test",
            "shuffle_mcq_choices": True,
            "choice_shuffle_seed": DEFAULT_CHOICE_SHUFFLE_SEED,
            "num_samples": 2,
        }

    def test_prepare_data_default_examples_match_checked_in_example_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "example.jsonl"

        prepare_data(output_path=output_path)

        generated = output_path.read_text(encoding="utf-8").strip().splitlines()
        checked_in = (
            Path("resources_servers/bunsen_bench/data/example.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )

        assert [json.loads(line) for line in generated] == [json.loads(line) for line in checked_in]
