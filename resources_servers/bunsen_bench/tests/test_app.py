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
from unittest.mock import MagicMock

from pathlib import Path

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
    DEFAULT_SYSTEM_PROMPT,
    convert_rows,
    prepare_data,
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
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
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

    def test_prepare_data_default_examples_match_checked_in_example_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "example.jsonl"

        prepare_data(output_path=output_path)

        generated = output_path.read_text(encoding="utf-8").strip().splitlines()
        checked_in = (
            Path("resources_servers/bunsen_bench/data/example.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )

        assert [json.loads(line) for line in generated] == [json.loads(line) for line in checked_in]
