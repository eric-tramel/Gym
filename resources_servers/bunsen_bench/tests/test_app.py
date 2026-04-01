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
from unittest.mock import MagicMock

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
