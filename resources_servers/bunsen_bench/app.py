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
import re
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


CHOICE_TAG_PATTERN = re.compile(r"<choice>\s*(.*?)\s*</choice>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def _extract_answer(text: Optional[str]) -> str:
    if not text:
        return ""

    stripped_text = text.strip()
    for pattern in (CHOICE_TAG_PATTERN, ANSWER_TAG_PATTERN):
        match = pattern.search(stripped_text)
        if match:
            return match.group(1).strip()

    return stripped_text


def _normalize_answer(text: Optional[str]) -> str:
    if not text:
        return ""

    return " ".join(text.split()).casefold()


class BunsenBenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class BunsenBenchRunRequest(BaseRunRequest):
    task_id: str
    prompt: str
    expected_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BunsenBenchVerifyRequest(BunsenBenchRunRequest, BaseVerifyRequest):
    pass


class BunsenBenchVerifyResponse(BaseVerifyResponse):
    task_id: str
    prompt: str
    expected_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_answer: str
    normalized_expected_answer: str
    normalized_model_answer: str
    matched: bool


class BunsenBenchResourcesServer(SimpleResourcesServer):
    config: BunsenBenchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: BunsenBenchVerifyRequest) -> BunsenBenchVerifyResponse:
        model_answer = _extract_answer(body.response.output_text)
        normalized_expected_answer = _normalize_answer(body.expected_answer)
        normalized_model_answer = _normalize_answer(model_answer)
        matched = bool(normalized_expected_answer) and normalized_model_answer == normalized_expected_answer

        return BunsenBenchVerifyResponse(
            **body.model_dump(),
            reward=float(matched),
            model_answer=model_answer,
            normalized_expected_answer=normalized_expected_answer,
            normalized_model_answer=normalized_model_answer,
            matched=matched,
        )


if __name__ == "__main__":
    BunsenBenchResourcesServer.run_webserver()
