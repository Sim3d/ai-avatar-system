"""Offline Bedrock LLM contract tests; runnable without AWS credentials."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config import settings
from app.services.llm import LLMService


class BedrockClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = patch.multiple(
            settings,
            LLM_PROVIDER="bedrock",
            LLM_MODEL="global.anthropic.claude-sonnet-4-6",
            AWS_REGION="us-east-1",
            ANTHROPIC_API_KEY="",
        )
        self.config.start()
        self.addCleanup(self.config.stop)
        self.client_patch = patch("app.services.llm.anthropic.AsyncAnthropicBedrock")
        self.client_factory = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.client = self.client_factory.return_value
        self.client.close = AsyncMock()

    async def test_bedrock_client_uses_region_and_no_api_key(self):
        service = LLMService()
        self.client_factory.assert_called_once_with(aws_region="us-east-1")
        self.assertIs(service.client, self.client)
        # Downstream code paths are the Anthropic ones.
        self.assertEqual(service.provider, "anthropic")
        self.assertEqual(service.model, "global.anthropic.claude-sonnet-4-6")
        await service.close()
        self.client.close.assert_awaited_once()

    async def test_generate_response_reuses_anthropic_path(self):
        self.client.messages.create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Bonjour")],
                usage=None,
            )
        )
        service = LLMService()
        result = await service.generate_response([{"role": "user", "content": "Salut"}])
        self.assertEqual(result, "Bonjour")
        kwargs = self.client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "global.anthropic.claude-sonnet-4-6")
        # Prompt caching must survive the Bedrock path — Bedrock supports
        # cache_control on Claude models.
        self.assertEqual(kwargs["system"][0]["cache_control"], {"type": "ephemeral"})
        await service.close()

    async def test_streaming_reuses_anthropic_path(self):
        async def text_stream():
            yield "Bon"
            yield "jour"

        stream_cm = AsyncMock()
        stream_cm.__aenter__.return_value = SimpleNamespace(text_stream=text_stream())
        self.client.messages.stream = lambda **kwargs: stream_cm

        service = LLMService()
        result = [part async for part in service.stream_response(
            [{"role": "user", "content": "Salut"}]
        )]
        self.assertEqual(result, ["Bon", "jour"])
        await service.close()


class BedrockSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_client_initializes_without_credentials(self):
        with patch.multiple(
            settings,
            LLM_PROVIDER="bedrock",
            LLM_MODEL="global.anthropic.claude-sonnet-4-6",
            AWS_REGION="us-east-1",
        ):
            service = LLMService()
        self.assertEqual(type(service.client).__name__, "AsyncAnthropicBedrock")
        await service.close()


if __name__ == "__main__":
    unittest.main()
