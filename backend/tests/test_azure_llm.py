"""Offline Azure LLM contract tests; runnable without the API/GPU test fixtures."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
from azure.core.exceptions import ClientAuthenticationError
from pydantic import ValidationError

from app.config import Settings, settings
from app.services.llm import LLMAuthError, LLMService


class AzureSettingsTests(unittest.TestCase):
    def make_settings(self, **overrides):
        values = {
            "SECRET_KEY": "s" * 64,
            "JWT_SECRET_KEY": "j" * 64,
            "ENVIRONMENT": "development",
            "LLM_PROVIDER": "azure_openai",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT": "avatar-chat",
            "AZURE_OPENAI_AUTH": "entra",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    def test_entra_requires_no_api_key(self):
        self.assertEqual(self.make_settings(AZURE_OPENAI_API_KEY="").AZURE_OPENAI_AUTH, "entra")

    def test_invalid_configuration_fails_before_network_access(self):
        for overrides in (
            {"AZURE_OPENAI_ENDPOINT": ""},
            {"AZURE_OPENAI_ENDPOINT": "http://example.openai.azure.com"},
            {"AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/openai/v1/"},
            {"AZURE_OPENAI_ENDPOINT": "https://user:password@example.com"},
            {"AZURE_OPENAI_DEPLOYMENT": " "},
            {"AZURE_OPENAI_API_VERSION": ""},
            {"AZURE_OPENAI_AUTH": "invalid"},
            {"AZURE_OPENAI_AUTH": "api_key", "AZURE_OPENAI_API_KEY": ""},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                self.make_settings(**overrides)

    def test_other_providers_do_not_require_azure_settings(self):
        self.make_settings(LLM_PROVIDER="openai", AZURE_OPENAI_ENDPOINT="")


class AzureClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = patch.multiple(
            settings,
            LLM_PROVIDER="azure_openai",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="avatar-chat",
            AZURE_OPENAI_AUTH="entra",
            AZURE_OPENAI_API_KEY="",
            AZURE_MANAGED_IDENTITY_CLIENT_ID=None,
        )
        self.config.start()
        self.addCleanup(self.config.stop)
        self.credential_patch = patch("azure.identity.aio.DefaultAzureCredential")
        self.credential = self.credential_patch.start().return_value
        self.credential.close = AsyncMock()
        self.addCleanup(self.credential_patch.stop)
        self.provider_patch = patch("azure.identity.aio.get_bearer_token_provider")
        self.provider = self.provider_patch.start()
        self.addCleanup(self.provider_patch.stop)
        self.client_patch = patch("app.services.llm.openai.AsyncAzureOpenAI")
        self.client_factory = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.client = self.client_factory.return_value
        self.client.close = AsyncMock()

    async def test_entra_client_uses_refreshable_provider_and_deployment(self):
        service = LLMService()
        self.provider.assert_called_once_with(
            self.credential, "https://cognitiveservices.azure.com/.default"
        )
        self.assertIs(
            self.client_factory.call_args.kwargs["azure_ad_token_provider"],
            self.provider.return_value,
        )
        self.assertIsNone(self.client_factory.call_args.kwargs["api_key"])
        self.client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Bonjour"))]
            )
        )
        result = await service.generate_response([{"role": "user", "content": "Salut"}])
        self.assertEqual(result, "Bonjour")
        self.assertEqual(
            self.client.chat.completions.create.call_args.kwargs["model"], "avatar-chat"
        )
        await service.close()
        self.client.close.assert_awaited_once()
        self.credential.close.assert_awaited_once()

    async def test_streaming_reuses_existing_chunk_handling(self):
        async def chunks():
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Bonjour"))]
            )

        self.client.chat.completions.create = AsyncMock(return_value=chunks())
        service = LLMService()
        result = [part async for part in service.stream_response(
            [{"role": "user", "content": "Salut"}]
        )]
        self.assertEqual(result, ["Bonjour"])
        self.assertTrue(self.client.chat.completions.create.call_args.kwargs["stream"])
        await service.close()

    async def test_api_key_mode_does_not_create_azure_credential(self):
        with patch.multiple(settings, AZURE_OPENAI_AUTH="api_key", AZURE_OPENAI_API_KEY="test-key"):
            service = LLMService()
        self.provider.assert_not_called()
        self.assertEqual(self.client_factory.call_args.kwargs["api_key"], "test-key")
        await service.close()
        self.credential.close.assert_not_awaited()

    async def test_azure_auth_errors_keep_existing_error_contract(self):
        response = httpx.Response(401, request=httpx.Request("POST", "https://example.com"))
        self.client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError("Denied", response=response, body=None)
        )
        service = LLMService()
        with self.assertRaises(LLMAuthError):
            await service.generate_response([{"role": "user", "content": "Salut"}])
        await service.close()

    async def test_credential_is_closed_even_if_client_close_fails(self):
        service = LLMService()
        self.client.close.side_effect = RuntimeError("close failed")
        with self.assertRaises(RuntimeError):
            await service.close()
        self.credential.close.assert_awaited_once()

    async def test_entra_token_failure_is_an_auth_error_for_both_paths(self):
        self.client.chat.completions.create = AsyncMock(
            side_effect=ClientAuthenticationError("No Azure identity available")
        )
        service = LLMService()
        with self.assertRaises(LLMAuthError):
            await service.generate_response([{"role": "user", "content": "Salut"}])
        with self.assertRaises(LLMAuthError):
            async for _ in service.stream_response([{"role": "user", "content": "Salut"}]):
                pass
        await service.close()


class AzureSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_service_initializes_and_closes_without_token_request(self):
        with patch.multiple(
            settings,
            LLM_PROVIDER="azure_openai",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="avatar-chat",
            AZURE_OPENAI_AUTH="entra",
        ):
            service = LLMService()
        self.assertEqual(service.model, "avatar-chat")
        await service.close()

    async def test_real_sdk_awaits_token_provider_without_network(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "avatar-chat",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Bonjour"},
                }],
            })

        token = AsyncMock(return_value="offline-entra-token")
        async with openai.AsyncAzureOpenAI(
            azure_endpoint="https://example.openai.azure.com",
            api_version="2024-10-21",
            azure_ad_token_provider=token,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            response = await client.chat.completions.create(
                model="avatar-chat", messages=[{"role": "user", "content": "Salut"}]
            )
        self.assertEqual(response.choices[0].message.content, "Bonjour")
        token.assert_awaited_once()
        self.assertEqual(requests[0].headers["Authorization"], "Bearer offline-entra-token")
        self.assertNotIn("api-key", requests[0].headers)
        self.assertIn("/deployments/avatar-chat/chat/completions", requests[0].url.path)


class ExistingProviderTests(unittest.TestCase):
    def test_existing_providers_keep_their_clients_and_models(self):
        for provider in ("anthropic", "openai", "ollama"):
            with self.subTest(provider=provider), patch.multiple(
                settings, LLM_PROVIDER=provider, LLM_MODEL="existing-model"
            ), patch("app.services.llm.anthropic.AsyncAnthropic") as anthropic_client, patch(
                "app.services.llm.openai.AsyncOpenAI"
            ) as openai_client:
                service = LLMService()
                factory = anthropic_client if provider == "anthropic" else openai_client
                self.assertIs(service.client, factory.return_value)
                self.assertEqual(service.model, "existing-model")


if __name__ == "__main__":
    unittest.main()
