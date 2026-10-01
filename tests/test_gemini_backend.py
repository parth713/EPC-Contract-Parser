"""Offline check of the Gemini adapter: request construction and response parsing (no network)."""
import asyncio

from google.genai import types

from epc_parser.config import Settings
from epc_parser.llm import GeminiBackend, LLMClient
from epc_parser.models import PageLabelLLM


def test_gemini_backend_roundtrip(tmp_path):
    s = Settings(api_key="fake")
    s.cache_dir = tmp_path
    s.model_escalation = "gemini-3.1-pro-preview"  # exercise the pro: minimal -> low upgrade branch
    backend = GeminiBackend(s)
    seen = {}

    async def fake_generate_content(*, model, contents, config):
        seen.update(model=model, config=config, parts=contents[0].parts)
        return types.GenerateContentResponse(
            candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="thinking...", thought=True),
                                                           types.Part(text='{"printed_page_label": "GCC-14"}')]),
                finish_reason=types.FinishReason.STOP)],
            usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=1200, candidates_token_count=20,
                                                                      thoughts_token_count=5))

    backend.client.aio.models.generate_content = fake_generate_content
    llm = LLMClient(s, backend)
    res = asyncio.run(llm.call("label", role="escalation", system="sys", parts=[b"\xff\xd8jpeg", "user"],
                               schema=PageLabelLLM, thinking="minimal", media_resolution="high", max_output_tokens=64))
    assert res.ok and res.parsed.printed_page_label == "GCC-14"
    assert seen["model"] == s.model_escalation
    assert seen["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW  # pro: minimal -> low
    assert seen["config"].media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH
    assert seen["parts"][0].inline_data.mime_type == "image/jpeg"
    report = llm.cost.report()
    assert report["by_model"][s.model_escalation]["input_tokens"] == 1200
    # cached on second call
    res2 = asyncio.run(llm.call("label", role="escalation", system="sys", parts=[b"\xff\xd8jpeg", "user"],
                                schema=PageLabelLLM, thinking="minimal", media_resolution="high", max_output_tokens=64))
    assert res2.from_cache


def test_truncation_is_reported(tmp_path):
    s = Settings(api_key="fake")
    s.cache_dir = tmp_path
    backend = GeminiBackend(s)

    async def fake(*, model, contents, config):
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text='{"printed_')]),
            finish_reason=types.FinishReason.MAX_TOKENS)])

    backend.client.aio.models.generate_content = fake
    res = asyncio.run(LLMClient(s, backend).call("x", role="reader", system=None, parts=["u"], schema=PageLabelLLM))
    assert not res.ok and res.truncated
