import pytest

from docsense.llm.base import FakeProvider, LLMProvider
from docsense.llm.factory import get_provider


def test_fake_provider_satisfies_protocol():
    assert isinstance(FakeProvider(), LLMProvider)


def test_factory_returns_fake():
    provider = get_provider("fake")
    assert provider.name == "fake"


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider("gpt-neo")


def test_factory_reads_env(monkeypatch):
    from docsense import settings

    monkeypatch.setenv("LLM_PROVIDER", "fake")
    settings.get_settings.cache_clear()
    try:
        assert get_provider().name == "fake"
    finally:
        settings.get_settings.cache_clear()


def test_fake_complete_echoes_prompt():
    provider = FakeProvider()
    answer = provider.complete("What is the revenue?", system="sys", max_tokens=10)
    assert "What is the revenue?" in answer
    assert provider.calls[0]["system"] == "sys"


def test_stream_concatenates_to_complete():
    provider = FakeProvider(canned="streamed answer text")
    assert "".join(provider.stream("q")) == "streamed answer text"
