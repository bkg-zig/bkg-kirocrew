"""BKG API TTS provider tests; all HTTP traffic stays on a loopback test server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from aiohttp import web
import pytest

import kiro_crew.voice_reply as voice_reply


@asynccontextmanager
async def _bkg_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio: bytes = b"ID3-test-mp3",
    voice_rows: list[object] | None = None,
    synth_status: int = 200,
):
    state: dict[str, Any] = {"voice_auth": None, "synth_auth": None, "payload": None}

    async def voices(request: web.Request) -> web.Response:
        state["voice_auth"] = request.headers.get("Authorization")
        rows = voice_rows if voice_rows is not None else [
            {"id": "de-DE-Neural-A", "name": "Mia", "language": "de-DE"},
            "de-DE-Neural-B",
        ]
        return web.json_response({"voices": rows})

    async def synthesize(request: web.Request) -> web.Response:
        state["synth_auth"] = request.headers.get("Authorization")
        state["payload"] = await request.json()
        if synth_status != 200:
            return web.Response(status=synth_status, text="private upstream detail")
        return web.Response(body=audio, content_type="audio/mpeg")

    app = web.Application()
    app.router.add_get("/v1/voices", voices)
    app.router.add_post("/v1/synthesize/stream", synthesize)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]

    monkeypatch.setenv("BKG_API_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("BKG_API_KEY", "unit-test-bkg-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        yield state
    finally:
        await runner.cleanup()


def test_bkg_provider_is_valid_and_resolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BKG_API_KEY", "unit-test-bkg-key")
    assert voice_reply.PROVIDER_BKG_API == "bkg-api"
    assert voice_reply.PROVIDER_BKG_API in voice_reply.VALID_PROVIDERS
    assert voice_reply.resolve_configured_provider({"provider": "bkg-api"}) == "bkg-api"
    assert voice_reply.is_available("bkg-api") is True


def test_bkg_provider_does_not_reuse_anthropic_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Secret:
        def reveal(self) -> str:
            return "vault-bkg-key"

    class Vault:
        def __init__(self, config_path: Path) -> None:
            assert config_path == tmp_path

        def get(self, name: str) -> Secret:
            assert name == "BKG_APIKEY"
            return Secret()

    monkeypatch.delenv("BKG_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-tts-key")
    monkeypatch.setattr("kiro_crew.config.paths.config_dir", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.secrets.SecretVault", Vault)

    assert voice_reply._read_bkg_api_key_sync() == "vault-bkg-key"


@pytest.mark.asyncio
async def test_bkg_catalogue_and_mp3_synthesis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with _bkg_api(monkeypatch) as state:
        voices = await voice_reply.list_bkg_api_voices()
        assert voices == [
            {
                "id": "de-DE-Neural-A",
                "name": "Mia",
                "language": "de-DE",
                "languageCode": "de-DE",
            },
            {
                "id": "de-DE-Neural-B",
                "name": "de-DE-Neural-B",
                "language": "",
                "languageCode": "",
            },
        ]
        assert state["voice_auth"] == "Bearer unit-test-bkg-key"

        audio_path = await voice_reply.synthesize_speech(
            "**Hallo Welt**",
            provider=voice_reply.PROVIDER_BKG_API,
            voice_id="de-DE-Neural-A",
            rate="110%",
            pitch="+10%",
        )
        assert audio_path is not None
        try:
            assert audio_path.endswith(".mp3")
            assert Path(audio_path).read_bytes() == b"ID3-test-mp3"
        finally:
            Path(audio_path).unlink(missing_ok=True)

        assert state["synth_auth"] == "Bearer unit-test-bkg-key"
        assert state["payload"] == {
            "text": "Hallo Welt",
            "voice_id": "de-DE-Neural-A",
            "rate": "110",
            "pitch": "+10%",
            "format": "mp3",
            "sample_rate": 22050,
        }


@pytest.mark.asyncio
async def test_bkg_synthesis_rejects_oversized_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _bkg_api(monkeypatch, audio=b"12345"):
        monkeypatch.setattr(voice_reply, "BKG_API_MAX_AUDIO_BYTES", 4)
        with pytest.raises(RuntimeError, match="response limit"):
            await voice_reply.synthesize_speech("hello", provider=voice_reply.PROVIDER_BKG_API)


@pytest.mark.asyncio
async def test_bkg_catalogue_rejects_empty_voices(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _bkg_api(monkeypatch, voice_rows=[]):
        with pytest.raises(RuntimeError, match="empty voice catalogue"):
            await voice_reply.list_bkg_api_voices()


@pytest.mark.asyncio
async def test_bkg_synthesis_error_does_not_echo_upstream_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _bkg_api(monkeypatch, synth_status=401):
        with pytest.raises(voice_reply.BkgApiError) as exc_info:
            await voice_reply.synthesize_speech("hello", provider=voice_reply.PROVIDER_BKG_API)
        message = str(exc_info.value)
        assert "HTTP 401" in message
        assert "private upstream detail" not in message
