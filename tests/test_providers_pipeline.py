"""Provider adapters (with fake SDK clients, no network) and end-to-end pipeline runs on the synthetic sample."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from promo_finder import providers
from promo_finder.config import load_config
from promo_finder.pipeline import rerank, run
from promo_finder.providers import FileExchangeProvider, PendingResponse, ProviderError

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "synthetic_show"
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


# ---- Anthropic adapter -----------------------------------------------------------------------
class FakeAnthropicMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def fake_anthropic_response(text='{"ok": true}', stop="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, stop_details=None, model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0))


def make_anthropic(monkeypatch, response, **cfg_over):
    import anthropic
    beta = FakeAnthropicMessages(response)
    plain = FakeAnthropicMessages(response)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(beta=SimpleNamespace(messages=beta),
                                                                             messages=plain))
    cfg = load_config(overrides={"provider": {"anthropic": cfg_over}} if cfg_over else None)
    return providers.AnthropicProvider(cfg), beta, plain


def test_anthropic_request_shape(monkeypatch):
    prov, beta, _ = make_anthropic(monkeypatch, fake_anthropic_response())
    res = prov.generate_json("t", "sys", "user", SCHEMA)
    assert res.data == {"ok": True} and res.usage["input_tokens"] == 100
    kw = beta.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "sys"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert kw["output_config"]["effort"] == "medium"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kw and "thinking" not in kw


def test_anthropic_without_fallback_uses_plain_endpoint(monkeypatch):
    prov, beta, plain = make_anthropic(monkeypatch, fake_anthropic_response(), use_refusal_fallback=False)
    prov.generate_json("t", "s", "u", SCHEMA)
    assert plain.calls and not beta.calls and "fallbacks" not in plain.calls[0]


@pytest.mark.parametrize("stop,msg", [("refusal", "declined"), ("max_tokens", "truncated")])
def test_anthropic_bad_stop_reasons(monkeypatch, stop, msg):
    prov, _, _ = make_anthropic(monkeypatch, fake_anthropic_response(stop=stop))
    with pytest.raises(ProviderError, match=msg):
        prov.generate_json("t", "s", "u", SCHEMA)


def test_anthropic_missing_credentials_is_fatal(monkeypatch):
    prov, _, _ = make_anthropic(monkeypatch, TypeError("Could not resolve authentication method"))
    with pytest.raises(ProviderError) as e:
        prov.generate_json("t", "s", "u", SCHEMA)
    assert e.value.fatal


# ---- OpenAI-compatible adapter ---------------------------------------------------------------
def test_openai_compatible_request_shape(monkeypatch):
    import openai
    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(model="m", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3), choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='```json\n{"ok": true}\n```'))])

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    monkeypatch.setenv("TEST_KEY", "x")
    cfg = load_config(overrides={"provider": {"openai_compatible": {"model": "m", "api_key_env": "TEST_KEY"}}})
    res = providers.OpenAICompatibleProvider(cfg).generate_json("t", "s", "u", SCHEMA)
    assert res.data == {"ok": True}
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["messages"][0] == {"role": "system", "content": "s"}


def test_openai_compatible_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    cfg = load_config(overrides={"provider": {"openai_compatible": {"model": "m", "api_key_env": "NOPE_KEY"}}})
    with pytest.raises(ProviderError, match="NOPE_KEY"):
        providers.OpenAICompatibleProvider(cfg)
    with pytest.raises(ProviderError, match="model is empty"):
        providers.OpenAICompatibleProvider(load_config())


# ---- File exchange ---------------------------------------------------------------------------
def test_file_exchange_pending_then_answer_then_stale(tmp_path):
    fx = FileExchangeProvider(tmp_path, responder_label="test")
    with pytest.raises(PendingResponse):
        fx.generate_json("task1", "sys", "user", SCHEMA)
    assert (tmp_path / "requests" / "task1.json").exists() and (tmp_path / "requests" / "task1.md").exists()
    (tmp_path / "responses" / "task1.json").write_text('Sure! ```json\n{"ok": true}\n```', encoding="utf-8")
    assert fx.generate_json("task1", "sys", "user", SCHEMA).data == {"ok": True}
    # prompt changes -> the old answer must not be reused
    with pytest.raises(PendingResponse):
        fx.generate_json("task1", "sys", "DIFFERENT user", SCHEMA)
    assert list((tmp_path / "responses" / "stale").iterdir())


def test_caching_provider_serves_repeat_from_disk(tmp_path):
    class Counting(providers.Provider):
        name, model, n = "fake", "m", 0

        def generate_json(self, task_id, system, user, schema):
            self.n += 1
            return providers.ModelResult(data={"ok": True}, raw_text='{"ok": true}', provider="fake", model="m")

    inner = Counting()
    cp = providers.CachingProvider(inner, tmp_path)
    cp.generate_json("a", "s", "u", SCHEMA)
    again = cp.generate_json("a", "s", "u", SCHEMA)
    assert inner.n == 1 and again.cached


# ---- End-to-end ------------------------------------------------------------------------------
def test_heuristic_end_to_end(tmp_path):
    res = run(SAMPLE, tmp_path, load_config(overrides={"provider": {"name": "heuristic"}}))
    m = res["manifest"]
    assert m["status"] == "complete" and m["episodes_found"] == 2 and m["skipped_files"]
    for name in ("results.json", "summary.md", "report.html", "ranking.csv", "human_review_sheet.csv",
                 "parse_report.json", "run_manifest.json"):
        assert (tmp_path / name).exists(), name
    for c in res["candidates"]:
        assert 20 <= c["metrics"]["total_duration"] <= 60 and c["confidence"] == "low"
    assert "HEURISTIC BASELINE" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def _cand(ep, start, end, hook, cliff, title):
    keys = ["hook", "cliffhanger", "standalone", "emotion_curiosity", "visual", "spoiler_safety"]
    return {"title": title, "segments": [{"episode": ep, "start_id": start, "end_id": end}], "hook_id": hook,
            "hook": "h", "cliffhanger_id": cliff, "cliffhanger": "c", "why_it_works": "w", "stranger_view": "s",
            "evidence_ids": [hook], "scores": {k: {"score": 4, "reason": "r"} for k in keys},
            "context_dependencies": [], "spoiler_risk": "low", "spoiler_note": "", "weaknesses": [],
            "confidence": "high"}


def test_file_provider_end_to_end_with_repair(tmp_path):
    cfg = load_config(overrides={"provider": {"name": "file"}, "review": {"cold_viewer": True}})
    first = run(SAMPLE, tmp_path, cfg, responder_label="unit test")
    assert first["manifest"]["status"] == "incomplete" and first["manifest"]["episodes_pending"] == [1, 2]
    resp = tmp_path / "exchange" / "responses"
    good = _cand(1, "L002", "L017", "L003", "L017", "dead sister")
    too_short = _cand(1, "L002", "L008", "L003", "L008", "too short")
    (resp / "mine_E01.json").write_text(json.dumps({"episode_note": "n", "candidates": [good, too_short]}), encoding="utf-8")
    (resp / "mine_E02.json").write_text(json.dumps({"episode_note": "n", "candidates": [
        _cand(2, "L005", "L013", "L005", "L012", "pier")]}), encoding="utf-8")
    second = run(SAMPLE, tmp_path, cfg, responder_label="unit test")
    m = second["manifest"]
    assert m["status"] == "complete" and "repair_E01_1" in m["tasks_pending"]
    assert {c["id"] for c in second["candidates"]} == {"E01-C1", "E02-C1"}
    # answer the repair with a fixed (longer) version of the short candidate
    (resp / "repair_E01_1.json").write_text(json.dumps({"episode_note": "", "candidates": [
        _cand(1, "L002", "L014", "L003", "L014", "fixed")]}), encoding="utf-8")
    third = run(SAMPLE, tmp_path, cfg, responder_label="unit test")
    ids = {c["id"] for c in third["candidates"]}
    assert "E01-C2r" in ids
    fixed = next(c for c in third["candidates"] if c["id"] == "E01-C2r")
    assert fixed["repaired"] and fixed["confidence"] == "medium"
    # pending montage/review tasks are listed, never silently dropped
    assert "montage" in third["manifest"]["tasks_pending"]
    assert "unit test" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    # rerank is model-free and changes nothing but scores/ordering
    rr = rerank(tmp_path / "results.json", load_config(overrides={"weights": {"visual": 0.9}}), tmp_path / "rr")
    assert set(rr["ranking"]["ranked"]) <= ids


def test_extract_json_strips_think_blocks():
    text = '<think>maybe {"ok": false}</think>' + chr(10) + '{"ok": true}'
    assert providers.extract_json(text) == {"ok": True}
