"""
The optional LLM judge (approach C), Gemini provider, with fake HTTP and a fake clock.

No network call, no real key, no real waiting. The real-API run is separate
and its answers are cached in eval/results/llm_judgements.jsonl.
"""
import json
import os

import pytest

from eval import product_llm
from eval.product_llm import GEMINI_URL, call_gemini, gemini_text, judge_pairs

FAKE_KEY = "fake-gemini-key-not-real"
ITEMS = [
    {"abt_id": 1, "buy_id": 101, "abt_title": "Sony Turntable - PSLX350H",
     "buy_title": "Sony PS-LX350H Belt-Drive Turntable"},
    {"abt_id": 2, "buy_id": 202, "abt_title": "OmniMount TV Top Shelf Mount - CCH1B",
     "buy_title": "OmniMount TV Top Shelf Mount - CCH1P"},
]


def ok(text, model="gemini-3.8-flash"):
    """A successful Interactions API response, shaped like the documented example."""
    return 200, {
        "id": "v1_x", "model": model, "object": "interaction", "status": "completed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": {"total_input_tokens": 80, "total_output_tokens": 1, "total_thought_tokens": 30},
    }


def overloaded():
    return 503, {"error": {"message": "gemini-3.8-flash is currently experiencing high demand",
                           "code": "service_unavailable"}}


class FakeHTTP:
    """Returns scripted (status, body) responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, headers, body, timeout=120):
        self.requests.append({"url": url, "headers": headers, "body": body})
        return self.responses.pop(0)


class FakeClock:
    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


@pytest.fixture
def cache(tmp_path):
    return str(tmp_path / "results" / "llm_judgements.jsonl")


def test_request_shape_and_answer_parsing(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    http, clock = FakeHTTP([ok("YES"), ok("NO.")]), FakeClock()
    v = judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True, post=http, sleep=clock)

    assert v == {("1", "101"): True, ("2", "202"): False}
    req = http.requests[0]
    assert req["url"] == GEMINI_URL
    assert req["headers"] == {"x-goog-api-key": FAKE_KEY}      # key only in the header
    body = req["body"]
    assert body["model"] == "gemini-3.8-flash"
    assert body["store"] is False                              # Google does not keep the interaction
    assert body["generation_config"] == {"thinking_level": "low", "max_output_tokens": 2000, "seed": 42}
    assert "Sony Turntable - PSLX350H" in body["input"] and "YES or NO" in body["input"]
    assert FAKE_KEY not in json.dumps(body)


def test_free_tier_is_paced_between_calls(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    clock = FakeClock()
    judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True,
                post=FakeHTTP([ok("YES"), ok("NO")]), sleep=clock)
    assert clock.waits == [product_llm.DEFAULT_CALL_DELAY["gemini"]]   # one gap between two calls


def test_cache_records_usage_and_never_the_key(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True,
                post=FakeHTTP([ok("YES"), ok("Maybe")]), sleep=FakeClock())
    text = open(cache, encoding="utf-8").read()
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[0] == {
        "provider": "gemini", "model": "gemini-3.8-flash", "abt_id": "1", "buy_id": "101",
        "answer_text": "YES", "same": True, "stop_reason": "completed", "served_by": "gemini-3.8-flash",
        "input_tokens": 80, "output_tokens": 1, "thought_tokens": 30,
    }
    assert rows[1]["same"] is None
    assert FAKE_KEY not in text


def test_rerun_uses_cache_without_key_or_network(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    first = judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True,
                        post=FakeHTTP([ok("YES"), ok("NO")]), sleep=FakeClock())
    monkeypatch.delenv("GEMINI_API_KEY")
    http = FakeHTTP([])
    assert judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=False, post=http) == first
    assert http.requests == []


def test_overload_is_retried_with_backoff(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    http, clock = FakeHTTP([overloaded(), overloaded(), ok("YES")]), FakeClock()
    v = judge_pairs(ITEMS[:1], "gemini-3.8-flash", cache, assume_yes=True, post=http, sleep=clock)
    assert v == {("1", "101"): True}
    assert len(http.requests) == 3 and clock.waits == [10, 20]


def test_overload_that_never_ends_stops_cleanly_and_writes_nothing(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    n = len(product_llm.RETRY_WAITS) + 1
    http, clock = FakeHTTP([overloaded()] * n), FakeClock()
    with pytest.raises(SystemExit, match="API error after 0 calls: HTTP 503"):
        judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True, post=http, sleep=clock)
    assert len(http.requests) == n and clock.waits == product_llm.RETRY_WAITS
    assert not os.path.exists(cache)


def test_client_errors_are_not_retried(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    http, clock = FakeHTTP([(404, {"error": {"message": "model not found"}})]), FakeClock()
    with pytest.raises(SystemExit, match="HTTP 404: model not found"):
        judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True, post=http, sleep=clock)
    assert len(http.requests) == 1 and clock.waits == []


def test_failure_mid_run_keeps_answers_already_received(monkeypatch, cache):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    http = FakeHTTP([ok("YES"), (400, {"error": {"message": "bad request"}})])
    with pytest.raises(SystemExit, match="API error after 1 calls"):
        judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True, post=http, sleep=FakeClock())
    rows = [json.loads(line) for line in open(cache, encoding="utf-8")]
    assert [(r["abt_id"], r["same"]) for r in rows] == [("1", True)]


def test_missing_key_and_free_tier_message(monkeypatch, cache, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    http = FakeHTTP([])
    with pytest.raises(SystemExit, match="GEMINI_API_KEY is not set"):
        judge_pairs(ITEMS, "gemini-3.8-flash", cache, assume_yes=True, post=http)
    assert http.requests == []
    assert "free tier" in capsys.readouterr().out


def test_unknown_provider_is_rejected(cache):
    with pytest.raises(SystemExit, match="Unknown LLM provider"):
        judge_pairs(ITEMS, "x", cache, assume_yes=True, provider="openai")


def test_gemini_text_reads_only_model_output_text():
    data = {"steps": [
        {"type": "user_input", "content": [{"type": "text", "text": "question"}]},
        {"type": "thought", "content": [{"type": "text", "text": "thinking"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "YES"}]},
    ]}
    assert gemini_text(data) == "YES"
    assert gemini_text({}) == ""


def test_call_gemini_returns_usage_fields():
    res = call_gemini(FAKE_KEY, "gemini-3.8-flash", "p", post=FakeHTTP([ok("NO")]), sleep=FakeClock())
    assert res == {"text": "NO", "stop_reason": "completed", "served_by": "gemini-3.8-flash",
                   "input_tokens": 80, "output_tokens": 1, "thought_tokens": 30}
