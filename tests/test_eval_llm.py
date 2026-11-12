"""
The optional LLM judge (approach C), Anthropic provider, tested with a fake `anthropic` module.

No network call is made and no real key is used: the fake module is put in
sys.modules, so `import anthropic` inside judge_pairs gets it. These tests do
not prove the real API accepts the request; they cover everything around it:
the request we build, key handling, the confirmation prompt, parsing,
caching and how verdicts change the predictions.
"""
import json
import sys
import types
from types import SimpleNamespace

import pytest

from eval import product_llm
from eval.metrics import evaluate
from eval.product_llm import apply_verdicts, judge_pairs

FAKE_KEY = "fake-test-key-not-real"


class FakeAPIStatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class FakeAPIConnectionError(Exception):
    pass


def install_fake_anthropic(monkeypatch, answers=None, error=None):
    """
    answers: text the fake model returns, one per call, in order.
    error: an exception to raise on the first call instead.
    Returns a dict that records the client kwargs and every request.
    """
    log = {"client_kwargs": None, "requests": []}
    answers = list(answers or [])

    class FakeMessages:
        def create(self, **kwargs):
            log["requests"].append(kwargs)
            if error is not None:
                raise error
            text = answers.pop(0)
            return SimpleNamespace(
                content=[SimpleNamespace(type="thinking", thinking=""),
                         SimpleNamespace(type="text", text=text)],
                stop_reason="end_turn",
                model=kwargs["model"],
                usage=SimpleNamespace(input_tokens=100, output_tokens=7),
            )

    class FakeAnthropic:
        def __init__(self, **kwargs):
            log["client_kwargs"] = kwargs
            self.beta = SimpleNamespace(messages=FakeMessages())

    fake = types.ModuleType("anthropic")
    fake.Anthropic = FakeAnthropic
    fake.APIStatusError = FakeAPIStatusError
    fake.APIConnectionError = FakeAPIConnectionError
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return log


ITEMS = [
    {"abt_id": 1, "buy_id": 101, "abt_title": "Sony Turntable - PSLX350H",
     "buy_title": "Sony PS-LX350H Belt-Drive Turntable"},
    {"abt_id": 2, "buy_id": 202, "abt_title": "OmniMount TV Top Shelf Mount - CCH1B",
     "buy_title": "OmniMount TV Top Shelf Mount - CCH1P"},
    {"abt_id": 3, "buy_id": 303, "abt_title": "Canon Black Ink Cartridge - PG40BK",
     "buy_title": "Canon PG-40 Black Ink Cartridge - PG-40-BK-TWIN"},
]


@pytest.fixture
def cache(tmp_path):
    return str(tmp_path / "results" / "llm_judgements.jsonl")


def test_sends_one_request_per_pair_with_the_expected_shape(monkeypatch, cache):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    log = install_fake_anthropic(monkeypatch, answers=["YES", "NO", "Maybe"])

    verdicts = judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=True, provider="anthropic")

    assert verdicts == {("1", "101"): True, ("2", "202"): False, ("3", "303"): None}
    assert log["client_kwargs"] == {"api_key": FAKE_KEY}       # key comes from the env var
    assert len(log["requests"]) == 3
    req = log["requests"][0]
    assert req["model"] == "claude-opus-5-5"
    assert req["max_tokens"] == product_llm.MAX_TOKENS
    assert req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in req and "temperature" not in req
    prompt = req["messages"][0]["content"]
    assert req["messages"][0]["role"] == "user"
    assert "Sony Turntable - PSLX350H" in prompt and "Sony PS-LX350H Belt-Drive Turntable" in prompt


def test_answers_are_cached_and_the_key_is_never_written(monkeypatch, cache):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    install_fake_anthropic(monkeypatch, answers=["YES", "NO", "Maybe"])
    judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=True, provider="anthropic")

    with open(cache, encoding="utf-8") as f:
        text = f.read()
    rows = [json.loads(line) for line in text.splitlines()]
    assert [(r["abt_id"], r["buy_id"], r["same"]) for r in rows] == [
        ("1", "101", True), ("2", "202", False), ("3", "303", None)]
    assert rows[0]["answer_text"] == "YES" and rows[0]["output_tokens"] == 7
    assert FAKE_KEY not in text


def test_second_run_uses_the_cache_and_needs_no_key(monkeypatch, cache):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    install_fake_anthropic(monkeypatch, answers=["YES", "NO", "Maybe"])
    first = judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=True, provider="anthropic")

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    log = install_fake_anthropic(monkeypatch, error=AssertionError("must not call the API"))
    second = judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=False, provider="anthropic")
    assert second == first
    assert log["requests"] == [] and log["client_kwargs"] is None


def test_cache_is_per_model(monkeypatch, cache):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    install_fake_anthropic(monkeypatch, answers=["YES"])
    judge_pairs(ITEMS[:1], "claude-opus-5-5", cache, assume_yes=True, provider="anthropic")
    log = install_fake_anthropic(monkeypatch, answers=["NO"])
    v = judge_pairs(ITEMS[:1], "claude-haiku-5-5", cache, assume_yes=True, provider="anthropic")
    assert v == {("1", "101"): False} and len(log["requests"]) == 1


def test_missing_key_stops_before_any_call(monkeypatch, cache):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    log = install_fake_anthropic(monkeypatch, answers=["YES"] * 3)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set"):
        judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=True, provider="anthropic")
    assert log["requests"] == []


def test_answering_anything_but_yes_cancels(monkeypatch, cache):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    log = install_fake_anthropic(monkeypatch, answers=["YES"] * 3)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    with pytest.raises(SystemExit, match="Cancelled"):
        judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=False, provider="anthropic")
    assert log["requests"] == []


def test_typing_yes_sends_the_calls(monkeypatch, cache):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    log = install_fake_anthropic(monkeypatch, answers=["YES"] * 3)
    monkeypatch.setattr("builtins.input", lambda prompt="": " YES ")
    judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=False, provider="anthropic")
    assert len(log["requests"]) == 3


def test_api_error_on_first_call_stops_and_leaves_no_cache_file(monkeypatch, cache):
    # The real case seen on 2026-11-12: a 400 for an account with no credit.
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    install_fake_anthropic(monkeypatch, error=FakeAPIStatusError(400, "credit balance is too low"))
    with pytest.raises(SystemExit, match="API error after 0 calls: 400"):
        judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=True, provider="anthropic")
    import os
    assert not os.path.exists(cache)


def test_connection_error_is_reported(monkeypatch, cache):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    install_fake_anthropic(monkeypatch, error=FakeAPIConnectionError())
    with pytest.raises(SystemExit, match="API error after 0 calls: network error"):
        judge_pairs(ITEMS, "claude-opus-5-5", cache, assume_yes=True, provider="anthropic")


def test_apply_verdicts_changes_only_borderline_rows():
    preds = [101, None, 303, 404]            # A+B decisions for 4 test rows
    border = [1, 2, 3]                        # rows sent to the LLM
    items = [{"abt_id": 2, "buy_id": 202}, {"abt_id": 3, "buy_id": 303}, {"abt_id": 4, "buy_id": 404}]
    verdicts = {("2", "202"): True, ("3", "303"): False, ("4", "404"): None}
    out, n_unparsed = apply_verdicts(preds, border, items, verdicts)
    assert out == [101, 202, None, 404]       # YES adds, NO removes, unparsed keeps
    assert n_unparsed == 1
    assert preds == [101, None, 303, 404]     # input is not modified


def test_llm_verdicts_flow_into_metrics_hand_computed():
    golds = [{101}, {202}, {999}, {404}]
    preds = [101, None, 303, 404]
    _, before = evaluate(golds, preds)        # 2 correct, 1 wrong, 1 missed
    out, _ = apply_verdicts(preds, [1, 2],
                            [{"abt_id": 2, "buy_id": 202}, {"abt_id": 3, "buy_id": 303}],
                            {("2", "202"): True, ("3", "303"): False})
    _, after = evaluate(golds, out)           # 3 correct, 0 wrong, 1 missed
    assert (before["precision"], before["recall"]) == (2 / 3, 2 / 4)
    assert (after["precision"], after["recall"]) == (1.0, 3 / 4)


def test_cache_covers_only_when_every_pair_is_cached(tmp_path):
    from eval.product_llm import cache_covers
    path = str(tmp_path / "c.jsonl")
    assert not cache_covers(ITEMS, "m", path)                      # no file
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"model": "m", "abt_id": "1", "buy_id": "101", "same": True}) + "\n")
    assert not cache_covers(ITEMS, "m", path)                      # 1 of 3
    assert cache_covers(ITEMS[:1], "m", path)
    assert not cache_covers(ITEMS[:1], "other-model", path)        # cache is per model
    assert not cache_covers([], "m", path)


def test_verdict_effects_hand_computed():
    from eval.product_llm import verdict_effects
    golds = [{101}, {202}, {999}, {404}, {505}]
    before = [101, None, 303, 404, None]       # correct, missed, wrong, correct, missed
    border = [1, 2, 3, 4]
    items = [{"abt_id": 2, "buy_id": 202}, {"abt_id": 3, "buy_id": 303},
             {"abt_id": 4, "buy_id": 404}, {"abt_id": 5, "buy_id": 606}]
    verdicts = {("2", "202"): True,    # missed -> correct (LLM right)
                ("3", "303"): False,   # wrong -> abstain (LLM right)
                ("4", "404"): False,   # correct -> missed (LLM wrong)
                ("5", "606"): None}    # unparsed, kept
    after, _ = apply_verdicts(before, border, items, verdicts)
    e = verdict_effects(golds, before, after, border, items, verdicts)
    assert e == {"sent": 4, "answered": 3, "llm_correct": 2, "yes": 1, "no": 2, "changed": 3,
                 "missed_to_correct": 1, "wrong_to_abstain": 1, "correct_to_missed": 1,
                 "abstain_to_wrong": 0}
