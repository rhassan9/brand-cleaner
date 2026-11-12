"""
Approach C (optional): an LLM judges only the borderline pairs of approach A+B.

Runs only with `python -m eval.run_product_eval --use-llm`. Two providers:

    gemini     (default) Google Gemini Interactions API over plain HTTPS,
               key from GEMINI_API_KEY, model gemini-3.5-flash-lite
    anthropic  Anthropic Messages API via the `anthropic` SDK,
               key from ANTHROPIC_API_KEY, model claude-opus-5-5

Safeguards:
- only test pairs whose A+B score is within LLM_MARGIN of the A+B threshold
  are sent, closest first, and never more than --max-llm-calls of them;
- the number of calls and an estimated cost are printed and you must type
  "yes" before anything is sent (or pass --yes);
- the API key is read from an environment variable, never from code or
  files in this repo, and is never written to the cache;
- every answer is cached in eval/results/llm_judgements.jsonl, so reruns (and
  anyone reproducing the reported numbers) need no new calls and no key.

Only product titles from the public Abt-Buy benchmark are sent.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

LLM_MARGIN = 0.05
DEFAULT_PROVIDER = "gemini"
PROVIDERS = {
    "gemini": {"env": "GEMINI_API_KEY", "default_model": "gemini-3.5-flash-lite"},
    "anthropic": {"env": "ANTHROPIC_API_KEY", "default_model": "claude-opus-5-5"},
}
DEFAULT_MODEL = PROVIDERS[DEFAULT_PROVIDER]["default_model"]

# USD per million tokens (input, output). Anthropic first-party rates. Gemini is
# used on Google's free tier, which has no charge within its quota.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
FREE_TIER_NOTE = {"gemini": "no charge on the Gemini free tier (within its rate limits)"}

# Conservative per-call assumptions for the estimate: prompt is about 4 chars
# per token plus overhead; output includes thinking at a low level.
EST_PROMPT_OVERHEAD_TOKENS = 120
EST_OUTPUT_TOKENS = 400
MAX_TOKENS = 2000

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GEMINI_SEED = 42
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_WAITS = [10, 20, 40, 80, 160]       # seconds between attempts on overload / rate limit
DEFAULT_CALL_DELAY = {"gemini": 4.0, "anthropic": 0.0}   # pacing for free-tier rate limits

PROMPT = (
    "Two product listings from different online electronics shops.\n\n"
    "Listing A: {a}\n"
    "Listing B: {b}\n\n"
    "Do both listings refer to the same real-world product (same manufacturer "
    "and model)? An accessory for the product, a bundle, or a different model "
    "is not the same product.\n"
    "Answer with exactly one word: YES or NO."
)


class LLMCallError(Exception):
    """A call failed for good (not retryable, or retries used up)."""


def select_borderline(scores: List[float], threshold: float, max_calls: int,
                      margin: float = LLM_MARGIN) -> List[int]:
    """Indices whose score is within margin of the threshold, closest first, capped."""
    idx = [i for i, s in enumerate(scores) if abs(s - threshold) <= margin]
    idx.sort(key=lambda i: (abs(scores[i] - threshold), i))
    return idx[:max_calls]


def apply_verdicts(preds: List, border: List[int], items: List[dict],
                   verdicts: Dict[Tuple[str, str], Optional[bool]]) -> Tuple[List, int]:
    """
    Replace the threshold decision with the LLM verdict for each borderline pair.
    YES keeps the candidate, NO abstains, and an unparsed answer keeps the
    original decision. Returns (new predictions, number of unparsed answers).
    """
    out = list(preds)
    n_unparsed = 0
    for k, it in zip(border, items):
        v = verdicts[(str(it["abt_id"]), str(it["buy_id"]))]
        if v is None:
            n_unparsed += 1
        else:
            out[k] = it["buy_id"] if v else None
    return out, n_unparsed


def cache_covers(items: List[dict], model: str, cache_path: str) -> bool:
    """True when every item already has a cached answer for this model (no calls needed)."""
    cache = load_cache(cache_path)
    return bool(items) and all((model, str(it["abt_id"]), str(it["buy_id"])) in cache for it in items)


def verdict_effects(golds: List, before: List, after: List, border: List[int], items: List[dict],
                    verdicts: Dict[Tuple[str, str], Optional[bool]]) -> Dict[str, int]:
    """
    What the LLM changed on the borderline rows, for the report.
    llm_correct counts verdicts that agree with the gold label of the candidate
    (YES for a gold match, NO otherwise).
    """
    from eval.metrics import ABSTAIN_MISSED, CORRECT_MATCH, WRONG_MATCH, classify, gold_set
    eff = {"sent": len(border), "answered": 0, "llm_correct": 0, "yes": 0, "no": 0,
           "changed": 0, "missed_to_correct": 0, "wrong_to_abstain": 0,
           "correct_to_missed": 0, "abstain_to_wrong": 0}
    for k, it in zip(border, items):
        v = verdicts[(str(it["abt_id"]), str(it["buy_id"]))]
        if v is None:
            continue
        eff["answered"] += 1
        eff["yes" if v else "no"] += 1
        eff["llm_correct"] += int(v == (it["buy_id"] in gold_set(golds[k])))
        b, a = classify(golds[k], before[k]), classify(golds[k], after[k])
        if before[k] != after[k]:
            eff["changed"] += 1
        if b == ABSTAIN_MISSED and a == CORRECT_MATCH:
            eff["missed_to_correct"] += 1
        elif b == WRONG_MATCH and a != WRONG_MATCH:
            eff["wrong_to_abstain"] += 1
        elif b == CORRECT_MATCH and a != CORRECT_MATCH:
            eff["correct_to_missed"] += 1
        elif b != WRONG_MATCH and a == WRONG_MATCH:
            eff["abstain_to_wrong"] += 1
    return eff


def estimate_cost(pairs: List[Tuple[str, str]], model: str) -> Tuple[int, int, Optional[float]]:
    tin = sum(EST_PROMPT_OVERHEAD_TOKENS + (len(a) + len(b)) // 4 for a, b in pairs)
    tout = EST_OUTPUT_TOKENS * len(pairs)
    price = PRICES.get(model)
    cost = None if price is None else (tin * price[0] + tout * price[1]) / 1_000_000
    return tin, tout, cost


def parse_answer(text: str) -> Optional[bool]:
    words = text.strip().upper().replace(".", " ").split()
    if not words:
        return None
    if words[0] == "YES":
        return True
    if words[0] == "NO":
        return False
    return None


def load_cache(path: str) -> Dict[Tuple[str, str, str], dict]:
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[(r["model"], str(r["abt_id"]), str(r["buy_id"]))] = r
    return cache


# ------------------------------------------------------------------ Gemini

def http_post_json(url: str, headers: Dict[str, str], body: dict, timeout: float = 120) -> Tuple[int, dict]:
    """POST JSON and return (status, parsed body). HTTP errors are returned, not raised."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return e.code, {"error": {"message": raw[:200]}}


def gemini_text(data: dict) -> str:
    """Text of the model output steps in an Interactions API response."""
    parts = []
    for step in data.get("steps") or []:
        if step.get("type") == "model_output":
            for block in step.get("content") or []:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "".join(parts)


def call_gemini(key: str, model: str, prompt: str, post: Callable = http_post_json,
                sleep: Callable = time.sleep) -> dict:
    body = {
        "model": model,
        "input": prompt,
        "store": False,
        "generation_config": {"thinking_level": "low", "max_output_tokens": MAX_TOKENS, "seed": GEMINI_SEED},
    }
    for attempt in range(len(RETRY_WAITS) + 1):
        status, data = post(GEMINI_URL, {"x-goog-api-key": key}, body)
        if status == 200:
            usage = data.get("usage") or {}
            return {
                "text": gemini_text(data),
                "stop_reason": data.get("status"),
                "served_by": data.get("model"),
                "input_tokens": usage.get("total_input_tokens"),
                "output_tokens": usage.get("total_output_tokens"),
                "thought_tokens": usage.get("total_thought_tokens"),
            }
        msg = ((data.get("error") or {}).get("message") or "")[:200]
        if status in RETRY_STATUS and attempt < len(RETRY_WAITS):
            wait = RETRY_WAITS[attempt]
            print(f"    HTTP {status} ({msg[:80]}); retrying in {wait}s")
            sleep(wait)
            continue
        raise LLMCallError(f"HTTP {status}: {msg}")
    raise LLMCallError("retries used up")


# ------------------------------------------------------------------ Anthropic

def make_anthropic_caller(key: str):
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    def call(model: str, prompt: str) -> dict:
        try:
            resp = client.beta.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIStatusError as e:
            raise LLMCallError(f"{e.status_code} {e.message}")
        except anthropic.APIConnectionError:
            raise LLMCallError("network error")
        return {
            "text": "".join(b.text for b in resp.content if b.type == "text"),
            "stop_reason": resp.stop_reason,
            "served_by": resp.model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
    return call


# ------------------------------------------------------------------ driver

def judge_pairs(items: List[dict], model: str, cache_path: str, assume_yes: bool,
                provider: str = DEFAULT_PROVIDER, call_delay: Optional[float] = None,
                post: Callable = http_post_json, sleep: Callable = time.sleep
                ) -> Dict[Tuple[str, str], Optional[bool]]:
    """
    items: dicts with abt_id, buy_id, abt_title, buy_title.
    Returns {(abt_id, buy_id): True/False/None}. None means no usable answer.
    post and sleep are injectable so tests can run without the network.
    """
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown LLM provider {provider!r}; choose from {sorted(PROVIDERS)}.")
    env = PROVIDERS[provider]["env"]
    delay = DEFAULT_CALL_DELAY[provider] if call_delay is None else call_delay
    cache = load_cache(cache_path)
    todo = [it for it in items if (model, str(it["abt_id"]), str(it["buy_id"])) not in cache]
    print(f"\nLLM judge ({provider}: {model}): {len(items)} borderline pairs, "
          f"{len(items) - len(todo)} cached, {len(todo)} new calls needed.")
    if todo:
        tin, tout, cost = estimate_cost([(it["abt_title"], it["buy_title"]) for it in todo], model)
        if cost is not None:
            cost_txt = f"about ${cost:.2f}"
        else:
            cost_txt = FREE_TIER_NOTE.get(provider, "unknown (model not in price table)")
        print(f"Estimated usage: ~{tin} input tokens, up to ~{tout} output tokens, cost {cost_txt}.")
        key = os.environ.get(env)
        if not key:
            raise SystemExit(f"{env} is not set; cannot call the LLM.")
        if not assume_yes:
            reply = input("Send these calls? Type yes to continue: ").strip().lower()
            if reply != "yes":
                raise SystemExit("Cancelled; no calls were made.")
        if provider == "gemini":
            def call(m, p):
                return call_gemini(key, m, p, post=post, sleep=sleep)
        else:
            call = make_anthropic_caller(key)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # The cache file is opened per answer, so a failed first call leaves no file.
        for n, it in enumerate(todo, 1):
            if n > 1 and delay:
                sleep(delay)
            prompt = PROMPT.format(a=it["abt_title"], b=it["buy_title"])
            try:
                res = call(model, prompt)
            except LLMCallError as e:
                raise SystemExit(f"API error after {n - 1} calls: {e}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise SystemExit(f"Network error after {n - 1} calls: {e}")
            rec = {
                "provider": provider, "model": model,
                "abt_id": str(it["abt_id"]), "buy_id": str(it["buy_id"]),
                "answer_text": res["text"].strip()[:50], "same": parse_answer(res["text"]),
                **{k: v for k, v in res.items() if k != "text"},
            }
            with open(cache_path, "a", encoding="utf-8") as out:
                out.write(json.dumps(rec) + "\n")
            cache[(model, rec["abt_id"], rec["buy_id"])] = rec
            if n % 5 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} calls done")
        new = [cache[(model, str(i["abt_id"]), str(i["buy_id"]))] for i in todo]
        used_in = sum(r.get("input_tokens") or 0 for r in new)
        used_out = sum(r.get("output_tokens") or 0 for r in new)
        used_thought = sum(r.get("thought_tokens") or 0 for r in new)
        extra = f", {used_thought} thinking" if used_thought else ""
        print(f"Actual usage for new calls: {used_in} input, {used_out} output{extra} tokens.")
    return {(str(it["abt_id"]), str(it["buy_id"])): cache[(model, str(it["abt_id"]), str(it["buy_id"]))]["same"]
            for it in items}


if __name__ == "__main__":
    sys.exit("Run python -m eval.run_product_eval --use-llm instead.")
