"""
Approach C (optional): an LLM judges only the borderline pairs of approach A+B.

Runs only with `python -m eval.run_product_eval --use-llm`. Safeguards:
- only test pairs whose A+B score is within LLM_MARGIN of the A+B threshold
  are sent, closest first, and never more than --max-llm-calls of them;
- the number of calls and an estimated cost are printed and you must type
  "yes" before anything is sent (or pass --yes);
- the API key is read from the ANTHROPIC_API_KEY environment variable, never
  from code or files in this repo;
- every answer is cached in eval/results/llm_judgements.jsonl, so reruns (and
  anyone reproducing the reported numbers) need no new calls.

Only product titles from the public Abt-Buy benchmark are sent.
"""
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

DEFAULT_MODEL = "claude-opus-5-5"
LLM_MARGIN = 0.05
# USD per million tokens (input, output), Anthropic first-party rates.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Conservative per-call assumptions for the estimate: prompt is about 4 chars
# per token plus overhead; output includes adaptive thinking at low effort.
EST_PROMPT_OVERHEAD_TOKENS = 120
EST_OUTPUT_TOKENS = 400
MAX_TOKENS = 2000

PROMPT = (
    "Two product listings from different online electronics shops.\n\n"
    "Listing A: {a}\n"
    "Listing B: {b}\n\n"
    "Do both listings refer to the same real-world product (same manufacturer "
    "and model)? An accessory for the product, a bundle, or a different model "
    "is not the same product.\n"
    "Answer with exactly one word: YES or NO."
)


def select_borderline(scores: List[float], threshold: float, max_calls: int,
                      margin: float = LLM_MARGIN) -> List[int]:
    """Indices whose score is within margin of the threshold, closest first, capped."""
    idx = [i for i, s in enumerate(scores) if abs(s - threshold) <= margin]
    idx.sort(key=lambda i: (abs(scores[i] - threshold), i))
    return idx[:max_calls]


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


def judge_pairs(items: List[dict], model: str, cache_path: str, assume_yes: bool) -> Dict[Tuple[str, str], Optional[bool]]:
    """
    items: dicts with abt_id, buy_id, abt_title, buy_title.
    Returns {(abt_id, buy_id): True/False/None}. None means no usable answer.
    """
    cache = load_cache(cache_path)
    todo = [it for it in items if (model, str(it["abt_id"]), str(it["buy_id"])) not in cache]
    print(f"\nLLM judge ({model}): {len(items)} borderline pairs, {len(items) - len(todo)} cached, "
          f"{len(todo)} new calls needed.")
    if todo:
        tin, tout, cost = estimate_cost([(it["abt_title"], it["buy_title"]) for it in todo], model)
        cost_txt = "unknown (model not in price table)" if cost is None else f"about ${cost:.2f}"
        print(f"Estimated usage: ~{tin} input tokens, up to ~{tout} output tokens, cost {cost_txt}.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set; cannot call the LLM.")
        if not assume_yes:
            reply = input("Send these calls? Type yes to continue: ").strip().lower()
            if reply != "yes":
                raise SystemExit("Cancelled; no calls were made.")
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as out:
            for n, it in enumerate(todo, 1):
                prompt = PROMPT.format(a=it["abt_title"], b=it["buy_title"])
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
                    raise SystemExit(f"API error after {n - 1} calls: {e.status_code} {e.message}")
                except anthropic.APIConnectionError:
                    raise SystemExit(f"Network error after {n - 1} calls.")
                text = "".join(b.text for b in resp.content if b.type == "text")
                rec = {
                    "model": model, "abt_id": str(it["abt_id"]), "buy_id": str(it["buy_id"]),
                    "answer_text": text.strip()[:50], "same": parse_answer(text),
                    "stop_reason": resp.stop_reason, "served_by": resp.model,
                    "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                cache[(model, rec["abt_id"], rec["buy_id"])] = rec
                if n % 25 == 0:
                    print(f"  {n}/{len(todo)} calls done")
        used_in = sum(cache[(model, str(i["abt_id"]), str(i["buy_id"]))]["input_tokens"] for i in todo)
        used_out = sum(cache[(model, str(i["abt_id"]), str(i["buy_id"]))]["output_tokens"] for i in todo)
        print(f"Actual usage for new calls: {used_in} input, {used_out} output tokens.")
    return {(str(it["abt_id"]), str(it["buy_id"])): cache[(model, str(it["abt_id"]), str(it["buy_id"]))]["same"]
            for it in items}


if __name__ == "__main__":
    sys.exit("Run python -m eval.run_product_eval --use-llm instead.")
