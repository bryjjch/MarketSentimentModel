"""
Weak supervision: label unlabeled financial text with a larger LLM, then use labels to train BERT.

Expects JSONL input with a text field. Writes JSONL with "text" and "label".

Requires an API key for the chosen provider unless using --provider echo (offline stub):
- OpenAI: OPENAI_API_KEY
- Google AI Studio (Gemini): GOOGLE_API_KEY or GEMINI_API_KEY

Example:
  python -m training.pseudo_label --input data/raw_news.jsonl --output data/pseudo.jsonl --model gpt-4o-mini
  python -m training.pseudo_label --provider google --model gemini-2.0-flash --input data/raw_news.jsonl --output data/pseudo.jsonl
  finsense-pseudo-label --input data/raw_news.jsonl --output data/pseudo.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from .common import SENTIMENT_ID_TO_STR, SENTIMENT_STR_TO_ID


SYSTEM_PROMPT = """You label financial-news sentences for sentiment toward the underlying company or market event.
Reply with exactly one token: negative, neutral, or positive.
Definitions:
- negative: clear bad news, downside, losses, lawsuits, downgrades, bearish tone.
- positive: clear good news, upside, beats, upgrades, bullish tone.
- neutral: factual reporting without clear positive or negative tilt, or mixed/unclear."""

USER_TEMPLATE = """Classify sentiment (one word: negative, neutral, or positive):\n\n\"\"\"{text}\"\"\""""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pseudo-label financial text with an LLM")
    p.add_argument("--input", required=True, type=Path, help="JSONL with a text column")
    p.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    p.add_argument("--text_key", default="text", help="Field name for article/headline body")
    p.add_argument(
        "--provider",
        choices=("openai", "google", "echo"),
        default="openai",
        help="LLM backend: openai (Chat Completions), google (Gemini / Google AI Studio), or echo (offline)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model id for the provider (default: gpt-4o-mini for openai, gemini-2.0-flash for google)",
    )
    p.add_argument("--max_chars", type=int, default=4000, help="Truncate each example for the LLM")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep_s", type=float, default=0.05, help="Pause between API calls")
    p.add_argument("--limit", type=int, default=0, help="If >0, only label the first N rows")
    p.add_argument("--resume", action="store_true", help="Skip rows already present in output (by text)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_jsonl_texts(path: Path, text_key: str, limit: int) -> list[str]:
    """Read JSONL file and return list of texts"""
    texts: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get(text_key)
            if t is None:
                continue
            t = str(t).strip()
            if t:
                texts.append(t)
            if limit and len(texts) >= limit:
                break
    return texts


def normalize_label(raw: str) -> int | None:
    """Map model text to 0/1/2: prefer whole-word sentiment tokens, else stripped alphanumeric lookup."""
    low = raw.strip().lower()
    for word, lid in (("negative", 0), ("neutral", 1), ("positive", 2)):
        if re.search(rf"\b{word}\b", low):
            return lid
    token = re.sub(r"[^a-z]+", "", low)
    return SENTIMENT_STR_TO_ID.get(token)


def google_api_key() -> str:
    """Get Google API key from environment variables"""
    return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()


def label_google(text: str, model: str, temperature: float) -> int:
    """Label text with Google AI Studio"""
    import google.generativeai as genai

    key = google_api_key()
    if not key:
        raise EnvironmentError("Set GOOGLE_API_KEY or GEMINI_API_KEY (Google AI Studio)")

    genai.configure(api_key=key)
    user = USER_TEMPLATE.format(text=text)
    m = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
    resp = m.generate_content(
        user,
        generation_config={"temperature": temperature},
    )
    if not resp.candidates:
        raise ValueError("Empty response (no candidates; possibly blocked)")
    parts = resp.candidates[0].content.parts
    raw = "".join(getattr(p, "text", "") or "" for p in parts).strip()
    lid = normalize_label(raw)
    if lid is None:
        raise ValueError(f"Unparseable model output: {raw!r}")
    return lid


def label_openai(text: str, model: str, temperature: float) -> int:
    """Label text with OpenAI"""
    from openai import OpenAI

    client = OpenAI()
    user = USER_TEMPLATE.format(text=text)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    lid = normalize_label(content)
    if lid is None:
        raise ValueError(f"Unparseable model output: {content!r}")
    return lid


def label_echo(text: str, seed: int) -> int:
    """Label text with echo (offline)"""
    rng = random.Random((hash(text) & 0xFFFFFFFF) ^ seed)
    return rng.randint(0, 2)


def load_done_texts(path: Path) -> set[str]:
    """Load done texts from a JSONL file"""
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text")
            if isinstance(t, str):
                done.add(t)
    return done


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if args.model is None:
        args.model = "gemini-2.0-flash" if args.provider == "google" else "gpt-4o-mini"

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")
    if args.provider == "google" and not google_api_key():
        raise EnvironmentError("GOOGLE_API_KEY or GEMINI_API_KEY is not set (Google AI Studio)")

    texts = read_jsonl_texts(args.input, args.text_key, args.limit)
    if not texts:
        raise ValueError("No texts found in input")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    done = load_done_texts(args.output) if args.resume else set()

    n_ok = 0
    n_skip = 0
    n_fail = 0

    with args.output.open(mode, encoding="utf-8") as out:
        for i, raw_text in enumerate(texts):
            if raw_text in done:
                n_skip += 1
                continue
            snippet = raw_text[: args.max_chars]
            try:
                if args.provider == "openai":
                    lid = label_openai(snippet, args.model, args.temperature)
                elif args.provider == "google":
                    lid = label_google(snippet, args.model, args.temperature)
                else:
                    lid = label_echo(snippet, args.seed)
                rec = {"text": raw_text, "label": int(lid), "label_name": SENTIMENT_ID_TO_STR[lid]}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)
            except Exception as e:
                n_fail += 1
                print(f"[warn] row {i} failed: {e}")

    print(f"Labeled {n_ok} rows, skipped {n_skip}, failed {n_fail}. Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
