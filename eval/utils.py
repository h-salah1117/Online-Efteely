import os, re, time, tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS = {
    "islamweb": r"islamweb\.net/.*?/fatwa/(\d+)",
    "binbaz":   r"binbaz\.org\.sa/fatwas/(\d+)",
    "islamqa":  r"islamqa\.info/.*?/answers/(\d+)",
}


def link_to_id(link):
    """Turn a fatwa URL into a stable id like 'islamweb_12582' (None if unknown site)."""
    for site, pattern in PATTERNS.items():
        m = re.search(pattern, link or "")
        if m:
            return f"{site}_{m.group(1)}"
    return None


def get_groq_key():
    """GROQ_API_KEY from the environment, else from .streamlit/secrets.toml."""
    if os.getenv("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    with open(os.path.join(ROOT, ".streamlit", "secrets.toml"), "rb") as f:
        return tomllib.load(f)["GROQ_API_KEY"]


def get_secret(name):
    """A key from the environment, else from .streamlit/secrets.toml (gitignored)."""
    if os.getenv(name):
        return os.environ[name]
    with open(os.path.join(ROOT, ".streamlit", "secrets.toml"), "rb") as f:
        return tomllib.load(f).get(name)


GEMINI_FALLBACKS = ["gemini-3.5-flash", "gemini-2.5-flash"]


def gemini_json(prompt, temperature=0):
    """Fallback when Groq is rate-limited: try each Gemini model, return (parsed JSON, model)."""
    import json, requests
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    last = None
    for model in GEMINI_FALLBACKS:
        for attempt in range(3):
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key}, timeout=120,
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": temperature,
                                           "responseMimeType": "application/json"}})
            if r.ok:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text), model
            last = f"{model}: {r.status_code}"
            if r.status_code not in (429, 500, 503):
                break
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini fallback failed ({last})")


class DailyLimitError(RuntimeError):
    """Groq's tokens/requests-per-day limit: waiting a minute won't help."""


def with_retry(fn, tries=8):
    """Groq free tier returns 429 a lot; wait and retry instead of crashing the run.
    Per-day limits are raised immediately as DailyLimitError so callers can stop cleanly."""
    from groq import RateLimitError
    for attempt in range(tries):
        try:
            return fn()
        except RateLimitError as e:
            if "per day" in str(e).lower():
                raise DailyLimitError(str(e)[:200]) from e
            wait = 10 * (attempt + 1)
            print(f"   429 rate limit, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("Too many rate-limit retries")
