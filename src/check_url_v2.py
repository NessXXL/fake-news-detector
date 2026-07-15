"""
check_url_v2: two-signal misinformation check (style + evidence).

Signal 1 (STYLE):    fine-tuned RoBERTa — how the text is written.
Signal 2 (EVIDENCE): web corroboration — do independent outlets report
                     the same STORY (not just the same topic)?

Verdicts are three-tier and evidence-first:
  CORROBORATED    — multiple trusted outlets report the same story
  UNVERIFIED      — no independent corroboration found (could be fake,
                    satire, or just very fresh/niche news)
  LIKELY FAKE     — debunk coverage found, or both signals point to fake

Usage:
    pip install trafilatura ddgs
    python src/check_url_v2.py https://example.com/article
"""

import re
import sys
from urllib.parse import urlparse

import torch
import trafilatura
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

MODEL_DIR = "models/roberta-fakenews/checkpoint-8000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRUSTED = {
    "apnews.com", "reuters.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "npr.org", "aljazeera.com",
    "cnn.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com", "afp.com",
    "dw.com", "france24.com", "euronews.com", "politico.com", "axios.com",
}
SATIRE = {
    "theonion.com", "babylonbee.com", "clickhole.com",
    "waterfordwhispersnews.com", "thebeaverton.com", "newsthump.com",
    "duffelblog.com",
}
DEBUNK_WORDS = ("fact check", "fact-check", "debunk", "false claim",
                "no evidence", "satire", "hoax", "misleading")

# Social platforms, blog hosts and aggregators repost anything (including a
# source's own posts) — they are NOT independent corroboration.
SOCIAL_AGGREGATORS = {
    "facebook.com", "m.facebook.com", "youtube.com", "m.youtube.com",
    "twitter.com", "x.com", "reddit.com", "instagram.com", "tiktok.com",
    "pinterest.com", "threads.net", "threads.com", "bsky.app",
    "mastodon.social", "linkedin.com", "t.me", "telegram.me", "vk.com",
    "ok.ru", "tumblr.com", "medium.com", "quora.com",
    "upstract.com", "flipboard.com", "news.google.com", "ground.news",
    "feedly.com", "paperblog.com", "head-topics.com", "headtopics.com",
}
# Anything whose domain contains these fragments is also treated as social.
SOCIAL_FRAGMENTS = ("facebook.", "youtube.", "threads.", "twitter.",
                    "reddit.", "tiktok.", "instagram.")

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "with", "by", "from", "as", "after",
    "over", "new", "says", "say", "said",
}
STORY_MATCH_THRESHOLD = 0.6  # lower to 0.5 if genuine reprints get rejected


def domain_of(url: str) -> str:
    d = urlparse(url).netloc.lower()
    return d[4:] if d.startswith("www.") else d


def is_social(domain: str) -> bool:
    return (domain in SOCIAL_AGGREGATORS
            or any(f in domain for f in SOCIAL_FRAGMENTS))


def content_words(title: str) -> set:
    words = re.findall(r"[a-z']+", title.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def fetch_article(url: str):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None, None
    text = trafilatura.extract(downloaded, include_comments=False)
    meta = trafilatura.extract_metadata(downloaded)
    title = meta.title if meta and meta.title else None
    return text, title


@torch.no_grad()
def style_score(text: str) -> float:
    """Returns P(fake) from the fine-tuned RoBERTa."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    enc = tokenizer(text, truncation=True, max_length=256,
                    return_tensors="pt").to(DEVICE)
    probs = torch.softmax(model(**enc).logits, dim=1)[0].cpu().numpy()
    return float(probs[1])


def gather_evidence(title: str, source_domain: str):
    """Search the headline; classify hits into independent / trusted /
    debunk / rejected. Returns a details dict."""
    query = re.sub(r"[\"'\u201c\u201d]", "", title)[:120]
    story_words = content_words(title)

    hits = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=25):
            hits.append({"title": r.get("title", ""), "url": r.get("href", "")})

    independent, trusted_hits, debunk_hits, rejected = set(), set(), [], []
    for h in hits:
        d = domain_of(h["url"])
        if not d or d == source_domain:
            continue
        if is_social(d):
            rejected.append(f"[social] {h['title'][:45]}")
            continue
        title_l = h["title"].lower()
        if any(w in title_l for w in DEBUNK_WORDS):
            debunk_hits.append(h["title"][:80])
            continue
        overlap = len(story_words & content_words(h["title"]))
        if not story_words or overlap / len(story_words) < STORY_MATCH_THRESHOLD:
            rejected.append(f"[topic-only] {h['title'][:45]}")
            continue
        independent.add(d)
        if d in TRUSTED:
            trusted_hits.add(d)

    return {
        "query": query,
        "independent": sorted(independent),
        "trusted": sorted(trusted_hits),
        "debunks": debunk_hits[:3],
        "rejected": rejected[:6],
    }


def decide(p_fake_style: float, ev: dict, is_satire_source: bool):
    """
    Transparent, evidence-first decision rules (in priority order):
    1. Known satire source                       -> SATIRE
    2. Debunk coverage found                     -> LIKELY FAKE
    3. >=2 trusted outlets carry the story       -> CORROBORATED (real)
    4. No independent corroboration at all       -> UNVERIFIED
       (style cannot rescue an unverifiable story — we showed empirically
        that professional satire fools the style model)
    5. Weak corroboration (1-3 independent)      -> lean on style:
         style says fake (>=0.5) -> LIKELY FAKE, else WEAKLY CORROBORATED
    """
    if is_satire_source:
        return "SATIRE (known satire outlet)"
    if ev["debunks"]:
        return "LIKELY FAKE (debunk coverage found)"
    if len(ev["trusted"]) >= 2:
        return "CORROBORATED — LIKELY REAL"
    if len(ev["independent"]) == 0:
        return "UNVERIFIED — no independent source reports this story"
    if p_fake_style >= 0.5:
        return "LIKELY FAKE (weak corroboration + fake-leaning style)"
    return "WEAKLY CORROBORATED — treat with caution"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else input("Article URL: ").strip()
    src = domain_of(url)

    print(f"Fetching: {url}")
    text, title = fetch_article(url)
    if not text or len(text) < 100:
        print("Could not extract article text (paywall/JS-heavy site).")
        return
    if not title:
        title = text.split(".")[0][:120]
    print(f"Title: {title}\n")

    p_fake_style = style_score(f"{title}. {text}")
    ev = gather_evidence(title, src)
    verdict = decide(p_fake_style, ev, src in SATIRE)

    print("=" * 56)
    print(f"STYLE model      P(fake) = {p_fake_style:.3f}")
    print(f"EVIDENCE check")
    print(f"  search query:        {ev['query']}")
    print(f"  independent domains: {len(ev['independent'])} {ev['independent'][:10]}")
    print(f"  trusted outlets:     {ev['trusted'] or 'none'}")
    if ev["debunks"]:
        print(f"  debunk-style hits:   {ev['debunks']}")
    if ev["rejected"]:
        print(f"  rejected hits:       {ev['rejected']}")
    print("-" * 56)
    print(f"VERDICT: {verdict}")
    print("=" * 56)
    print("\nHow to read this: STYLE judges how the text is written; "
          "EVIDENCE judges whether independent outlets report the same "
          "story. Verdicts are evidence-first because we showed the style "
          "model is blind to professionally written satire. UNVERIFIED "
          "means no corroboration was found — possibly fake, possibly "
          "just very fresh or niche. Neither signal verifies individual "
          "claims inside the article.")


if __name__ == "__main__":
    main()
