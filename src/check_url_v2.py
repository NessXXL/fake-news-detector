"""
check_url_v2: two-signal misinformation check.

Signal 1 (STYLE):    fine-tuned RoBERTa — how the text is written.
Signal 2 (EVIDENCE): web corroboration — do independent sources report
                     the same STORY (not just the same topic)?

Usage:
    pip install trafilatura ddgs
    python src/check_url_v2.py https://example.com/article

Honest limitations (state these in the report):
- Corroboration checks whether the EVENT is reported elsewhere, not whether
  every claim in the article is true.
- Very fresh news may have few sources yet (false "suspicious").
- Story matching uses word overlap between headlines; heavily rephrased
  headlines of the same story can be rejected (semantic embeddings would
  fix this — future work).
"""

import re
import sys
from urllib.parse import urlparse

import torch
import trafilatura
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from ddgs import DDGS          # new package name
except ImportError:
    from duckduckgo_search import DDGS  # older name, same API

MODEL_DIR = "models/roberta-fakenews"
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

# Social platforms and aggregators repost anything (including the source's
# own posts) — they are NOT independent corroboration.
SOCIAL_AGGREGATORS = {
    "facebook.com", "m.facebook.com", "youtube.com", "m.youtube.com",
    "twitter.com", "x.com", "reddit.com", "instagram.com", "tiktok.com",
    "pinterest.com", "threads.net", "linkedin.com", "t.me", "telegram.me",
    "upstract.com", "flipboard.com", "news.google.com", "ground.news",
    "feedly.com", "paperblog.com", "head-topics.com", "headtopics.com",
}

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "with", "by", "from", "as", "after",
    "over", "new", "says", "say", "said",
}

# Require this share of the article headline's content words to appear in a
# search-result title before counting it as the same story.
# Lower it to 0.5 if genuine reprints are being rejected too often.
STORY_MATCH_THRESHOLD = 0.6


def domain_of(url: str) -> str:
    d = urlparse(url).netloc.lower()
    return d[4:] if d.startswith("www.") else d


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


def corroboration_score(title: str, source_domain: str):
    """
    Search the headline; count independent domains whose result title
    matches the same STORY (word-overlap check), not just the same topic.
    Returns (suspicion in [0,1], details dict). Higher suspicion = less
    corroborated.
    """
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
        if d in SOCIAL_AGGREGATORS or d.endswith(".facebook.com"):
            rejected.append(f"[social/aggregator] {h['title'][:45]}")
            continue
        title_l = h["title"].lower()
        if any(w in title_l for w in DEBUNK_WORDS):
            debunk_hits.append(h)
            continue
        # STORY MATCH: a topic-only hit (e.g. any NASA/Mars page for a fake
        # "lump on Mars" story) shares few content words with the headline
        # and gets rejected here.
        overlap = len(story_words & content_words(h["title"]))
        if not story_words or overlap / len(story_words) < STORY_MATCH_THRESHOLD:
            rejected.append(h["title"][:60])
            continue
        independent.add(d)
        if d in TRUSTED:
            trusted_hits.add(d)

    n_ind, n_tru = len(independent), len(trusted_hits)
    if n_tru >= 2:
        suspicion = 0.0
    elif n_tru == 1 or n_ind >= 4:
        suspicion = 0.25
    elif n_ind >= 2:
        suspicion = 0.5
    elif n_ind == 1:
        suspicion = 0.75
    else:
        suspicion = 1.0
    if debunk_hits:
        suspicion = max(suspicion, 0.75)   # debunks are a red flag on their own

    details = {
        "query": query,
        "independent_domains": sorted(independent)[:10],
        "trusted_domains": sorted(trusted_hits),
        "debunk_results": [h["title"][:80] for h in debunk_hits[:3]],
        "rejected": rejected[:6],
    }
    return suspicion, details


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

    if src in SATIRE:
        print(f"NOTE: {src} is a known satire outlet — content is intentionally "
              "fictional, regardless of any score below.\n")

    p_fake_style = style_score(text)
    suspicion, det = corroboration_score(title, src)

    combined = 0.5 * p_fake_style + 0.5 * suspicion
    verdict = ("LIKELY FAKE / UNVERIFIED" if combined >= 0.5
               else "LIKELY REAL")

    print("=" * 56)
    print(f"STYLE model      P(fake) = {p_fake_style:.3f}")
    print(f"EVIDENCE check   suspicion = {suspicion:.2f}")
    print(f"  search query:        {det['query']}")
    print(f"  independent domains: {len(det['independent_domains'])} "
          f"{det['independent_domains']}")
    print(f"  trusted outlets:     {det['trusted_domains'] or 'none'}")
    if det["debunk_results"]:
        print(f"  debunk-style hits:   {det['debunk_results']}")
    if det.get("rejected"):
        print(f"  rejected hits:       {det['rejected']}")
    print("-" * 56)
    print(f"COMBINED = {combined:.2f}  ->  {verdict}")
    print("=" * 56)
    print("\nHow to read this: STYLE judges how the text is written; "
          "EVIDENCE judges whether independent outlets report the same story. "
          "They can disagree — professionally written satire fools STYLE "
          "but fails EVIDENCE; a clumsy retelling of a true story does the "
          "opposite. Neither signal verifies individual claims.")


if __name__ == "__main__":
    main()
