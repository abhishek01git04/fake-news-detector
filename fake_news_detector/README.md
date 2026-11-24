# Fake News Detector (built on Loki)

A textual fake-news detection system. It does more than classify writing style: it breaks a news text into
individual factual claims, searches the web for evidence, checks each claim against that evidence, and
combines the results into a verdict for the whole article.

## Architecture

```
 Input (text or news URL)
        │
        ▼
 article.py ── extracts the article body from the URL (httpx + BeautifulSoup)
        │
        ▼
 detector.py ── FakeNewsDetector
        │   Loki pipeline (factcheck/):
        │     1. Decompose        → split the text into atomic claims        (LLM)
        │     2. Checkworthy      → keep only verifiable factual claims      (LLM)
        │     3. QueryGenerator   → write search queries for each claim      (LLM)
        │     4. Retriever        → fetch web evidence (Serper / Google)
        │     5. ClaimVerify      → evidence SUPPORTS / REFUTES / IRRELEVANT (LLM)
        │   Verdict layer (this project):
        │     factuality ≥ 0.70 → REAL
        │     0.40 – 0.70       → MISLEADING
        │     < 0.40            → FAKE
        │     no verified claim → UNVERIFIABLE
        ▼
 app.py (Flask UI + JSON API) ── storage.py (SQLite history)
 evaluate.py (accuracy / precision / recall / F1 on a labelled CSV)
```

## Setup

From the repository root (`OpenFactVerification/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

Add your keys to `demo_data/api_config.yaml` (git ignores this file):

```yaml
OPENAI_API_KEY: sk-...       # required
SERPER_API_KEY: ...          # required, from https://serper.dev
```

## Run it for free

| Part | Free option | Get it |
|------|-------------|--------|
| Web search | Serper free plan: 2,500 searches, no card | https://serper.dev → `SERPER_API_KEY` |
| LLM (recommended) | Google Gemini free tier, fast | https://aistudio.google.com/apikey → `GEMINI_API_KEY` |
| LLM (alternative) | Groq free tier, fast | https://console.groq.com/keys → `GROQ_API_KEY` |
| LLM (offline) | Ollama on your PC, no key, no limits, but slow on CPU | `ollama pull qwen2.5:7b` |

```powershell
python -m fake_news_detector.app --provider gemini     # recommended
python -m fake_news_detector.app --provider groq
python -m fake_news_detector.app --provider ollama     # offline LLM (search still uses Serper)
python -m fake_news_detector.evaluate --provider gemini
```

Free tiers limit requests per minute. The client now waits and retries automatically when it hits the limit,
so a check can take a few minutes, and a large evaluation can use up the daily quota. If a provider renames
its model, pass `--model <new-name>`.

## Run

```powershell
# Web app → http://127.0.0.1:5000
python -m fake_news_detector.app --api_config demo_data/api_config.yaml

# Without API keys (simulated results, for trying out the UI)
python -m fake_news_detector.app --demo

# Cheaper model or Claude
python -m fake_news_detector.app --model gpt-4o-mini
python -m fake_news_detector.app --model claude-sonnet-4-5   # needs ANTHROPIC_API_KEY
```

### JSON API

```
POST /api/check          {"text": "..."}  or  {"url": "https://..."}
GET  /api/history        the 20 most recent checks
GET  /api/history/<id>   the full result of one check
```

Example response (shortened):
```json
{
  "verdict": "FAKE", "factuality": 0.0, "confidence": 0.83,
  "explanation": "Most factual claims are contradicted by retrieved evidence.",
  "stats": {"num_claims": 2, "num_verified": 2, "num_supported": 0, "num_refuted": 2},
  "claims": [{"claim": "...", "status": "REFUTED",
              "evidences": [{"url": "...", "reasoning": "...", "relationship": "REFUTES"}]}]
}
```

### Use as a library

```python
from fake_news_detector import FakeNewsDetector
det = FakeNewsDetector(api_config={"OPENAI_API_KEY": "...", "SERPER_API_KEY": "..."})
print(det.detect("Some news text").to_dict())
```

## Evaluation

```powershell
python -m fake_news_detector.evaluate --data fake_news_detector/data/sample_news.csv
python -m fake_news_detector.evaluate --data liar_test.csv --limit 100
```

The CSV needs `text,label` columns. Labels can be `REAL`/`FAKE`, `true`/`false`, `1`/`0`, or LIAR labels
(`mostly-true`, `barely-true`, `pants-fire`). FAKE is the positive class, and MISLEADING predictions count as
FAKE. UNVERIFIABLE predictions are reported as abstentions and shown in the coverage figure.
Results for each sample are written to `evaluation_results.csv`.

Datasets you can use: **LIAR** (short political statements), **FEVER** (claims checked against Wikipedia),
**FakeNewsNet** (full news articles). Each check makes several LLM and search calls, so run a sample with
`--limit` first.

## Project files

| File | Purpose |
|------|---------|
| `detector.py` | Wraps Loki, maps scores to a verdict and confidence, demo mode |
| `article.py` | Extracts article text from a URL |
| `app.py` | Flask web app and REST API |
| `storage.py` | SQLite history (`data/history.db`) |
| `evaluate.py` | Batch evaluation and metrics |
| `templates/`, `static/` | Frontend (HTML/CSS/JS) |
| `data/sample_news.csv` | 10 labelled example statements |

## Limitations
- Results depend on what web search returns. Very recent or local news may come back UNVERIFIABLE.
- Each check makes LLM API calls, which cost money, and takes roughly 30–90 seconds.
- Opinions and satire contain no check-worthy factual claims, so they are not judged.
- Demo mode only looks for keywords. It is for trying the UI, not for real results.
