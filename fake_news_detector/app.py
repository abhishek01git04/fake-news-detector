"""Flask web app + JSON API for the Fake News Detector.

Run from the repository root:
    python -m fake_news_detector.app --api_config demo_data/api_config.yaml
    python -m fake_news_detector.app --demo          # no API keys needed (fake results)
"""

from __future__ import annotations

import argparse
import logging

from flask import Flask, jsonify, render_template, request

from . import storage
from .article import fetch_article
from .detector import PROVIDER_CHOICES, FakeNewsDetector

app = Flask(__name__)
detector: FakeNewsDetector | None = None
log = logging.getLogger("fake_news_detector")

MAX_INPUT_CHARS = 8000


@app.get("/")
def index():
    return render_template("index.html", demo=detector.demo)


@app.post("/api/check")
def api_check():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    url = (payload.get("url") or "").strip()
    source_url = None
    try:
        if url:
            title, body = fetch_article(url)
            text = f"{title}\n{body}" if title else body
            source_url = url
        if not text:
            return jsonify(error="Please provide news text or a URL."), 400
        if len(text) > MAX_INPUT_CHARS:
            text = text[:MAX_INPUT_CHARS]
        result = detector.detect(text, source_url=source_url).to_dict()
        result["id"] = storage.save(result)
        return jsonify(result)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # network / LLM / API-key errors
        log.exception("Check failed")
        return jsonify(error=f"Fact-check failed: {e}"), 500


@app.get("/api/history")
def api_history():
    return jsonify(storage.recent())


@app.get("/api/history/<int:check_id>")
def api_history_item(check_id: int):
    item = storage.get(check_id)
    return (jsonify(item), 200) if item else (jsonify(error="Not found"), 404)


def main():
    global detector
    parser = argparse.ArgumentParser(description="Fake News Detector web app (powered by Loki)")
    parser.add_argument("--provider", default="openai", choices=PROVIDER_CHOICES,
                        help="gemini / groq / ollama are free")
    parser.add_argument("--model", default=None, help="Override the provider's default model")
    parser.add_argument("--client", default=None, help="gpt | claude | local_openai")
    parser.add_argument("--retriever", default="serper", help="serper | google")
    parser.add_argument("--api_config", default="demo_data/api_config.yaml")
    parser.add_argument("--demo", action="store_true", help="Run without API keys (simulated results)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    api_config = {}
    if not args.demo:
        from factcheck.utils.utils import load_yaml

        try:
            api_config = load_yaml(args.api_config) or {}
        except Exception as e:
            print(f"[warn] Could not load {args.api_config}: {e}. Falling back to environment variables.")

    detector = FakeNewsDetector(
        model=args.model, client=args.client, retriever=args.retriever, api_config=api_config,
        demo=args.demo, provider=args.provider,
    )
    mode = "DEMO MODE" if args.demo else f"provider={detector.provider}, model={detector.model}"
    print(f"\n  Fake News Detector running at http://{args.host}:{args.port}  ({mode})\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
