"""Evaluate the detector on a labelled CSV dataset.

CSV format (header required):  text,label      where label is REAL or FAKE
(also accepts true/false, 1/0 with 1 = REAL; LIAR-style labels are mapped below).

Usage (from repo root):
    python -m fake_news_detector.evaluate --data fake_news_detector/data/sample_news.csv
    python -m fake_news_detector.evaluate --data my.csv --limit 50 --api_config demo_data/api_config.yaml
    python -m fake_news_detector.evaluate --demo            # pipeline smoke test without keys

Metrics treat FAKE as the positive class. MISLEADING predictions count as FAKE.
UNVERIFIABLE predictions are reported as abstentions and excluded from the metrics (coverage is shown).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .detector import PROVIDER_CHOICES, FakeNewsDetector

LABEL_MAP = {
    "real": "REAL", "true": "REAL", "1": "REAL", "mostly-true": "REAL",
    "fake": "FAKE", "false": "FAKE", "0": "FAKE", "pants-fire": "FAKE", "barely-true": "FAKE",
}


def load_dataset(path: Path, limit: int | None) -> list[tuple[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = LABEL_MAP.get(str(r.get("label", "")).strip().lower())
            text = (r.get("text") or "").strip()
            if label and text:
                rows.append((text, label))
    return rows[:limit] if limit else rows


def metrics(pairs: list[tuple[str, str]]) -> dict:
    tp = sum(1 for g, p in pairs if g == "FAKE" and p == "FAKE")
    tn = sum(1 for g, p in pairs if g == "REAL" and p == "REAL")
    fp = sum(1 for g, p in pairs if g == "REAL" and p == "FAKE")
    fn = sum(1 for g, p in pairs if g == "FAKE" and p == "REAL")
    n = len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n, "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate the Fake News Detector on a labelled CSV")
    ap.add_argument("--data", default=str(Path(__file__).parent / "data" / "sample_news.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="evaluation_results.csv")
    ap.add_argument("--provider", default="openai", choices=PROVIDER_CHOICES, help="gemini / groq / ollama are free")
    ap.add_argument("--model", default=None)
    ap.add_argument("--client", default=None)
    ap.add_argument("--retriever", default="serper")
    ap.add_argument("--api_config", default="demo_data/api_config.yaml")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    data = load_dataset(Path(args.data), args.limit)
    if not data:
        sys.exit(f"No usable rows in {args.data} (need columns: text,label)")

    api_config = {}
    if not args.demo:
        from factcheck.utils.utils import load_yaml

        try:
            api_config = load_yaml(args.api_config) or {}
        except Exception as e:
            print(f"[warn] Could not load {args.api_config}: {e}. Using environment variables.")

    det = FakeNewsDetector(model=args.model, client=args.client, retriever=args.retriever,
                           api_config=api_config, demo=args.demo, provider=args.provider)

    records, scored = [], []
    for i, (text, gold) in enumerate(data, 1):
        try:
            r = det.detect(text)
            verdict, fact, err = r.verdict.value, r.factuality, ""
        except Exception as e:
            verdict, fact, err = "ERROR", None, str(e)
        pred = {"REAL": "REAL", "FAKE": "FAKE", "MISLEADING": "FAKE"}.get(verdict)
        if pred:
            scored.append((gold, pred))
        mark = "OK " if pred == gold else ("-- " if pred is None else "XX ")
        print(f"[{i}/{len(data)}] {mark} gold={gold:<4} verdict={verdict:<12} | {text[:70]}")
        records.append({"text": text, "gold": gold, "verdict": verdict, "factuality": fact, "error": err})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    m = metrics(scored)
    abstain = len(data) - m["n"]
    print("\n================ Evaluation ================")
    print(f"Samples: {len(data)}   Scored: {m['n']}   Abstained/errors: {abstain}   Coverage: {m['n']/len(data):.1%}")
    print(f"Accuracy : {m['accuracy']:.3f}")
    print(f"Precision: {m['precision']:.3f}   (positive class = FAKE)")
    print(f"Recall   : {m['recall']:.3f}")
    print(f"F1-score : {m['f1']:.3f}")
    print("\nConfusion matrix (rows = gold, cols = predicted)")
    print("              pred REAL   pred FAKE")
    print(f"  gold REAL   {m['tn']:>9}   {m['fp']:>9}")
    print(f"  gold FAKE   {m['fn']:>9}   {m['tp']:>9}")
    print(f"\nPer-sample results written to {args.out}")


if __name__ == "__main__":
    main()
