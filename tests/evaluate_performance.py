"""Lightweight performance evaluation utility for the FAQ chatbot.

Measure-first tooling for Phase 12. Uses only the standard library (no
benchmarking framework) and reports actual timings from the current machine.

Sections:

    1. Startup       FAQ file load time and chatbot/engine init time
                     (engine init includes first-time NLTK resource loading).
    2. Query path    per-query breakdown for representative queries
                     (preprocess, engine scoring, full chatbot response).
    3. Repeated      stats for one query run many times (avg/min/max/median/p95).
    4. Batch smoke   determinism + exception-free run over a mixed query set,
                     plus a Python-side peak-memory sample (tracemalloc).
    5. Top-K         top-3 candidate inspection for short/ambiguous queries
                     (debugging aid; not exposed to users).

Usage (from the project root):

    python tests/evaluate_performance.py

These numbers describe this machine only; nothing here is asserted in the
pytest suite, so the tests never depend on CPU speed.
"""

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from nlp.text_processor import preprocess_tokens
from services.chatbot_service import FAQChatbot, load_faqs

BASE_DIR = Path(__file__).resolve().parents[1]
FAQ_DATA_PATH = BASE_DIR / "data" / "faqs.json"

REPRESENTATIVE_QUERIES = [
    ("exact", "How can I reset my password?"),
    ("paraphrase", "I forgot my password. How can I change it?"),
    ("short", "password"),
    ("unrelated", "What is the weather today?"),
    ("ambiguous", "I need help with payment."),
    ("malicious", "<script>alert('XSS')</script>"),
]

AMBIGUOUS_QUERIES = [
    "I have a problem with my account.",
    "I need help with payment.",
    "My course is not working.",
]

SHORT_QUERIES = ["password", "certificate", "refund", "payment", "course"]

BATCH_QUERIES = [q for _, q in REPRESENTATIVE_QUERIES] * 40  # 240 calls


def _fmt_seconds(seconds):
    if seconds >= 1:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000:.3f} ms"


def main(loop_count=25):
    print("=" * 62)
    print("FAQ Chatbot performance evaluation (this machine only)")
    print("=" * 62)

    # 1. Startup ----------------------------------------------------------
    print("\n[1] Startup")
    t0 = time.perf_counter()
    faqs = load_faqs(FAQ_DATA_PATH)
    t1 = time.perf_counter()
    chatbot = FAQChatbot(faqs, threshold=0.35)
    t2 = time.perf_counter()
    print(f"  FAQ dataset load:        {_fmt_seconds(t1 - t0)}")
    print(f"  Chatbot/engine init:     {_fmt_seconds(t2 - t1)} "
          "(includes first-time NLTK resource loading)")
    print(f"  FAQs in memory:          {len(chatbot.faq_data)}")

    engine = chatbot.similarity_engine

    # 2. Query path --------------------------------------------------------
    print("\n[2] Query path (average over expanded warm loops)")
    print(f"  {'type':<12}{'preprocess':>14}{'engine score':>14}{'get_response':>16}")
    for label, query in REPRESENTATIVE_QUERIES:
        # Warm up once (one-time lazy initialization must not skew timing).
        chatbot.get_response(query)
        pre_times, eng_times, total_times = [], [], []
        for _ in range(loop_count):
            t0 = time.perf_counter()
            preprocess_tokens(query)
            t1 = time.perf_counter()
            engine.score_questions(query)
            t2 = time.perf_counter()
            chatbot.get_response(query)
            t3 = time.perf_counter()
            pre_times.append(t1 - t0)
            eng_times.append(t2 - t1)
            total_times.append(t3 - t2)
        print(
            f"  {label:<12}{_fmt_seconds(statistics.mean(pre_times)):>14}"
            f"{_fmt_seconds(statistics.mean(eng_times)):>14}"
            f"{_fmt_seconds(statistics.mean(total_times)):>16}"
        )
    print("  (the engine score column reflects score_questions(), which")
    print("   preprocesses internally; get_response preprocesses exactly once")
    print("   and reuses those tokens for both the length guard and matching.)")

    # 3. Repeated single query ---------------------------------------------
    print("\n[3] Repeated query: 'How can I reset my password?' x200")
    times = []
    chatbot.get_response("How can I reset my password?")
    for _ in range(200):
        t0 = time.perf_counter()
        chatbot.get_response("How can I reset my password?")
        times.append(time.perf_counter() - t0)
    times.sort()
    p95 = times[int(len(times) * 0.95) - 1]
    print(f"  avg={_fmt_seconds(statistics.mean(times))}  "
          f"min={_fmt_seconds(min(times))}  "
          f"max={_fmt_seconds(max(times))}")
    print(f"  median={_fmt_seconds(statistics.median(times))}  "
          f"p95={_fmt_seconds(p95)}")

    # 4. Batch smoke + memory sample ---------------------------------------
    print("\n[4] Batch smoke: 240 mixed requests (40x the representative set)")
    tracemalloc.start()
    first = {}
    exceptions = 0
    t0 = time.perf_counter()
    for query in BATCH_QUERIES:
        response = chatbot.get_response(query)
        key = query
        if key in first:
            if response != first[key]:
                exceptions += 1
                print(f"  NON-DETERMINISTIC result for {query!r}")
        else:
            first[key] = response
    elapsed = time.perf_counter() - t0
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  total time:               {_fmt_seconds(elapsed)}")
    print(f"  avg per request:          {_fmt_seconds(elapsed / len(BATCH_QUERIES))}")
    print(f"  non-deterministic:        {exceptions}")
    print(f"  Python-side peak memory:  {peak_kb / 1024:.2f} MiB (tracemalloc)")

    # 5. Top-K analysis for short / ambiguous queries ----------------------
    print("\n[5] Top-3 candidates for short/ambiguous queries (debugging only)")
    for query in SHORT_QUERIES + AMBIGUOUS_QUERIES:
        scores = engine.score_questions(query)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        cand = "; ".join(
            f"{engine.faq_data[i]['id']}:{engine.faq_data[i]['question']} "
            f"({scores[i]:.3f})"
            for i in top
        )
        verdict = chatbot.get_response(query)
        print(f"  {query!r}")
        print(f"      top3 -> {cand}")
        print(f"      chatbot -> matched={verdict['matched']} "
              f"faq={verdict['faq']['id'] if verdict['faq'] else None} "
              f"score={verdict['score']}")

    # 6. Matching evaluation reference -------------------------------------
    print("\n[6] Matching evaluation at 0.35 (for comparison)")
    eval_path = Path(__file__).resolve().parent / "data" / "matching_evaluation.json"
    with open(eval_path, "r", encoding="utf-8") as file:
        cases = json.load(file)
    faq_ids = {faq["id"] for faq in faqs}
    correct = false_positives = wrong_matches = false_negatives = 0
    fallbacks = 0
    for case in cases:
        expected = case["expected_faq_id"]
        response = chatbot.get_response(case["query"])
        actual = response["faq"]["id"] if response["matched"] else None
        if not response["matched"]:
            fallbacks += 1
        if expected is None:
            if actual is None:
                correct += 1
            else:
                false_positives += 1
        elif actual == expected:
            correct += 1
        elif actual is None:
            false_negatives += 1
        else:
            wrong_matches += 1
    total = len(cases)
    print(f"  total={total} correct={correct} wrong={wrong_matches} "
          f"fp={false_positives} fn={false_negatives} fallbacks={fallbacks}")
    print(f"  expected ids valid: {all(c['expected_faq_id'] in faq_ids or c['expected_faq_id'] is None for c in cases)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAQ chatbot performance evaluation")
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="iterations per representative query (default 25)",
    )
    args = parser.parse_args()
    main(loop_count=args.iterations)