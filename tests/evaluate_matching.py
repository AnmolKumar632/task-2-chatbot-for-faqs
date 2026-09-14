"""Lightweight matching-quality evaluator for the FAQ chatbot.

Usage (from the project root):

    python tests/evaluate_matching.py [threshold]

Reads tests/data/matching_evaluation.json, runs each query through the real
FAQChatbot service, and reports classification metrics. A query whose
expected_faq_id is null must be rejected; all other queries must be matched
to exactly that FAQ id.

Metrics definitions:

    Correct             expected answer -> exactly the expected FAQ, OR
                        expected rejection -> rejected
    Wrong match (FP)    expected rejection -> an FAQ was returned
    Wrong match (mismatch)  expected FAQ -> a different FAQ was returned
    False negative      expected FAQ -> rejected

This is a small, hand-written evaluation set for development insight; it is
not a production-grade benchmark.
"""

import json
import sys
from pathlib import Path

from services.chatbot_service import FAQChatbot, load_faqs

BASE_DIR = Path(__file__).resolve().parents[1]
FAQ_DATA_PATH = BASE_DIR / "data" / "faqs.json"
EVALUATION_DATA_PATH = Path(__file__).resolve().parent / "data" / "matching_evaluation.json"


def load_evaluation_cases():
    with open(EVALUATION_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(threshold):
    faq_data = load_faqs(FAQ_DATA_PATH)
    chatbot = FAQChatbot(faq_data, threshold=threshold)
    faq_ids = {faq["id"] for faq in faq_data}

    cases = load_evaluation_cases()
    results = []
    for case in cases:
        expected = case["expected_faq_id"]
        response = chatbot.get_response(case["query"])
        actual = response["faq"]["id"] if response["matched"] else None

        if expected is None:
            outcome = "correct" if actual is None else "false_positive"
        elif actual == expected:
            outcome = "correct"
        elif actual is None:
            outcome = "false_negative"
        else:
            outcome = "wrong_match"

        results.append(
            {
                "query": case["query"],
                "type": case["type"],
                "expected": expected,
                "actual": actual,
                "score": response["score"],
                "outcome": outcome,
            }
        )

    correct = sum(1 for r in results if r["outcome"] == "correct")
    false_positives = sum(1 for r in results if r["outcome"] == "false_positive")
    wrong_matches = sum(1 for r in results if r["outcome"] == "wrong_match")
    false_negatives = sum(1 for r in results if r["outcome"] == "false_negative")
    total = len(results)

    print(f"Threshold: {threshold}")
    print(f"Total queries: {total}")
    print(f"Correct:       {correct}  ({correct / total * 100:.1f}%)")
    print(f"False pos.:    {false_positives}  (expected rejection but answered)")
    print(f"Wrong match:   {wrong_matches}  (expected FAQ but different FAQ)")
    print(f"False neg.:    {false_negatives}  (expected FAQ but rejected)")
    print(f"Accuracy:      {correct / total * 100:.1f}%")
    print()
    for r in results:
        if r["outcome"] != "correct":
            expected = r["expected"] if r["expected"] is not None else "no-match"
            actual = r["actual"] if r["actual"] is not None else "rejected"
            print(
                f"[{r['outcome']:<12}] {r['type']:<12} score={r['score']:.4f} "
                f"expected={expected} actual={actual} :: {r['query']}"
            )
    return results


def threshold_scan():
    faq_data = load_faqs(FAQ_DATA_PATH)
    cases = load_evaluation_cases()
    print("threshold | correct | fp | mismatch | fn | accuracy")
    for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        chatbot = FAQChatbot(faq_data, threshold=threshold)
        correct = 0
        false_positives = 0
        wrong_matches = 0
        false_negatives = 0
        for case in cases:
            expected = case["expected_faq_id"]
            response = chatbot.get_response(case["query"])
            actual = response["faq"]["id"] if response["matched"] else None
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
        print(
            f"{threshold:>9.2f} | {correct:>7} | {false_positives:>3} | "
            f"{wrong_matches:>8} | {false_negatives:>3} | {correct / total * 100:>6.1f}%"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        threshold_scan()
    else:
        threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
        evaluate(threshold)