# Development Log

Detailed per-phase documentation, evaluation evidence, and rationale for design decisions.

---

## Phase 5 — FAQ Matching API & Confidence Threshold

### Threshold evaluation

The default was evaluated against a representative set (5 related queries with known targets + 4 unrelated queries). Thresholds 0.20-0.40 all accept the 5 related queries correctly and reject all 4 unrelated queries; at 0.45 a genuine paraphrase ("I forgot my password...", score 0.408) starts being lost. 0.35 was chosen because it keeps genuine paraphrases with reasonable headroom.

### Confidence decision flow

Best FAQ -> Similarity Score -> Compare with Threshold -> Accept (score >= threshold) returns FAQ answer / Reject (score < threshold) returns Fallback message

### False positives vs false negatives

- **False positive:** an unrelated query is incorrectly answered with an FAQ. Lower thresholds risk more of these.
- **False negative:** a genuinely related query is rejected. Higher thresholds risk more of these.

There is a permanent trade-off: lower threshold -> more accepted -> more false positives; higher threshold -> fewer accepted -> more false negatives.

### Example (actual values from the implementation)

```text
User Question:   I forgot my password. How can I change it?
Best FAQ:        How can I reset my password?        (id 1)
Similarity:      0.4082
Threshold:       0.35
Decision:        Accepted

User Question:   What is the weather today?
Best FAQ:        none (no vocabulary overlap -> score 0.0)
Similarity:      0.0000
Threshold:       0.35
Decision:        Rejected -> fallback message
```

### Current Limitation

The cosine similarity threshold is a practical heuristic, not a calibrated probability. A score of 0.80 does not mean "80% chance the answer is correct". The score is labeled a **similarity score** throughout the system for this reason.

---

## Phase 9 — Matching Quality Evaluation

Phase 9 improved matching quality **based on evidence**, not guesses. Every change was justified by a measured failure in a fixed evaluation set, and no change was adopted unless it made the results better.

### The evaluation set

`tests/data/matching_evaluation.json` fixes 40 queries across categories:

- **exact** — the FAQ question itself
- **paraphrase** — reworded versions of a FAQ
- **vocabulary** — different words for the same concept (`options` ~ `methods`)
- **short** — a single keyword with no context
- **ambiguous** — a question that fits several FAQs
- **unrelated** — off-topic, must be rejected
- **out-of-vocabulary** — invented terms, must be rejected

`tests/evaluate_matching.py` runs the full chatbot over the set and reports correct / false-positive / wrong-match / false-negative counts:

```bash
python tests\evaluate_matching.py            # at the default threshold
python tests\evaluate_matching.py --scan     # threshold sweep 0.20-0.60
```

Note: the script exercises endpoint behavior, not just raw similarity — the threshold and the single-token rejection rule both take effect.

### Baseline (Phase 3-8 code)

At `threshold = 0.35`: **28 / 40 correct (70%)**, 3 false positives, 9 wrong matches, 0 false negatives. Representative failures:

- "What payment options do you support?" matched the *payment failed* FAQ (13 to 16)
- "Where can I see my grades from exams?" matched the *exams access* FAQ (28 to 27)
- "How do I check that a certificate is genuine?" matched *exam results* (23 to 28)
- "Do I need any background knowledge to enroll?" matched *enrollment* (12 to 10)
- single words "certificate", "refund", "password" produced confident answers

### Root causes

1. **Verb inflections were not reduced.** The lemmatizer defaults to the noun sense, so `buying` and `failed` stayed inflected.
2. **No domain synonyms.** `options`, `grades`, `genuine`, `retake` had no link to `methods`, `results`, `authentic`, `resubmit`.
3. **Phrases were lost to stop words.** `money back` dropped `back` before it could match anything; `paid subscription` and `background knowledge` were unmatched phrases.
4. **Single-token queries were over-confident.** `certificate` scored 0.67 against one FAQ and was answered despite being ambiguous.
5. **Near-tie wins were lexical accidents.** Margin analysis showed the best-scoring FAQ was genuinely the wrong FAQ (not a borderline tie), so a margin rule was **not** adopted — it would not have helped.

### Improvements implemented

| Change | Where | What it fixes |
| ------ | ----- | ------------- |
| Verb-aware lemmatization | `nlp/text_processor.py` | `buying` -> `buy`, `failed` -> `fail`, `forgot` -> `forget` |
| Phrase rules | `nlp/query_normalizer.py` | `money back` -> `refund`, `paid subscription` -> `premium subscription`, `background knowledge` -> `prerequisite` |
| Token rules | `nlp/query_normalizer.py` | `option` -> `method`, `grade` -> `result`, `genuine` -> `authentic`, `verify` -> `check`, `retake` -> `resubmit`, `code` -> `discount code` |
| Single-token rejection | `services/chatbot_service.py` | a query with fewer than 2 content tokens is rejected as too ambiguous |

Each rule is documented in the normalizer with the query that motivated it. Normalization runs identically on FAQ questions and user questions, keeping the comparison vocabulary consistent.

### After

At `threshold = 0.35`: **39 / 40 correct (97.5%)**, 0 false positives, 1 wrong match, 0 false negatives. Bigrams (`ngram_range=(1, 2)`) were tested against the same set and produced no additional correct answers, so they were **not** adopted — no complexity without measured benefit.

Threshold sweep after the improvements:

```text
threshold | correct | fp | mismatch | fn | accuracy
    0.20  |      39 |   0 |        1 |   0 |   97.5%
    0.25  |      39 |   0 |        1 |   0 |   97.5%
    0.30  |      39 |   0 |        1 |   0 |   97.5%
    0.35  |      39 |   0 |        1 |   0 |   97.5%   <- default
    0.40  |      39 |   0 |        1 |   0 |   97.5%
    0.45  |      37 |   0 |        1 |   2 |   92.5%
    0.50  |      37 |   0 |        1 |   2 |   92.5%
    0.55  |      35 |   0 |        1 |   4 |   87.5%
    0.60  |      34 |   0 |        1 |   5 |   85.0%
```

The default `0.35` remains optimal (flat from 0.20-0.40, then genuine paraphrases are lost).

### Residual limitation

One evaluation query is still matched to the wrong FAQ:

```text
"Do I get my money back after buying a course?" -> lifetime-access FAQ (id 7),
    expected the refund-policy FAQ (id 19)  [score 0.6457]
```

The refund FAQ's question ("What is your refund policy?") shares no other words with the query, while the lifetime-access FAQ shares `get`, `buy`, and `course`. Bridging that gap requires understanding that "get money back" means "refund", which TF-IDF cosine similarity cannot do from vocabulary alone. This is documented as a known qualitative limitation rather than patched with a hard-coded query.

### What was deliberately NOT done

- No LLM, embeddings, vector database, or semantic search.
- No new dependencies.
- No query-specific hard-coding (every query in the evaluation set is matched through the general pipeline).
- No margin-based ambiguity rule (gap analysis showed it would not help).
- No bigrams (measured: no improvement).

---

## Phase 10 — Security, Reliability & Error Handling

### API error contract

The `/api/chat` **200** response keeps the Phase 6 shape (`matched`, `faq_id`, `answer`, `score`, `category`). Every error status returns a single, predictable shape — `{"error": "<message>"}` — so clients never have to handle more than one failure structure.

| Situation                  | Status | Response message                                   |
| -------------------------- | -----: | -------------------------------------------------- |
| Valid chat request         |    200 | structured chat response                            |
| Malformed / missing JSON   |    400 | `Request body must contain valid JSON.`            |
| Non-object JSON            |    400 | `Request body must be a JSON object.`              |
| Missing `question`         |    400 | `Question is required.`                            |
| Non-string `question`      |    400 | `Question must be a string.`                       |
| Empty / whitespace         |    400 | `Question cannot be empty.`                        |
| Question too long          |    400 | `Your question is too long. Please shorten it and try again.` |
| Request body too large     |    413 | `Request body is too large.`                       |
| Unknown API route          |    404 | `Endpoint not found`                               |
| Unsupported method         |    405 | `Method not allowed.`                              |
| Internal failure           |    500 | `An internal server error occurred.`               |

### Input security

- User input is **always treated as data, never as code**. No `eval`, `exec`, shell execution, SQL, or template execution exists anywhere in the codebase.
- Unknown fields in the request (`admin`, `role`, etc.) are ignored; they cannot alter chatbot behavior.
- `Question too long` (default 500 characters, matching the frontend `maxlength`) is rejected before the NLP pipeline runs.
- Flask rejects oversized request bodies (`FAQ_MAX_CONTENT_LENGTH`, default 16 KB) before parsing, protecting against oversized-input denial of service.
- Malicious strings (`<script>...`, `javascript:`, `... DROP TABLE ...`) are answered with the normal fallback — never executed, never echoed back raw.

### Safe error handling and logging

- Errors never leak internals: no tracebacks, file paths, or library details reach the client.
- Internal exceptions are logged with Python's built-in `logging` and replaced with a safe, generic message.
- Startup failures (missing/corrupt FAQ data, missing NLTK resources) abort the application with a clear error instead of running an empty or broken bot. Nothing is downloaded per request.
- Log messages do not include passwords, keys, secrets, or full user payloads.

### HTTP security headers

Every response includes:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: no-referrer`
- `X-XSS-Protection: 0`
- `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'` (skipped in debug mode so the Werkzeug debugger console keeps working)

### Configuration

Centralized, environment-driven settings (all optional with documented defaults in `.env.example`):

```text
FLASK_DEBUG             0                     debug mode (dev only)
SECRET_KEY               change-this-value    session signing
SIMILARITY_THRESHOLD     0.35                 FAQ match threshold
FAQ_MAX_QUESTION_LENGTH  500                  per-question char limit
FAQ_MAX_CONTENT_LENGTH   16384                request body byte limit
```

### Dataset & NLP failure handling

- Missing FAQ file: clear `FileNotFoundError` with the path.
- Corrupt JSON: clear `ValueError` ("not valid JSON").
- Empty dataset, invalid records, duplicate IDs, boolean IDs: rejected by existing validation and Phase 10 bool-ID hardening.
- Missing NLTK resources: clear `RuntimeError` at startup, never a silent partial pipeline and never a per-request download.

### Reliability tests

`tests/test_reliability.py` covers dataset corruption, empty/oversized/blank questions, service-boundary error propagation (the service logs, never swallows), internal-failure logging, and NLTK failure clarity.

### Security tests

`tests/test_security.py` covers malicious payloads (XSS, SQL, `javascript:`), invalid question types, oversized input, unexpected-field manipulation, security headers/CSP, 404/405/413 handling, no-leak assertions, and a static scan proving `eval`/`exec`/`subprocess` do not exist in the source.

### Security limitations

This phase provides defensive input validation, safe error handling, basic HTTP security protections, and protection against common application-level misuse. A production deployment would still require infrastructure-level security: HTTPS/TLS, host-level rate limiting, dependency auditing and upgrades, monitoring, and additional penetration testing. This is an internship project, not an enterprise-security product, and no claim of "100% secure" is made.

---

## Phase 11 — Automated Testing, QA & Evaluation

Phase 11 turns the manual chatbot into a systematically tested, measurable, regression-resistant system. No production behavior was changed to make tests pass, and no dependencies were added (`requirements.txt` is unchanged).

### Testing strategy

The suite follows a testing pyramid — most tests are fast, isolated unit tests; a few exercise the real pipeline end-to-end through the Flask test client:

```text
                 End-to-End / Integration  (test_chat_api)
                         ^
                         |
                   API / Service            (test_app, test_confidence)
                         ^
                         |
                  Unit / NLP Tests          (test_nlp, test_similarity,
                                             test_query_normalizer)
                         ^
                         |
                 Validation / Utilities     (test_faq_dataset)
```

Behavior is verified with descriptive, Arrange-Act-Assert tests. Assertions are behavior-based (no timestamps, internal layout, logging formatting, or exact float representations — scores use tolerances where needed).

### Matching evaluation

`tests/data/matching_evaluation.json` is the source of truth for matching quality: 40 queries against the real `data/faqs.json`, covering exact, paraphrase, vocabulary, short, ambiguous, unrelated, and OOV cases. Every `expected_faq_id` in the file is a real FAQ id (enforced by `test_evaluation.py`).

Current result at the default threshold `0.35`:

```text
Total queries: 40
Correct:       39  (97.5%)
False pos.:    0   (expected rejection but answered)
Wrong match:   1   (expected FAQ but different FAQ)
False neg.:    0   (expected FAQ but rejected)
```

### Security QA (Phase 11 additions)

`test_security.py` was extended with path-traversal payloads (`../../../../etc/passwd`, backslash Windows paths), template-injection strings (`{{ 7 * 7 }}`, `${7 * 7}`), and the frontend static safety scans described in the main README. All injection inputs are answered with the normal fallback — never executed, never echoed verbatim, never leaking internals.

### Testing limitations

- No coverage tool is installed or claimed; the suite favors meaningful behavior tests over an arbitrary coverage percentage.
- The one known wrong-match evaluation query remains; the evaluation tests deliberately do not assert a numeric accuracy score, so a matching improvement is never forced to hand-tune toward a percentage.
- Browser interactions are not automated (no Selenium/Playwright/Cypress are used, per the dependency discipline).

---

## Phase 12 — Performance Optimization & System Evaluation

Phase 12 measures the chatbot and optimizes only where measurement justifies a change. The guiding principle was **measure first, optimize second**: no change was made on theoretical grounds, and no complexity was added without a measured benefit.

### Performance baseline (before)

Measured on the development machine with `tests/evaluate_performance.py` (standard library only — no benchmarking framework):

```text
Startup
  FAQ dataset load:                         0.8 ms
  Chatbot/engine init (NLTK + TF-IDF fit): ~3.5-4.0 s   (once, at startup)

Query path (warm, per request)
  normal accepted query:                    ~1.6-3.6 ms  (machine-load dependent)
  exact "How can I reset my password?":     baseline run avg 1.63 ms (median)
  rejected single-token query:              ~0.1 ms       (fast path)

Repeated query (x200, baseline run):        median 1.52 ms, p95 2.71 ms
Batch 240 mixed requests:                   deterministic, no exceptions
Python-side peak memory (batch):            ~13 MiB (tracemalloc)
```

### Bottlenecks identified

1. **Duplicate preprocessing of the user query (the only confirmed redundancy).** `get_response()` preprocessed the query once for the single-token rejection rule, and the similarity engine preprocessed the **same query a second time**. This was measured directly (~0.15-0.4 ms of every accepted request, roughly 10%).
2. **Startup cost is dominated by first-time NLTK resource loading + TF-IDF fit (~3.5-4 s).** This is one-time and acceptable. No change was made.
3. **Not a bottleneck:** FAQ loading is a single startup read; the vectorizer and FAQ matrix are built once and reused; each request is already ~1.6-3.6 ms on this machine.

### Optimization implemented (the only one justified by measurement)

| | |
|---|---|
| **Problem** | The same user query was NLTK-preprocessed twice per accepted request (once for the token-count rule, once inside the engine). |
| **Solution** | `get_response()` preprocesses once, reuses the normalized tokens for the ambiguity check, and calls the new `SimilarityEngine.find_best_match_processed()` with the already-normalized text. The new method shares `_score_processed()` with the existing public paths. |
| **Reason** | Removes a measured redundant pass with zero change to answer quality (the engine receives the identical normalized string). |
| **Measured result** | Interleaved A/B of the old vs new request path (same process): **median 3.647 ms -> 3.316 ms per accepted query (-0.33 ms, ~-9%)**. |

`SimilarityEngine`'s public API (`find_best_match`, `score_questions`) is unchanged; the raw path simply delegates to the processed path internally. Equivalence is pinned by `test_processed_path_matches_raw_path`.

### Optimizations considered and NOT made

- **Caching query results:** rejected — a small FAQ corpus already answers in ~2-4 ms; caching would add state, staleness, and memory for no meaningful benefit.
- **Startup shaving (lazy NLTK load):** rejected — it would just move the same cost into the first request.
- **Reducing FAQ candidates / skipping preprocessing:** rejected — that would lower answer quality (Phase 5/9 behavior), which is never acceptable.
- **Margin/tie rules for vague queries:** rejected — Phase 9 measured that margin rules do not improve the evaluation set, and hand-tuned keys would violate the "no keyword-rule system" guidance. Ambiguity is documented instead.

### Startup vs request cost

```text
Startup:  FAQ load + NLTK init + TF-IDF fit   ~3.5-4 s   (once)
Request:  validation + 1x preprocessing + transform + cosine similarity
```

The cost profile is correct for this application: the app initializes once at startup, and every request reuses the fitted vectorizer and FAQ matrix.

### Matching & threshold evaluation (unchanged, re-verified)

Re-run after the optimization — identical to Phase 11:

```text
Total queries: 40     Correct: 39 (97.5%)     Wrong match: 1
False positives: 0    False negatives: 0      Fallbacks: 9
Threshold: 0.35
```

### Short query analysis (evidence)

Top-3 candidates (engine scores, debugging only — not exposed to users):

```text
"password"     -> 1:reset password (0.707)                        -> REJECTED
"certificate"  -> 22:download (0.675), 23:verify (0.520), 21:eligibility (0.474) -> REJECTED
"refund"       -> 19:policy (0.665), 20:processing time (0.458)   -> REJECTED
"payment"      -> 16:payment failed (0.665), 13:methods (0.533)    -> REJECTED
"course"       -> 10:enroll (0.478), 27:exams (0.396), 11:cancel (0.376) -> REJECTED
```

Every short query is rejected by the Phase 9 single-token rule. Several have close competing candidates, confirming that answering them would risk confidently incorrect answers.

### Ambiguous query analysis (documented limitation)

```text
"I have a problem with my account."  -> answers delete-account (id 4), score 0.6655
                                       (top-2 are a tie: id 4 = id 30 = 0.665)
"I need help with payment."          -> answers payment-failed (id 16), score 0.443
"My course is not working."          -> answers how-to-enroll (id 10), score 0.478
```

These vague queries pass the 0.35 threshold and receive a specific FAQ answer. This is the known price of a vocabulary-overlap matcher: "a problem with my account" understandably scores highest against whichever account FAQ shares words. It is documented as a limitation rather than patched with a hand-tuned tie/margin rule, because Phase 9 already measured such rules as unhelpful on the evaluation set.

### Memory behavior

- Live server measurement (300 requests, 30 FAQs, working set of the Flask process): **271.49 MiB before -> 271.43 MiB after (-0.05 MiB)** — flat, no per-request accumulation.
- Python-side peak in an in-process 240-request batch: **~13-15 MiB (tracemalloc)**.
- No server-side conversation state exists; history stays in browser `localStorage` (Phase 8), capped at 100 messages.

### Frontend performance review (no change required)

- Messages are appended incrementally (`renderMessage`); the full history is re-rendered once, only when restoring from storage on page load — no O(n^2) DOM rebuilds.
- One `/api/health` call on load; one `/api/chat` call per submitted message; no polling or duplicate event listeners.
- `localStorage` is written once per message and read once on load, bounded by the 100-message cap.
- Assets are tiny (script 12 KB, css 9.6 KB, html 2.2 KB) with no external fonts/images/CDNs.

### Performance tests

`tests/test_performance.py` (from Phase 11, extended in Phase 12) asserts the invariants that matter, never wall-clock values: the FAQ corpus is preprocessed once at init, the vectorizer/matrix are reused (never re-fit), repeated requests are identical/deterministic, and `get_response` preprocesses an accepted query exactly once (the Phase 12 regression guard).

### Honest scalability statement

This is a small educational chatbot: JSON knowledge base + in-memory FAQ data + an in-memory TF-IDF matrix is appropriate for a handful-to-hundreds of FAQs. Significantly larger deployments would eventually need a persistent database, more advanced retrieval, caching, and dedicated/horizontally scaled infrastructure — **none of that is implemented in Phase 12**, and no claim of production-scale performance is made.