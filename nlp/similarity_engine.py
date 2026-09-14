"""TF-IDF vectorization and cosine similarity matching engine.

The engine converts FAQ questions into TF-IDF vectors once at startup, then
reuses that fitted representation for every user question:

    FAQ questions
      -> preprocess_text()
      -> TfidfVectorizer.fit_transform()
      -> stored FAQ matrix

    User question
      -> preprocess_text()
      -> same TfidfVectorizer.transform()
      -> cosine_similarity() against the FAQ matrix
      -> best FAQ + similarity score

The engine does NOT apply confidence thresholds or fallback policies; it only
reports which FAQ is the closest match and how similar it is.
"""

import math
import os
from collections import Counter


_PORTABLE_ENGINE = os.environ.get("FAQ_PORTABLE_NLP", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


if not _PORTABLE_ENGINE:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

from nlp.text_processor import preprocess_text


class _PortableTfidf:
    """Small dependency-free TF-IDF model for serverless cold starts."""

    def __init__(self, documents):
        tokenized = [document.split() for document in documents]
        document_count = len(tokenized)
        document_frequency = Counter(
            token for tokens in tokenized for token in set(tokens)
        )
        self.idf = {
            token: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self.documents = [self._vectorize(tokens) for tokens in tokenized]

    def _vectorize(self, tokens):
        counts = Counter(tokens)
        total = len(tokens)
        vector = {
            token: (count / total) * self.idf[token]
            for token, count in counts.items()
        }
        length = math.sqrt(sum(value * value for value in vector.values()))
        return {token: value / length for token, value in vector.items()}

    def scores(self, document):
        tokens = document.split()
        if not tokens:
            return [0.0] * len(self.documents)
        query = self._vectorize(tokens)
        return [
            float(sum(query.get(token, 0.0) * value for token, value in vector.items()))
            for vector in self.documents
        ]


class SimilarityEngine:
    """Compares user questions against a FAQ knowledge base."""

    def __init__(self, faq_data):
        if not faq_data:
            raise ValueError("FAQ dataset cannot be empty.")

        self.faq_data = faq_data
        self.processed_questions = self._prepare_faq_documents(faq_data)

        if _PORTABLE_ENGINE:
            self._vectorizer = _PortableTfidf(self.processed_questions)
            self._faq_matrix = self._vectorizer.documents
        else:
            self._vectorizer = TfidfVectorizer()
            self._faq_matrix = self._vectorizer.fit_transform(
                self.processed_questions
            )

    def _prepare_faq_documents(self, faq_data):
        """Preprocess every FAQ question and reject any that become empty."""
        processed = []
        for index, faq in enumerate(faq_data, start=1):
            normalized = preprocess_text(faq["question"])
            if not normalized:
                raise ValueError(
                    f"FAQ record {index} becomes empty after preprocessing: "
                    f"{faq['question']!r}"
                )
            processed.append(normalized)
        return processed

    @property
    def document_count(self):
        """Number of FAQ documents in the vectorized matrix."""
        return len(self.faq_data)

    def _score_processed(self, processed_text):
        """Cosine similarity scores for an already-preprocessed query string.

        Shared by both public scoring paths so a caller that already
        preprocessed the query can avoid a redundant NLTK pass.
        """
        if not processed_text:
            return [0.0] * len(self.faq_data) if _PORTABLE_ENGINE else np.zeros(
                len(self.faq_data), dtype=float
            )

        if _PORTABLE_ENGINE:
            return self._vectorizer.scores(processed_text)

        query_vector = self._vectorizer.transform([processed_text])
        if query_vector.nnz == 0:
            return np.zeros(len(self.faq_data), dtype=float)

        scores = cosine_similarity(query_vector, self._faq_matrix)
        return np.ravel(scores)

    def score_questions(self, user_question):
        """Return one cosine-similarity score per FAQ for a user question."""
        processed = preprocess_text(user_question)
        return self._score_processed(processed)

    def find_best_match(self, user_question):
        """Return the highest-scoring FAQ and its similarity score.

        Returns {"faq": None, "score": 0.0} when the question has no
        similarity with any FAQ instead of fabricating a match.
        """
        processed = preprocess_text(user_question)
        return self.find_best_match_processed(processed)

    def find_best_match_processed(self, processed_text):
        """Like find_best_match but expects an already-preprocessed question.

        Callers that already ran the NLP pipeline (for example to enforce the
        single-token rejection rule) pass the normalized text here so the
        query is not preprocessed a second time.
        """
        scores = self._score_processed(processed_text)
        best_index, best_score = max(
            enumerate(scores), key=lambda item: item[1]
        )
        best_score = float(best_score)

        if best_score <= 0.0:
            return {"faq": None, "score": 0.0}

        return {"faq": self.faq_data[best_index], "score": best_score}