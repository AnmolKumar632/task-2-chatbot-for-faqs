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

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.text_processor import preprocess_text


class SimilarityEngine:
    """Compares user questions against a FAQ knowledge base."""

    def __init__(self, faq_data):
        if not faq_data:
            raise ValueError("FAQ dataset cannot be empty.")

        self.faq_data = faq_data
        self.processed_questions = self._prepare_faq_documents(faq_data)

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
        return self._faq_matrix.shape[0]

    def _score_processed(self, processed_text):
        """Cosine similarity scores for an already-preprocessed query string.

        Shared by both public scoring paths so a caller that already
        preprocessed the query can avoid a redundant NLTK pass.
        """
        if not processed_text:
            return np.zeros(len(self.faq_data), dtype=float)

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
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        if best_score <= 0.0:
            return {"faq": None, "score": 0.0}

        return {"faq": self.faq_data[best_index], "score": best_score}