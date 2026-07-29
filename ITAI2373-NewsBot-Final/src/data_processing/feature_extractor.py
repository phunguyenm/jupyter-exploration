"""
feature_extractor.py — Feature Extraction for NewsBot 2.0

This module converts raw text into numbers (features) that machine
learning models can understand and work with.

Think of it like translating human language into math.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    """
    Extracts numerical features from news article text.

    Three types of features:
    1. TF-IDF features — based on word importance
    2. Basic features — word count, sentence count, etc.
    3. Custom features — readability, punctuation density, etc.

    Example:
        extractor = FeatureExtractor()
        extractor.fit(list_of_articles)
        features = extractor.transform("Apple reported record profits today.")
    """

    def __init__(self, max_features=5000):
        """
        Args:
            max_features: Maximum number of TF-IDF features to keep
        """
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2)  # Use single words AND pairs of words
        )
        self.is_fitted = False

    def fit(self, documents):
        """
        Learn vocabulary and statistics from a list of articles.

        Args:
            documents: List of article text strings
        """
        self.tfidf.fit(documents)
        self.is_fitted = True
        print(f"✅ Feature extractor fitted on {len(documents)} documents.")

    def extract_tfidf_features(self, text):
        """
        Convert text into TF-IDF feature vector.

        TF-IDF gives higher scores to words that are important
        in this article but rare across all articles.

        Args:
            text: A single article string

        Returns:
            A sparse matrix of TF-IDF scores
        """
        if not self.is_fitted:
            return {"error": "Call fit() first."}
        return self.tfidf.transform([text])

    def extract_basic_features(self, text):
        """
        Extract simple counting features from text.

        Returns:
            Dictionary with word count, sentence count, etc.
        """
        if not text:
            return {}

        words = text.split()
        sentences = text.split(".")
        paragraphs = text.split("\n\n")

        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_word_length": round(
                sum(len(w) for w in words) / max(len(words), 1), 2
            ),
            "unique_words": len(set(w.lower() for w in words)),
            "lexical_diversity": round(
                len(set(w.lower() for w in words)) / max(len(words), 1), 3
            )
        }

    def extract_custom_features(self, text):
        """
        Extract custom features useful for news classification.

        Returns:
            Dictionary with punctuation density, capital ratio, etc.
        """
        if not text:
            return {}

        total_chars = len(text)
        if total_chars == 0:
            return {}

        # Count different character types
        capital_letters = sum(1 for c in text if c.isupper())
        punctuation = sum(1 for c in text if c in ".,!?;:\"'")
        digits = sum(1 for c in text if c.isdigit())
        exclamations = text.count("!")
        questions = text.count("?")
        quotes = text.count('"')

        return {
            "capital_ratio": round(capital_letters / total_chars, 4),
            "punctuation_density": round(punctuation / total_chars, 4),
            "digit_ratio": round(digits / total_chars, 4),
            "exclamation_count": exclamations,
            "question_count": questions,
            "quote_count": quotes,
            "has_numbers": digits > 0
        }

    def extract_all_features(self, text):
        """
        Extract all feature types for a single article.

        Args:
            text: A news article string

        Returns:
            Dictionary combining all feature types
        """
        basic = self.extract_basic_features(text)
        custom = self.extract_custom_features(text)

        return {
            "basic_features": basic,
            "custom_features": custom
        }

    def batch_extract(self, texts):
        """
        Extract features for a list of articles.

        Args:
            texts: List of article strings

        Returns:
            List of feature dictionaries
        """
        return [self.extract_all_features(text) for text in texts]
