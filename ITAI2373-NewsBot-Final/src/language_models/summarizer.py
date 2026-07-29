"""
summarizer.py — Intelligent Text Summarization for NewsBot 2.0

This module shortens long news articles into brief, readable summaries
by picking the most important sentences.

This is called "extractive summarization" — we extract real sentences
from the article rather than writing new ones from scratch.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)


class IntelligentSummarizer:
    """
    Summarizes news articles by selecting the most important sentences.

    How it works:
    1. Count how often each word appears in the article
    2. Score each sentence based on the words it contains
    3. Pick the top-scoring sentences as the summary

    Example:
        summarizer = IntelligentSummarizer()
        summary = summarizer.summarize_article(long_article_text)
        # → "Tesla reported record profits. CEO Elon Musk praised the team."
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def _score_sentences(self, text):
        """
        Internal helper: score each sentence by word importance.

        Returns a dictionary of {sentence: score}
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())

        # Count word frequencies (ignoring stop words)
        word_freq = {}
        for word in words:
            if word.isalpha() and word not in self.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Score each sentence
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[sentence] = (
                        sentence_scores.get(sentence, 0) + word_freq[word]
                    )

        return sentences, sentence_scores

    def summarize_article(self, article_text, summary_type="balanced"):
        """
        Generate a summary of a news article.

        Args:
            article_text: The full article as a string
            summary_type: "brief" (2 sentences), "balanced" (3), or "detailed" (5)

        Returns:
            A summary string
        """
        if not article_text or not article_text.strip():
            return "No text provided to summarize."

        # Decide how many sentences to include
        num_sentences = {"brief": 2, "detailed": 5}.get(summary_type, 3)

        sentences, sentence_scores = self._score_sentences(article_text)

        if not sentence_scores:
            return article_text[:200] + "..."

        # Pick the top-scoring sentences
        top_sentences = sorted(
            sentence_scores, key=sentence_scores.get, reverse=True
        )[:num_sentences]

        # Keep them in original order for readability
        ordered = [s for s in sentences if s in top_sentences]

        return " ".join(ordered)

    def summarize_multiple_articles(self, articles, focus_topic=None):
        """
        Combine and summarize multiple articles at once.

        Args:
            articles: A list of article text strings
            focus_topic: Optional keyword to bias toward (not yet implemented)

        Returns:
            A combined summary string
        """
        combined_text = " ".join(articles)
        return self.summarize_article(combined_text, summary_type="balanced")

    def generate_headline(self, article_text):
        """
        Generate a one-sentence headline from an article.

        Returns:
            A string with the most important sentence
        """
        summary = self.summarize_article(article_text, summary_type="brief")
        first_sentence = summary.split(".")[0]
        return first_sentence.strip() + "."

    def assess_summary_quality(self, original_text, summary):
        """
        Evaluate how well the summary captures the original article.

        Returns:
            Dictionary with length stats and compression ratio
        """
        original_words = len(original_text.split())
        summary_words = len(summary.split())

        if original_words == 0:
            return {"error": "Original text is empty."}

        return {
            "original_word_count": original_words,
            "summary_word_count": summary_words,
            "compression_ratio": round(summary_words / original_words, 3),
            "reduction_percent": round(
                (1 - summary_words / original_words) * 100, 1
            )
        }
