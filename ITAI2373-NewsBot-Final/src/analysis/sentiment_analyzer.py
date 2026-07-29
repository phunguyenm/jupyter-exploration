"""
sentiment_analyzer.py — Sentiment Analysis for NewsBot 2.0

This module reads the "mood" of a news article — is it positive,
negative, or neutral? It can also track how sentiment changes over time.

Think of it like a mood detector for news.
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


class SentimentEvolutionTracker:
    """
    Analyzes sentiment in news articles and tracks how it changes over time.

    Uses VADER (Valence Aware Dictionary and sEntiment Reasoner), which is
    specifically designed for analyzing short texts like news headlines.

    Example:
        tracker = SentimentEvolutionTracker()
        result = tracker.analyze_sentiment("Markets hit record highs today!")
        # → {"sentiment": "Positive", "confidence": 0.72, "scores": {...}}
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, article_text):
        """
        Analyze the sentiment of a single article.

        Args:
            article_text: A news article as a string

        Returns:
            Dictionary with:
            - "sentiment": "Positive", "Negative", or "Neutral"
            - "confidence": How strongly positive/negative (0.0 to 1.0)
            - "scores": Raw VADER scores (pos, neg, neu, compound)
        """
        if not article_text or not article_text.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "scores": {"pos": 0, "neg": 0, "neu": 1, "compound": 0}
            }

        scores = self.analyzer.polarity_scores(article_text)

        # VADER's compound score ranges from -1 (most negative) to +1 (most positive)
        if scores["compound"] >= 0.05:
            sentiment = "Positive"
        elif scores["compound"] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "confidence": round(abs(scores["compound"]), 4),
            "scores": scores
        }

    def track_sentiment_over_time(self, articles_with_dates):
        """
        Track sentiment across multiple articles with dates.

        Args:
            articles_with_dates: List of (date_string, article_text) tuples

        Returns:
            A list of sentiment results with dates, ordered as given
        """
        results = []
        for date, article in articles_with_dates:
            sentiment_result = self.analyze_sentiment(article)
            results.append({
                "date": date,
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"],
                "compound_score": sentiment_result["scores"]["compound"]
            })
        return results

    def detect_sentiment_anomalies(self, sentiment_timeline, threshold=0.8):
        """
        Find articles with unusually strong sentiment — these might be
        breaking news or highly emotional events.

        Args:
            sentiment_timeline: Output from track_sentiment_over_time()
            threshold: Confidence score above which we flag as an anomaly

        Returns:
            List of anomalous entries
        """
        anomalies = [
            item for item in sentiment_timeline
            if item["confidence"] > threshold
        ]
        return anomalies

    def batch_analyze(self, articles):
        """
        Analyze sentiment for a list of articles.

        Args:
            articles: List of article text strings

        Returns:
            List of sentiment results in the same order
        """
        return [self.analyze_sentiment(article) for article in articles]

    def get_sentiment_summary(self, articles):
        """
        Get an overall sentiment summary for a collection of articles.

        Returns:
            Dictionary with counts of Positive / Negative / Neutral articles
        """
        results = self.batch_analyze(articles)
        summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for result in results:
            summary[result["sentiment"]] += 1
        summary["total"] = len(articles)
        return summary
