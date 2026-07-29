"""
test_topic_modeling.py — Tests for Topic Modeling Component
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.topic_modeler import TopicDiscoveryEngine


SAMPLE_ARTICLES = [
    "Apple Inc. reported record quarterly earnings. CEO Tim Cook said iPhone sales exceeded expectations.",
    "Google launched a new artificial intelligence model for natural language processing tasks.",
    "The national soccer team won the championship after a dramatic penalty shootout in the final.",
    "Olympic athletes broke three world records at the swimming competition held in Paris.",
    "The president signed a new trade agreement to boost economic relations with Asia.",
    "Scientists discovered a breakthrough treatment for Alzheimer's disease using gene therapy.",
    "Tesla reported strong electric vehicle sales driven by demand in European markets.",
    "Researchers at MIT developed a new solar panel technology with improved efficiency.",
    "The central bank raised interest rates to combat rising inflation across the economy.",
    "A new climate agreement was signed by world leaders at the environmental summit."
]


def test_topic_model_fitting():
    engine = TopicDiscoveryEngine(n_topics=3)
    engine.fit_topics(SAMPLE_ARTICLES)
    assert engine.is_fitted == True
    print("✅ test_topic_model_fitting PASSED")


def test_get_topic_words():
    engine = TopicDiscoveryEngine(n_topics=3)
    engine.fit_topics(SAMPLE_ARTICLES)
    words = engine.get_topic_words(0, n_words=5)
    assert isinstance(words, list)
    assert len(words) == 5
    print(f"✅ test_get_topic_words PASSED: {words}")


def test_get_all_topics():
    engine = TopicDiscoveryEngine(n_topics=3)
    engine.fit_topics(SAMPLE_ARTICLES)
    topics = engine.get_all_topics(n_words=5)
    assert len(topics) == 3
    print(f"✅ test_get_all_topics PASSED: {len(topics)} topics found")


def test_article_topic_assignment():
    engine = TopicDiscoveryEngine(n_topics=3)
    engine.fit_topics(SAMPLE_ARTICLES)
    result = engine.get_article_topics("Apple announced new iPhone model with AI features.")
    assert "main_topic" in result
    assert "topic_scores" in result
    print(f"✅ test_article_topic_assignment PASSED: main_topic={result['main_topic']}")


def test_unfitted_model():
    engine = TopicDiscoveryEngine(n_topics=3)
    result = engine.get_article_topics("Some article text.")
    assert "error" in result
    print("✅ test_unfitted_model PASSED")


def test_topic_trend_tracking():
    engine = TopicDiscoveryEngine(n_topics=3)
    engine.fit_topics(SAMPLE_ARTICLES)
    articles_with_dates = [
        ("2026-07-01", SAMPLE_ARTICLES[0]),
        ("2026-07-02", SAMPLE_ARTICLES[1]),
        ("2026-07-03", SAMPLE_ARTICLES[2])
    ]
    result = engine.track_topic_trends(articles_with_dates)
    assert result["articles_analyzed"] == 3
    print(f"✅ test_topic_trend_tracking PASSED: {result['articles_analyzed']} articles tracked")


if __name__ == "__main__":
    print("\n=== Running Topic Modeling Tests ===\n")
    test_topic_model_fitting()
    test_get_topic_words()
    test_get_all_topics()
    test_article_topic_assignment()
    test_unfitted_model()
    test_topic_trend_tracking()
    print("\n✅ All topic modeling tests passed!\n")
