"""
test_classification.py — Tests for Classification and Sentiment Components
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.classifier import AdvancedNewsClassifier
from src.analysis.sentiment_analyzer import SentimentEvolutionTracker
from src.analysis.ner_extractor import EntityRelationshipMapper


TRAIN_TEXTS = [
    "Apple reported record iPhone sales this quarter driven by strong demand.",
    "Google launched a new AI model that outperforms previous versions.",
    "The football team won the championship after a dramatic final.",
    "The swimmer broke the world record at the Olympic Games.",
    "The president signed a new trade agreement with foreign nations.",
    "Congress passed a bill to reform the healthcare system.",
    "Scientists discovered a new treatment for Alzheimer's disease.",
    "Researchers developed a vaccine that shows promising results."
]

TRAIN_LABELS = [
    "Technology", "Technology",
    "Sports", "Sports",
    "Politics", "Politics",
    "Health", "Health"
]


def test_classifier_training():
    classifier = AdvancedNewsClassifier()
    classifier.train(TRAIN_TEXTS, TRAIN_LABELS)
    assert classifier.is_trained == True
    print("✅ test_classifier_training PASSED")


def test_classifier_prediction():
    classifier = AdvancedNewsClassifier()
    classifier.train(TRAIN_TEXTS, TRAIN_LABELS)
    result = classifier.predict_with_confidence("Tesla announced a new electric vehicle model.")
    assert "category" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    print(f"✅ test_classifier_prediction PASSED: {result['category']} ({result['confidence']})")


def test_untrained_classifier():
    classifier = AdvancedNewsClassifier()
    result = classifier.predict_with_confidence("Some article text.")
    assert "error" in result
    print("✅ test_untrained_classifier PASSED")


def test_positive_sentiment():
    tracker = SentimentEvolutionTracker()
    result = tracker.analyze_sentiment("This is wonderful, amazing, and fantastic news!")
    assert result["sentiment"] == "Positive"
    print(f"✅ test_positive_sentiment PASSED: {result['sentiment']}")


def test_negative_sentiment():
    tracker = SentimentEvolutionTracker()
    result = tracker.analyze_sentiment("This is terrible, horrible, and catastrophic.")
    assert result["sentiment"] == "Negative"
    print(f"✅ test_negative_sentiment PASSED: {result['sentiment']}")


def test_empty_sentiment():
    tracker = SentimentEvolutionTracker()
    result = tracker.analyze_sentiment("")
    assert result["sentiment"] == "Neutral"
    print("✅ test_empty_sentiment PASSED")


def test_sentiment_batch():
    tracker = SentimentEvolutionTracker()
    articles = ["Great news today!", "Terrible disaster occurred.", "Nothing happened."]
    results = tracker.batch_analyze(articles)
    assert len(results) == 3
    print(f"✅ test_sentiment_batch PASSED: {[r['sentiment'] for r in results]}")


if __name__ == "__main__":
    print("\n=== Running Classification Tests ===\n")
    test_classifier_training()
    test_classifier_prediction()
    test_untrained_classifier()
    test_positive_sentiment()
    test_negative_sentiment()
    test_empty_sentiment()
    test_sentiment_batch()
    print("\n✅ All classification tests passed!\n")
