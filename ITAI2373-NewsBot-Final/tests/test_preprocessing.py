"""
test_preprocessing.py — Tests for Data Processing Components

Tests for TextPreprocessor, FeatureExtractor, and DataValidator.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.feature_extractor import FeatureExtractor
from src.data_processing.data_validator import DataValidator


def test_text_cleaning():
    preprocessor = TextPreprocessor()
    result = preprocessor.clean_text("Hello WORLD!!! Visit https://example.com today 123")
    assert "https" not in result, "URLs should be removed"
    assert result == result.lower(), "Text should be lowercase"
    assert "123" not in result, "Numbers should be removed"
    print(f"✅ test_text_cleaning PASSED: '{result}'")


def test_stopword_removal():
    preprocessor = TextPreprocessor()
    text = "the cat sat on the mat"
    result = preprocessor.remove_stopwords(text)
    assert "the" not in result.split(), "Stop words should be removed"
    print(f"✅ test_stopword_removal PASSED: '{result}'")


def test_lemmatization():
    preprocessor = TextPreprocessor()
    text = "running cats are jumping"
    result = preprocessor.lemmatize(text)
    assert isinstance(result, str), "Result should be a string"
    print(f"✅ test_lemmatization PASSED: '{result}'")


def test_validation():
    validator = DataValidator(min_length=10)
    assert validator.validate_article("This is a valid article.")["valid"] == True
    assert validator.validate_article("")["valid"] == False
    assert validator.validate_article("Hi")["valid"] == False
    print("✅ test_validation PASSED")


def test_feature_extraction():
    extractor = FeatureExtractor()
    features = extractor.extract_basic_features("Apple reported strong earnings this quarter.")
    assert "word_count" in features
    assert features["word_count"] > 0
    print(f"✅ test_feature_extraction PASSED: {features}")


def test_duplicate_detection():
    validator = DataValidator()
    article = "This is a test article about technology."
    first = validator.validate_article(article)
    second = validator.validate_article(article)
    assert first["checks"]["not_duplicate"] == True
    assert second["checks"]["not_duplicate"] == False
    print("✅ test_duplicate_detection PASSED")


if __name__ == "__main__":
    print("\n=== Running Preprocessing Tests ===\n")
    test_text_cleaning()
    test_stopword_removal()
    test_lemmatization()
    test_validation()
    test_feature_extraction()
    test_duplicate_detection()
    print("\n✅ All preprocessing tests passed!\n")
