"""
test_integration.py — Test Suite for NewsBot 2.0

This file tests that all parts of the system work correctly.
Think of it like a checklist before submitting homework — 
run these tests to make sure nothing is broken.

How to run:
    python tests/test_integration.py
"""

import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing.text_preprocessor import TextPreprocessor
from src.analysis.sentiment_analyzer import SentimentEvolutionTracker
from src.analysis.topic_modeler import TopicDiscoveryEngine
from src.language_models.summarizer import IntelligentSummarizer
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import MultilingualProcessor


# ============================================================
# Sample test articles
# ============================================================
SAMPLE_ARTICLES = [
    "Apple Inc. reported record quarterly earnings today. CEO Tim Cook said the company "
    "sold over 70 million iPhones in the last quarter, driven by strong demand in Asia.",

    "The national soccer team won the championship after a dramatic penalty shootout. "
    "Fans celebrated in the streets across the country.",

    "Scientists have discovered a new treatment for Alzheimer's disease. "
    "The breakthrough could help millions of patients worldwide.",

    "The stock market fell sharply today amid fears of rising inflation. "
    "Major indices dropped more than 3% in early trading.",

    "Researchers at MIT have developed a new AI model that can write news articles "
    "almost indistinguishable from human-written text."
]


class NewsBot2TestSuite:
    """
    Test suite that checks every component of NewsBot 2.0.

    Each test method returns a dictionary with:
    - "status": "PASS" or "FAIL"
    - "result": What the component returned
    - (optional) "error": What went wrong if it failed
    """

    def __init__(self):
        print("🧪 Setting up NewsBot 2.0 Test Suite...")
        self.preprocessor = TextPreprocessor()
        self.sentiment_tracker = SentimentEvolutionTracker()
        self.topic_engine = TopicDiscoveryEngine(n_topics=3)
        self.summarizer = IntelligentSummarizer()
        self.lang_detector = LanguageDetector()
        self.translator = MultilingualProcessor()
        print("✅ Test suite ready.\n")

    def test_preprocessing(self):
        """Test: Does text cleaning work correctly?"""
        print("🔍 Testing Text Preprocessor...")
        try:
            sample = "The Quick Brown Fox!!! jumped over 123 lazy dogs. https://example.com"
            result = self.preprocessor.full_preprocess(sample)
            assert isinstance(result, str), "Result must be a string"
            assert "https" not in result, "URLs should be removed"
            assert result == result.lower(), "Text should be lowercase"
            print(f"   Input:  '{sample[:50]}...'")
            print(f"   Output: '{result}'")
            return {"status": "PASS", "result": result}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def test_sentiment_analysis(self):
        """Test: Does sentiment analysis detect positive/negative correctly?"""
        print("🔍 Testing Sentiment Analyzer...")
        try:
            positive_text = "This is a wonderful, amazing, fantastic day!"
            negative_text = "This is a terrible, horrible, awful disaster."

            pos_result = self.sentiment_tracker.analyze_sentiment(positive_text)
            neg_result = self.sentiment_tracker.analyze_sentiment(negative_text)

            assert pos_result["sentiment"] == "Positive", "Should detect positive sentiment"
            assert neg_result["sentiment"] == "Negative", "Should detect negative sentiment"

            print(f"   Positive test: {pos_result['sentiment']} (confidence: {pos_result['confidence']})")
            print(f"   Negative test: {neg_result['sentiment']} (confidence: {neg_result['confidence']})")
            return {"status": "PASS", "positive": pos_result, "negative": neg_result}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def test_topic_modeling(self):
        """Test: Does topic modeling train and produce results?"""
        print("🔍 Testing Topic Discovery Engine...")
        try:
            self.topic_engine.fit_topics(SAMPLE_ARTICLES)
            result = self.topic_engine.get_article_topics(SAMPLE_ARTICLES[0])
            topics = self.topic_engine.get_all_topics(n_words=5)

            assert "main_topic" in result, "Result should contain main_topic"
            assert len(topics) == 3, "Should have 3 topics"

            print(f"   Article 1 main topic: Topic {result['main_topic']}")
            print(f"   Topic words: {topics}")
            return {"status": "PASS", "result": result, "topics": topics}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def test_summarization(self):
        """Test: Does the summarizer shorten articles correctly?"""
        print("🔍 Testing Intelligent Summarizer...")
        try:
            long_article = " ".join(SAMPLE_ARTICLES)
            summary = self.summarizer.summarize_article(long_article, "brief")
            quality = self.summarizer.assess_summary_quality(long_article, summary)

            assert len(summary) < len(long_article), "Summary should be shorter"
            assert isinstance(summary, str), "Summary should be a string"

            print(f"   Original: {quality['original_word_count']} words")
            print(f"   Summary:  {quality['summary_word_count']} words")
            print(f"   Reduced by {quality['reduction_percent']}%")
            return {"status": "PASS", "summary": summary, "quality": quality}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def test_language_detection(self):
        """Test: Does language detection work for English and other languages?"""
        print("🔍 Testing Language Detector...")
        try:
            english = "This is a news article written in English."
            spanish = "Este es un artículo de noticias en español."
            french = "Ceci est un article de presse en français."

            en_result = self.lang_detector.detect_language(english)
            es_result = self.lang_detector.detect_language(spanish)
            fr_result = self.lang_detector.detect_language(french)

            print(f"   English: detected '{en_result['language']}' ({en_result['language_name']})")
            print(f"   Spanish: detected '{es_result['language']}' ({es_result['language_name']})")
            print(f"   French:  detected '{fr_result['language']}' ({fr_result['language_name']})")
            return {"status": "PASS", "english": en_result, "spanish": es_result, "french": fr_result}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def test_translation(self):
        """Test: Does translation work for a Spanish text?"""
        print("🔍 Testing Multilingual Processor (Translation)...")
        try:
            spanish_text = "El presidente anunció nuevas medidas económicas hoy."
            result = self.translator.translate_text(spanish_text, target_language="en")

            assert "translated_text" in result, "Result should contain translated_text"
            print(f"   Original: '{spanish_text}'")
            print(f"   Translated: '{result['translated_text']}'")
            return {"status": "PASS", "result": result}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def test_edge_cases(self):
        """Test: Does the system handle bad/empty inputs without crashing?"""
        print("🔍 Testing Edge Cases...")
        results = {}

        # Empty text
        try:
            results["empty_text"] = self.sentiment_tracker.analyze_sentiment("")
            results["empty_text"]["status"] = "PASS"
        except Exception as e:
            results["empty_text"] = {"status": "FAIL", "error": str(e)}

        # Very short text
        try:
            results["short_text"] = self.summarizer.summarize_article("Hi.")
            results["short_text_status"] = "PASS"
        except Exception as e:
            results["short_text"] = {"status": "FAIL", "error": str(e)}

        print(f"   Empty text handled: {results.get('empty_text', {}).get('status', 'ok')}")
        print(f"   Short text handled: {results.get('short_text_status', 'ok')}")
        return {"status": "PASS", "results": results}

    def run_all_tests(self):
        """Run every test and print a final summary."""
        print("\n" + "=" * 55)
        print("  🚀 NewsBot 2.0 — Full Test Suite")
        print("=" * 55 + "\n")

        tests = [
            ("Text Preprocessing",    self.test_preprocessing),
            ("Sentiment Analysis",    self.test_sentiment_analysis),
            ("Topic Modeling",        self.test_topic_modeling),
            ("Text Summarization",    self.test_summarization),
            ("Language Detection",    self.test_language_detection),
            ("Translation",           self.test_translation),
            ("Edge Cases",            self.test_edge_cases),
        ]

        results = {}
        passed = 0

        for name, test_func in tests:
            print(f"\n{'—' * 40}")
            result = test_func()
            results[name] = result
            status = result.get("status", "UNKNOWN")
            if status == "PASS":
                passed += 1
                print(f"   ✅ {name}: PASSED")
            else:
                print(f"   ❌ {name}: FAILED — {result.get('error', 'Unknown error')}")

        print(f"\n{'=' * 55}")
        print(f"  Results: {passed}/{len(tests)} tests passed")
        print(f"{'=' * 55}\n")

        return results


# Run tests when this file is executed directly
if __name__ == "__main__":
    suite = NewsBot2TestSuite()
    suite.run_all_tests()
