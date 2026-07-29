"""
cross_lingual_analyzer.py — Cross-Language Analysis for NewsBot 2.0

This module compares how the same news story is covered
in different languages and countries.

Think of it like reading the same news story in 5 different
newspapers from around the world and comparing what they say.
"""

from deep_translator import GoogleTranslator
from langdetect import detect


class CrossLingualAnalyzer:
    """
    Compares news coverage across different languages.

    What it can do:
    - Translate articles from any language to English
    - Compare how different languages cover the same topic
    - Find differences in tone, focus, and perspective
    - Identify which languages cover a topic most extensively

    Example:
        analyzer = CrossLingualAnalyzer()
        result = analyzer.compare_coverage({
            "en": ["US article about climate..."],
            "fr": ["French article about climate..."]
        })
    """

    def __init__(self):
        self.translator = GoogleTranslator

    def translate_to_english(self, text):
        """
        Translate any text to English.

        Args:
            text: Text in any language

        Returns:
            English translation string
        """
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception as e:
            return f"[Translation failed: {str(e)}]"

    def detect_language(self, text):
        """
        Detect what language a text is written in.

        Args:
            text: Any text string

        Returns:
            Language code string (e.g. "en", "fr", "es")
        """
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def compare_coverage(self, articles_by_language):
        """
        Compare how different languages cover the same topic.

        Args:
            articles_by_language: Dictionary like:
                {
                    "en": ["article1", "article2"],
                    "fr": ["article3"],
                    "es": ["article4", "article5"]
                }

        Returns:
            Comparison report dictionary
        """
        if not articles_by_language:
            return {"error": "No articles provided."}

        report = {}

        for language, articles in articles_by_language.items():
            # Translate all articles to English for comparison
            translated = []
            for article in articles:
                if language != "en":
                    translated_text = self.translate_to_english(article)
                else:
                    translated_text = article
                translated.append(translated_text)

            # Basic stats per language
            total_words = sum(len(a.split()) for a in articles)
            avg_length = round(total_words / max(len(articles), 1), 1)

            report[language] = {
                "article_count": len(articles),
                "total_words": total_words,
                "average_article_length": avg_length,
                "translated_samples": [t[:100] + "..." for t in translated[:2]]
            }

        # Find which language has the most coverage
        most_coverage = max(
            report.keys(),
            key=lambda lang: report[lang]["total_words"]
        )

        return {
            "languages_analyzed": list(articles_by_language.keys()),
            "coverage_by_language": report,
            "most_extensive_coverage": most_coverage,
            "insight": (
                f"'{most_coverage}' language sources provided the most extensive "
                f"coverage with {report[most_coverage]['total_words']} total words."
            )
        }

    def find_perspective_differences(self, same_topic_articles):
        """
        Find differences in how languages frame the same topic.

        Args:
            same_topic_articles: Dictionary of {language: article_text}

        Returns:
            Analysis of perspective differences
        """
        if not same_topic_articles:
            return {"error": "No articles provided."}

        perspectives = {}

        for language, article in same_topic_articles.items():
            # Translate to English
            english_version = (
                self.translate_to_english(article)
                if language != "en" else article
            )

            # Count words as a simple proxy for coverage depth
            word_count = len(english_version.split())

            # Get first sentence as the "angle" or framing
            first_sentence = english_version.split(".")[0] if english_version else ""

            perspectives[language] = {
                "word_count": word_count,
                "opening_angle": first_sentence[:150],
                "coverage_depth": (
                    "Detailed" if word_count > 300
                    else "Moderate" if word_count > 100
                    else "Brief"
                )
            }

        return {
            "topic_perspectives": perspectives,
            "note": (
                "Perspective differences reflect how media in different "
                "languages frame and prioritize the same story."
            )
        }

    def summarize_multilingual_coverage(self, articles_by_language):
        """
        Create a unified summary combining insights from all languages.

        Args:
            articles_by_language: Dictionary of {language: [articles]}

        Returns:
            A unified coverage summary
        """
        total_articles = sum(len(v) for v in articles_by_language.values())
        total_languages = len(articles_by_language)

        # Translate and combine a sample from each language
        combined_text = []
        for language, articles in articles_by_language.items():
            for article in articles[:1]:  # Take 1 sample per language
                translated = (
                    self.translate_to_english(article)
                    if language != "en" else article
                )
                combined_text.append(translated)

        return {
            "total_articles": total_articles,
            "languages_covered": total_languages,
            "language_list": list(articles_by_language.keys()),
            "combined_sample_count": len(combined_text),
            "summary": (
                f"Coverage spans {total_articles} articles across "
                f"{total_languages} languages: "
                f"{', '.join(articles_by_language.keys())}."
            )
        }
