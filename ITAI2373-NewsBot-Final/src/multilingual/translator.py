"""
translator.py — Translation Services for NewsBot 2.0

This module translates news articles from any language into English
(or any other target language) using Google Translate.
"""

from langdetect import detect
from deep_translator import GoogleTranslator


class MultilingualProcessor:
    """
    Handles translation and cross-language analysis of news articles.

    Uses 'deep-translator' with Google Translate as the backend.
    No API key needed for basic usage.

    Example:
        processor = MultilingualProcessor()
        result = processor.translate_text("Hola, cómo estás?", target_language="en")
        # → {"translated_text": "Hello, how are you?", "target_language": "en"}
    """

    def __init__(self):
        self.translator = GoogleTranslator

    def translate_text(self, text, target_language="en"):
        """
        Translate text to a target language.

        Args:
            text: The text to translate
            target_language: Language code to translate into (e.g. "en", "fr", "es")

        Returns:
            Dictionary with translated text and target language
        """
        if not text or not text.strip():
            return {"error": "No text provided."}

        try:
            translated = GoogleTranslator(
                source="auto",
                target=target_language
            ).translate(text)

            return {
                "original_text": text[:100] + ("..." if len(text) > 100 else ""),
                "translated_text": translated,
                "target_language": target_language
            }

        except Exception as e:
            return {
                "error": str(e),
                "original_text": text
            }

    def translate_to_english(self, text):
        """
        Convenience method — translate anything to English.

        Args:
            text: Text in any language

        Returns:
            English translation as a string (or error message)
        """
        result = self.translate_text(text, target_language="en")
        return result.get("translated_text", result.get("error", "Translation failed."))

    def analyze_cross_lingual(self, articles_by_language):
        """
        Compare article collections from different languages.

        Args:
            articles_by_language: Dictionary like:
                {"en": [article1, article2], "fr": [article3], ...}

        Returns:
            Summary statistics per language
        """
        results = {}
        for language, articles in articles_by_language.items():
            results[language] = {
                "article_count": len(articles),
                "total_words": sum(len(a.split()) for a in articles),
                "average_length": round(
                    sum(len(a.split()) for a in articles) / max(len(articles), 1), 1
                )
            }
        return results

    def extract_cultural_context(self, text, source_language):
        """
        Placeholder for cultural context analysis.

        Args:
            text: The source text
            source_language: Language code of the text

        Returns:
            Basic language and context information
        """
        return {
            "detected_language": source_language,
            "english_translation": self.translate_to_english(text),
            "note": "Full cultural context analysis requires domain-specific knowledge bases."
        }

    def batch_translate(self, texts, target_language="en"):
        """
        Translate a list of articles.

        Args:
            texts: List of text strings
            target_language: Language to translate into

        Returns:
            List of translated strings
        """
        results = []
        for text in texts:
            result = self.translate_text(text, target_language)
            results.append(result.get("translated_text", text))
        return results
