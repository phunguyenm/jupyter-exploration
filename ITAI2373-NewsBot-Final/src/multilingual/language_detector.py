"""
language_detector.py — Language Detection for NewsBot 2.0

This module automatically figures out what language an article is written in.
For example: "Hola mundo" → Spanish, "Bonjour" → French, "Hello" → English.
"""

from langdetect import detect, detect_langs


class LanguageDetector:
    """
    Detects the language of news articles automatically.

    Uses the 'langdetect' library which supports 55+ languages.

    Example:
        detector = LanguageDetector()
        result = detector.detect_language("This is an English article.")
        # → {"language": "en", "language_name": "English", "confidence": "High"}
    """

    # Map of language codes to human-readable names
    LANGUAGE_NAMES = {
        "en": "English", "es": "Spanish", "fr": "French",
        "de": "German",  "zh": "Chinese", "ar": "Arabic",
        "pt": "Portuguese", "it": "Italian", "ja": "Japanese",
        "ko": "Korean",  "ru": "Russian", "nl": "Dutch",
        "hi": "Hindi",   "tr": "Turkish", "pl": "Polish"
    }

    def __init__(self):
        pass

    def detect_language(self, text):
        """
        Detect the language of a text.

        Args:
            text: Any text string

        Returns:
            Dictionary with language code, name, and confidence
        """
        if not text or not text.strip():
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": "None"
            }

        try:
            lang_code = detect(text)
            lang_name = self.LANGUAGE_NAMES.get(lang_code, lang_code.upper())

            # Get confidence from detect_langs
            lang_probs = detect_langs(text)
            confidence_score = round(lang_probs[0].prob, 3) if lang_probs else 0.0

            return {
                "language": lang_code,
                "language_name": lang_name,
                "confidence": confidence_score,
                "confidence_level": "High" if confidence_score > 0.9 else "Medium"
            }

        except Exception as e:
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def batch_detect(self, texts):
        """
        Detect language for a list of articles.

        Args:
            texts: List of text strings

        Returns:
            List of detection results in the same order
        """
        return [self.detect_language(text) for text in texts]

    def filter_by_language(self, texts, target_language="en"):
        """
        Filter a list of articles to only keep ones in a specific language.

        Args:
            texts: List of article strings
            target_language: Language code to keep (e.g. "en", "fr")

        Returns:
            List of articles that are in the target language
        """
        filtered = []
        for text in texts:
            result = self.detect_language(text)
            if result["language"] == target_language:
                filtered.append(text)
        return filtered
