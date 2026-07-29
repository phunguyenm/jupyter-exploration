"""
text_preprocessor.py — Text Cleaning and Preparation for NewsBot 2.0

This file handles all the "cleaning" work before we analyze text.
Think of it like washing vegetables before you cook them.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (runs once)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class TextPreprocessor:
    """
    Cleans and prepares raw news article text for NLP analysis.

    What it does:
    - Removes extra spaces, special characters, and URLs
    - Converts text to lowercase
    - Removes common "stop words" like 'the', 'is', 'at'
    - Reduces words to their root form (e.g. 'running' → 'run')
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Remove noise from raw text.

        Steps:
        1. Lowercase everything
        2. Remove URLs
        3. Remove punctuation and numbers
        4. Remove extra whitespace
        """
        if not text or not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove punctuation and numbers, keep only letters and spaces
        text = re.sub(r"[^a-z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def remove_stopwords(self, text):
        """
        Remove common words that don't carry much meaning.
        Example: 'the cat sat on the mat' → 'cat sat mat'
        """
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word not in self.stop_words]
        return " ".join(filtered)

    def lemmatize(self, text):
        """
        Reduce words to their base/root form.
        Example: 'running' → 'run', 'better' → 'good'
        """
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(lemmatized)

    def full_preprocess(self, text):
        """
        Run the full cleaning pipeline on a single article.
        Use this for most cases.
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text

    def preprocess_batch(self, texts):
        """
        Clean a list of articles all at once.
        Returns a list of cleaned texts in the same order.
        """
        return [self.full_preprocess(text) for text in texts]

    def validate_text(self, text):
        """
        Check if text is valid before processing.
        Returns True if OK, False if something is wrong.
        """
        if not text:
            return False
        if not isinstance(text, str):
            return False
        if len(text.strip()) < 10:
            return False
        return True
