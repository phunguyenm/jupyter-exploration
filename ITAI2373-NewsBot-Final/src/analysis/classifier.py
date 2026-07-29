"""
classifier.py — News Article Classification for NewsBot 2.0

This module automatically figures out what category a news article belongs to
(e.g., Sports, Politics, Technology) and tells you how confident it is.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class AdvancedNewsClassifier:
    """
    Classifies news articles into categories with confidence scores.

    How it works:
    - Uses TF-IDF to turn text into numbers the computer understands
    - Uses Naive Bayes to make predictions based on those numbers
    - Returns the predicted category AND how confident it is (0-100%)

    Example:
        classifier = AdvancedNewsClassifier()
        classifier.train(training_texts, training_labels)
        result = classifier.predict_with_confidence("Apple released a new iPhone today.")
        # → {"category": "Technology", "confidence": 0.92, "scores": {...}}
    """

    def __init__(self):
        # TF-IDF converts words into numbers based on how important they are
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # Naive Bayes is a fast, reliable classification algorithm
        self.classifier = MultinomialNB()

        # Track whether the model has been trained yet
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Teach the classifier using example articles and their correct labels.

        Args:
            X_train: List of article texts (the training examples)
            y_train: List of category labels matching each article
                     e.g. ["Technology", "Sports", "Politics", ...]
        """
        X_vectors = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_vectors, y_train)
        self.is_trained = True
        print(f"✅ Classifier trained on {len(X_train)} articles.")

    def predict_with_confidence(self, article_text):
        """
        Predict the category of a new article and return confidence scores.

        Args:
            article_text: A single news article as a string

        Returns:
            A dictionary with:
            - "category": The predicted category name
            - "confidence": How sure the model is (0.0 to 1.0)
            - "scores": Confidence score for every category
        """
        if not self.is_trained:
            return {"error": "Model not trained yet. Call train() first."}

        X = self.vectorizer.transform([article_text])

        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)

        return {
            "category": prediction,
            "confidence": round(float(confidence), 2),
            "scores": dict(
                zip(self.classifier.classes_, probabilities.round(3).tolist())
            )
        }

    def explain_prediction(self, article_text):
        """
        Provide a simple explanation of why the classifier made its decision.

        Args:
            article_text: The article text to explain

        Returns:
            A string explanation
        """
        if not self.is_trained:
            return "Model not trained yet."

        result = self.predict_with_confidence(article_text)
        category = result["category"]
        confidence = result["confidence"]

        explanation = (
            f"The article was classified as '{category}' "
            f"with {confidence * 100:.1f}% confidence. "
            f"This decision is based on the most important words "
            f"found in the article that match patterns from the training data."
        )
        return explanation
