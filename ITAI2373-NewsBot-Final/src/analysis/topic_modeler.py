"""
topic_modeler.py — Topic Discovery Engine for NewsBot 2.0

This module finds hidden "themes" or "topics" across a large collection
of news articles without being told what to look for.

Think of it like sorting a big pile of newspapers automatically into
groups like "Economy News", "Sports News", "Health News", etc.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicDiscoveryEngine:
    """
    Discovers hidden topics in a collection of news articles using LDA.

    LDA = Latent Dirichlet Allocation — a popular topic modeling algorithm
    that reads many articles and figures out recurring themes.

    Example:
        engine = TopicDiscoveryEngine(n_topics=5)
        engine.fit_topics(list_of_articles)
        result = engine.get_article_topics("Tesla announced record profits...")
        # → {"main_topic": 2, "topic_scores": [0.05, 0.08, 0.72, 0.10, 0.05]}
    """

    def __init__(self, n_topics=10, method="lda"):
        """
        Args:
            n_topics: How many topics to find (default: 10)
            method:   Algorithm to use — currently supports "lda"
        """
        self.n_topics = n_topics
        self.method = method
        self.is_fitted = False

        # CountVectorizer counts how often each word appears
        self.vectorizer = CountVectorizer(
            stop_words="english",
            max_features=5000,    # Only use the top 5000 most common words
            min_df=2              # Ignore words that appear in fewer than 2 documents
        )

        # LDA finds topics based on word co-occurrence patterns
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,      # Makes results repeatable
            max_iter=20
        )

    def fit_topics(self, documents):
        """
        Train the topic model on a list of articles.

        Args:
            documents: A list of article texts (strings)
        """
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        self.model.fit(self.doc_matrix)
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"✅ Topic model trained. Found {self.n_topics} topics.")

    def get_topic_words(self, topic_id, n_words=10):
        """
        Get the top words that describe a specific topic.

        Args:
            topic_id: The topic number (0 to n_topics-1)
            n_words:  How many words to return

        Returns:
            A list of the most representative words for that topic
        """
        if not self.is_fitted:
            return []

        topic = self.model.components_[topic_id]
        top_indices = topic.argsort()[:-n_words - 1:-1]
        return [self.feature_names[i] for i in top_indices]

    def get_all_topics(self, n_words=8):
        """
        Show all topics with their top words.

        Returns:
            A dictionary like: {0: ["economy", "trade", ...], 1: ["game", "score", ...]}
        """
        if not self.is_fitted:
            return {}

        topics = {}
        for i in range(self.n_topics):
            topics[i] = self.get_topic_words(i, n_words)
        return topics

    def get_article_topics(self, article_text):
        """
        Get the topic distribution for a single article.

        Args:
            article_text: A news article as a string

        Returns:
            Dictionary with the main topic number and all topic scores
        """
        if not self.is_fitted:
            return {"error": "Model not fitted yet. Call fit_topics() first."}

        article_vector = self.vectorizer.transform([article_text])
        topic_scores = self.model.transform(article_vector)[0]

        return {
            "main_topic": int(topic_scores.argmax()),
            "main_topic_words": self.get_topic_words(int(topic_scores.argmax())),
            "topic_scores": {
                f"topic_{i}": round(float(score), 4)
                for i, score in enumerate(topic_scores)
            }
        }

    def track_topic_trends(self, articles_with_dates):
        """
        Analyze how topics change over time.

        Args:
            articles_with_dates: List of (date_string, article_text) tuples

        Returns:
            Summary of topic distribution over time
        """
        if not self.is_fitted:
            return {"error": "Model not fitted yet."}

        results = []
        for date, article in articles_with_dates:
            topic_info = self.get_article_topics(article)
            results.append({
                "date": date,
                "main_topic": topic_info["main_topic"],
                "main_topic_words": topic_info["main_topic_words"]
            })

        return {
            "articles_analyzed": len(articles_with_dates),
            "trend_data": results
        }

    def visualize_topics(self):
        """
        Print a simple text-based visualization of all topics.
        """
        if not self.is_fitted:
            print("❌ Model not fitted yet.")
            return

        print("\n📊 Discovered Topics:")
        print("=" * 50)
        for topic_id in range(self.n_topics):
            words = self.get_topic_words(topic_id)
            print(f"  Topic {topic_id:2d}: {', '.join(words)}")
        print("=" * 50)
