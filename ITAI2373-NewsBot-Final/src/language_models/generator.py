"""
generator.py — Content Generation for NewsBot 2.0

This module generates new text content based on news articles,
such as headlines, tags, and insight summaries.

Think of it like a writing assistant that reads your articles
and helps you create useful content from them.
"""

import re
from collections import Counter


class ContentGenerator:
    """
    Generates content from news articles including headlines,
    tags, and insight summaries.

    Example:
        generator = ContentGenerator()
        headline = generator.generate_headline("Apple reported record profits...")
        # → "Apple Reports Record Profits in Latest Quarter"
    """

    def __init__(self):
        # Common words to avoid when generating tags
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "was",
            "are", "were", "be", "been", "has", "have", "had", "that",
            "this", "it", "its", "as", "said", "says", "will", "can"
        }

    def generate_headline(self, article_text, max_words=12):
        """
        Generate a headline from an article by extracting
        the most informative sentence and shortening it.

        Args:
            article_text: The full article text
            max_words: Maximum words in the headline

        Returns:
            A headline string
        """
        if not article_text or not article_text.strip():
            return "No content available."

        # Split into sentences
        sentences = [s.strip() for s in article_text.split(".") if s.strip()]

        if not sentences:
            return article_text[:100]

        # Use the first sentence as base (usually most informative in news)
        headline = sentences[0]

        # Trim to max_words
        words = headline.split()
        if len(words) > max_words:
            headline = " ".join(words[:max_words]) + "..."

        # Title case
        return headline.title()

    def generate_tags(self, article_text, num_tags=5):
        """
        Generate keyword tags for an article by finding
        the most frequently used meaningful words.

        Args:
            article_text: The article text
            num_tags: How many tags to generate

        Returns:
            A list of tag strings
        """
        if not article_text:
            return []

        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{4,}\b', article_text.lower())

        # Remove stop words
        meaningful_words = [w for w in words if w not in self.stop_words]

        # Count frequencies
        word_counts = Counter(meaningful_words)

        # Return top tags
        tags = [word for word, count in word_counts.most_common(num_tags)]
        return tags

    def generate_insight_summary(self, articles):
        """
        Generate a high-level insight summary from multiple articles.

        Args:
            articles: List of article text strings

        Returns:
            A dictionary with key insights
        """
        if not articles:
            return {"error": "No articles provided."}

        # Collect all words across articles
        all_words = []
        for article in articles:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', article.lower())
            all_words.extend([w for w in words if w not in self.stop_words])

        word_counts = Counter(all_words)
        top_themes = [word for word, _ in word_counts.most_common(10)]

        # Generate per-article headlines
        headlines = [self.generate_headline(a) for a in articles[:5]]

        return {
            "total_articles_analyzed": len(articles),
            "top_themes": top_themes,
            "sample_headlines": headlines,
            "insight": (
                f"Analysis of {len(articles)} articles reveals recurring themes: "
                f"{', '.join(top_themes[:5])}."
            )
        }

    def generate_report_intro(self, topic, article_count, date_range=None):
        """
        Generate an introduction paragraph for a news report.

        Args:
            topic: The main topic of the report
            article_count: Number of articles analyzed
            date_range: Optional string like "July 2026"

        Returns:
            An introduction paragraph string
        """
        date_str = f" from {date_range}" if date_range else ""

        return (
            f"This report presents an analysis of {article_count} news articles "
            f"related to {topic}{date_str}. "
            f"Using advanced Natural Language Processing techniques, the system "
            f"identified key themes, sentiment patterns, and notable entities "
            f"across the collected articles. "
            f"The findings below highlight the most significant trends and insights "
            f"discovered through automated analysis."
        )

    def expand_query(self, query):
        """
        Expand a user search query with related terms.

        Args:
            query: The original search query string

        Returns:
            Dictionary with original query and suggested expansions
        """
        # Simple keyword expansion using common news synonyms
        expansions = {
            "ai": ["artificial intelligence", "machine learning", "deep learning"],
            "economy": ["economic", "financial", "gdp", "market"],
            "climate": ["environment", "global warming", "carbon", "emissions"],
            "election": ["vote", "campaign", "candidate", "ballot"],
            "tech": ["technology", "software", "digital", "innovation"],
            "health": ["medical", "healthcare", "disease", "treatment"],
            "war": ["conflict", "military", "troops", "ceasefire"]
        }

        query_lower = query.lower()
        suggested = []

        for keyword, related in expansions.items():
            if keyword in query_lower:
                suggested.extend(related)

        return {
            "original_query": query,
            "expanded_terms": suggested if suggested else ["No expansions found."],
            "full_query": query + " " + " ".join(suggested) if suggested else query
        }
