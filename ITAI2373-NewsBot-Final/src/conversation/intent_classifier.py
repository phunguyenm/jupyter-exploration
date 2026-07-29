"""
intent_classifier.py — Intent Detection for NewsBot 2.0

This module figures out WHAT the user wants when they type a message.
For example: "Show me sports news" → intent is "search"

Think of it like a receptionist who listens to what you say
and figures out which department to send you to.
"""


class IntentClassifier:
    """
    Classifies user queries into intent categories.

    Supported intents:
    - search      → User wants to find articles
    - summarize   → User wants a summary
    - analyze     → User wants sentiment/trend analysis
    - compare     → User wants to compare articles
    - explain     → User wants entity/relationship explanation
    - translate   → User wants translation
    - topics      → User wants topic discovery
    - help        → User needs help using the system
    - general     → Anything else

    Example:
        classifier = IntentClassifier()
        result = classifier.classify("Can you summarize today's tech news?")
        # → {"intent": "summarize", "confidence": 0.9, "topic": "tech"}
    """

    def __init__(self):
        # Keywords that signal each intent
        self.intent_keywords = {
            "search": [
                "find", "search", "show", "get", "fetch", "look for",
                "articles about", "news about", "what happened", "latest"
            ],
            "summarize": [
                "summarize", "summary", "shorten", "brief", "tldr",
                "main points", "key points", "condense", "overview"
            ],
            "analyze": [
                "sentiment", "feeling", "mood", "tone", "opinion",
                "positive", "negative", "analyze", "analysis", "trend"
            ],
            "compare": [
                "compare", "difference", "versus", "vs", "contrast",
                "similar", "different", "both", "between"
            ],
            "explain": [
                "explain", "relationship", "connection", "who is",
                "what is", "tell me about", "describe", "how are"
            ],
            "translate": [
                "translate", "translation", "language", "in english",
                "in spanish", "in french", "convert", "what does this mean"
            ],
            "topics": [
                "topics", "themes", "categories", "what topics",
                "trending", "popular", "discovery", "clusters"
            ],
            "help": [
                "help", "how do i", "what can you", "instructions",
                "guide", "tutorial", "commands", "options"
            ]
        }

        # News topic keywords
        self.topic_keywords = [
            "technology", "tech", "politics", "political", "sports",
            "business", "health", "science", "entertainment", "economy",
            "climate", "environment", "finance", "education", "culture"
        ]

    def classify(self, user_query):
        """
        Classify the intent of a user query.

        Args:
            user_query: The user's message as a string

        Returns:
            Dictionary with intent, confidence, and detected topic
        """
        if not user_query or not user_query.strip():
            return {
                "intent": "general",
                "confidence": 0.0,
                "topic": None,
                "original_query": user_query
            }

        query_lower = user_query.lower()

        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                intent_scores[intent] = score

        # Pick the intent with the highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            total_matches = sum(intent_scores.values())
            confidence = round(intent_scores[best_intent] / max(total_matches, 1), 2)
        else:
            best_intent = "general"
            confidence = 0.5

        # Detect topic
        detected_topic = None
        for topic in self.topic_keywords:
            if topic in query_lower:
                detected_topic = topic
                break

        return {
            "intent": best_intent,
            "confidence": confidence,
            "topic": detected_topic,
            "original_query": user_query,
            "all_scores": intent_scores
        }

    def get_supported_intents(self):
        """Return a list of all supported intents."""
        return list(self.intent_keywords.keys()) + ["general"]

    def explain_intent(self, intent):
        """
        Get a human-readable description of an intent.

        Args:
            intent: Intent string

        Returns:
            Description string
        """
        descriptions = {
            "search": "Find and retrieve news articles on a topic",
            "summarize": "Create a short summary of one or more articles",
            "analyze": "Analyze sentiment, tone, or trends in articles",
            "compare": "Compare coverage between different articles or sources",
            "explain": "Explain entities, relationships, or concepts in articles",
            "translate": "Translate articles or text to another language",
            "topics": "Discover themes and topics across article collections",
            "help": "Get help on how to use NewsBot",
            "general": "General conversation or unrecognized request"
        }
        return descriptions.get(intent, "Unknown intent.")
