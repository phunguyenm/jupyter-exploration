"""
query_processor.py — Conversational Interface for NewsBot 2.0

This module lets users chat with NewsBot using natural language.
It figures out what the user wants (their "intent") and responds helpfully.

Example conversation:
    User: "Summarize the top technology articles"
    Bot:  "I will create a summary of technology articles for you..."

    User: "What's the sentiment of today's news?"
    Bot:  "I will analyze the sentiment of today's articles..."
"""


class ConversationalInterface:
    """
    Handles natural language queries from users and routes them to
    the correct NewsBot component.

    Intents it can recognize:
    - "search"    → find articles
    - "summarize" → summarize articles
    - "analyze"   → sentiment analysis
    - "compare"   → compare multiple articles
    - "explain"   → explain entity relationships
    - "general"   → anything else

    Example:
        bot = ConversationalInterface(newsbot_system)
        response = bot.process_query("Can you summarize today's top articles?")
        # → {"intent": "summarize", "response": "I will create a summary..."}
    """

    def __init__(self, newsbot_system):
        """
        Args:
            newsbot_system: The main NewsBot2IntegratedSystem instance
        """
        self.newsbot = newsbot_system
        self.conversation_history = []

    def classify_intent(self, user_query):
        """
        Figure out what the user wants based on keywords in their message.

        Args:
            user_query: The user's message as a string

        Returns:
            An intent string: "summarize", "analyze", "search", etc.
        """
        query = user_query.lower()

        if any(word in query for word in ["summarize", "summary", "shorten", "brief"]):
            return "summarize"

        elif any(word in query for word in ["compare", "difference", "versus", "vs"]):
            return "compare"

        elif any(word in query for word in ["sentiment", "feeling", "mood", "opinion", "positive", "negative"]):
            return "analyze"

        elif any(word in query for word in ["explain", "relationship", "connection", "who is", "what is"]):
            return "explain"

        elif any(word in query for word in ["find", "search", "show", "get", "fetch", "articles about"]):
            return "search"

        elif any(word in query for word in ["translate", "language", "spanish", "french"]):
            return "translate"

        else:
            return "general"

    def extract_query_entities(self, user_query):
        """
        Pull out key information from the user's message.

        For example, from "Find positive articles about technology this week":
        - keywords: ["positive", "articles", "technology", "week"]
        - sentiment: "positive"
        - timeframe: "week"

        Args:
            user_query: The user's message

        Returns:
            Dictionary with extracted entities
        """
        entities = {
            "keywords": [],
            "sentiment": None,
            "timeframe": None,
            "topic": None
        }

        words = user_query.lower().split()

        # Extract sentiment hints
        for word in words:
            if word in ["positive", "negative", "neutral"]:
                entities["sentiment"] = word

        # Extract timeframe hints
        for word in words:
            if word in ["today", "yesterday", "week", "month", "year", "recent", "latest"]:
                entities["timeframe"] = word

        # Extract topic hints
        topics = ["technology", "politics", "sports", "business", "health",
                  "science", "entertainment", "economy", "climate"]
        for word in words:
            if word in topics:
                entities["topic"] = word

        entities["keywords"] = [w for w in words if len(w) > 3]

        return entities

    def process_query(self, user_query, conversation_context=None):
        """
        Main method — process a user's message and return a response.

        Args:
            user_query: The user's natural language message
            conversation_context: Optional previous messages for context

        Returns:
            Dictionary with intent, entities, and response text
        """
        intent = self.classify_intent(user_query)
        entities = self.extract_query_entities(user_query)
        response = self.generate_response(intent, entities)

        # Save to history for follow-up question handling
        self.conversation_history.append({
            "query": user_query,
            "intent": intent,
            "entities": entities,
            "response": response
        })

        return {
            "query": user_query,
            "intent": intent,
            "entities": entities,
            "response": response
        }

    def generate_response(self, intent, entities):
        """
        Generate a helpful response based on the detected intent.

        Args:
            intent: The classified intent string
            entities: Dictionary of extracted entities

        Returns:
            A response string
        """
        topic = entities.get("topic", "the requested topic")
        timeframe = entities.get("timeframe", "")
        timeframe_str = f" from {timeframe}" if timeframe else ""

        responses = {
            "search": (
                f"I will search for articles about {topic}{timeframe_str}. "
                f"Use the SemanticSearchEngine to find related content."
            ),
            "summarize": (
                f"I will create a summary of {topic} articles{timeframe_str}. "
                f"The IntelligentSummarizer will extract the key points."
            ),
            "analyze": (
                f"I will analyze the sentiment of {topic} articles{timeframe_str}. "
                f"The SentimentEvolutionTracker will measure the mood of the news."
            ),
            "compare": (
                f"I will compare articles about {topic}{timeframe_str}. "
                f"I'll look for differences and similarities in coverage."
            ),
            "explain": (
                f"I will explain the relationships between entities in {topic} articles. "
                f"The EntityRelationshipMapper will find connections."
            ),
            "translate": (
                f"I will detect the language and translate articles for you. "
                f"The MultilingualProcessor handles 55+ languages."
            ),
            "general": (
                "I can help you search, summarize, analyze sentiment, compare articles, "
                "explain entity relationships, or translate news. What would you like to do?"
            )
        }

        return responses.get(intent, responses["general"])

    def handle_follow_up(self, follow_up_query):
        """
        Handle a follow-up question that refers to the previous conversation.

        Args:
            follow_up_query: The follow-up message from the user

        Returns:
            Response with context from the previous exchange
        """
        if not self.conversation_history:
            return self.process_query(follow_up_query)

        previous = self.conversation_history[-1]
        previous_intent = previous["intent"]
        previous_topic = previous["entities"].get("topic", "the same topic")

        response = (
            f"Continuing from your previous '{previous_intent}' request about {previous_topic}. "
            f"Now processing: '{follow_up_query}'"
        )

        return {
            "query": follow_up_query,
            "previous_intent": previous_intent,
            "response": response
        }

    def get_conversation_history(self):
        """Return the full conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("🔄 Conversation history cleared.")
