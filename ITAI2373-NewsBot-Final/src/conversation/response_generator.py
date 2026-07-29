"""
response_generator.py — Response Generation for NewsBot 2.0

This module generates helpful, natural-sounding responses
to user queries based on what NewsBot found.

Think of it like a spokesperson who takes raw data
and turns it into clear, friendly messages.
"""


class ResponseGenerator:
    """
    Generates natural language responses for the NewsBot conversational interface.

    Takes analysis results and formats them into readable, helpful messages.

    Example:
        generator = ResponseGenerator()
        response = generator.generate_response("summarize", {"summary": "Apple profits rose..."})
        # → "Here's a summary of the article: Apple profits rose..."
    """

    def __init__(self):
        # Friendly intro phrases for different intents
        self.intros = {
            "search": [
                "Here's what I found:",
                "I found the following articles:",
                "Based on your search:"
            ],
            "summarize": [
                "Here's a summary:",
                "In brief:",
                "The key points are:"
            ],
            "analyze": [
                "Here's my analysis:",
                "Based on sentiment analysis:",
                "The tone of this content:"
            ],
            "compare": [
                "Here's a comparison:",
                "Looking at both sides:",
                "Comparing the articles:"
            ],
            "explain": [
                "Here's an explanation:",
                "Based on the article:",
                "Let me explain:"
            ],
            "translate": [
                "Here's the translation:",
                "Translated text:",
                "In English:"
            ],
            "topics": [
                "Here are the main topics:",
                "I discovered these themes:",
                "The key topics are:"
            ],
            "help": [
                "Here's how I can help:",
                "NewsBot can do the following:",
                "Available commands:"
            ],
            "general": [
                "I can help you with that.",
                "Here's what I know:",
                "Let me help:"
            ]
        }

    def generate_response(self, intent, result_data, topic=None):
        """
        Generate a natural language response based on intent and results.

        Args:
            intent: The classified intent string
            result_data: Dictionary of results from the relevant component
            topic: Optional topic string for personalization

        Returns:
            A formatted response string
        """
        intro = self.intros.get(intent, self.intros["general"])[0]
        topic_str = f" about {topic}" if topic else ""

        if intent == "summarize":
            summary = result_data.get("summary", "No summary available.")
            return f"{intro}\n\n{summary}"

        elif intent == "analyze":
            sentiment = result_data.get("sentiment", "Unknown")
            confidence = result_data.get("confidence", 0)
            return (
                f"{intro}\n\n"
                f"Sentiment: {sentiment} "
                f"(confidence: {confidence * 100:.0f}%)\n"
                f"This article{topic_str} has a {sentiment.lower()} tone."
            )

        elif intent == "search":
            articles = result_data.get("articles", [])
            count = len(articles)
            return (
                f"{intro}\n\n"
                f"Found {count} article(s){topic_str}.\n"
                + "\n".join(
                    f"• {a[:80]}..." for a in articles[:3]
                )
            )

        elif intent == "topics":
            topics = result_data.get("topics", [])
            return (
                f"{intro}\n\n"
                + "\n".join(
                    f"• Topic {i}: {', '.join(words)}"
                    for i, words in enumerate(topics[:5])
                )
            )

        elif intent == "translate":
            translated = result_data.get("translated_text", "Translation unavailable.")
            return f"{intro}\n\n{translated}"

        elif intent == "help":
            return (
                f"{intro}\n\n"
                "• Search: 'Find articles about technology'\n"
                "• Summarize: 'Summarize this article'\n"
                "• Analyze: 'What is the sentiment of this news?'\n"
                "• Compare: 'Compare these two articles'\n"
                "• Topics: 'What topics are trending?'\n"
                "• Translate: 'Translate this to English'\n"
                "• Explain: 'Who is mentioned in this article?'"
            )

        else:
            message = result_data.get("message", "Here is the result of your request.")
            return f"{intro}\n\n{message}"

    def format_error(self, error_message):
        """
        Format an error message in a friendly way.

        Args:
            error_message: The technical error string

        Returns:
            A user-friendly error response
        """
        return (
            f"I'm sorry, I ran into an issue processing your request.\n"
            f"Details: {error_message}\n"
            f"Please try rephrasing your question or ask for 'help'."
        )

    def format_no_results(self, query):
        """
        Generate a friendly "no results found" message.

        Args:
            query: The original user query

        Returns:
            A helpful no-results message
        """
        return (
            f"I couldn't find any results for '{query}'.\n"
            f"Try:\n"
            f"• Using different keywords\n"
            f"• Being more specific about the topic\n"
            f"• Checking the spelling of key terms"
        )

    def generate_greeting(self):
        """Return a friendly greeting message."""
        return (
            "👋 Hello! I'm NewsBot 2.0, your AI-powered news analysis assistant.\n\n"
            "I can help you:\n"
            "• Search and find news articles\n"
            "• Summarize long articles\n"
            "• Analyze sentiment and tone\n"
            "• Discover topics and trends\n"
            "• Translate multilingual content\n\n"
            "Type 'help' anytime to see all available commands!"
        )
