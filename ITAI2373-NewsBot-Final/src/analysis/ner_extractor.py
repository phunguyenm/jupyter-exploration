"""
ner_extractor.py — Named Entity Recognition for NewsBot 2.0

This module finds "named entities" in news articles — things like:
- People: "Tim Cook", "Joe Biden"
- Organizations: "Apple", "NASA"
- Locations: "California", "Tokyo"

Think of it like highlighting all the important nouns in a news article.
"""

from transformers import pipeline


class EntityRelationshipMapper:
    """
    Extracts named entities and relationships from news articles using
    a pre-trained Hugging Face Transformers NER model.

    Example:
        mapper = EntityRelationshipMapper()
        entities = mapper.extract_entities("Apple CEO Tim Cook visited California.")
        # → [{"text": "Apple", "label": "ORG"},
        #    {"text": "Tim Cook", "label": "PER"},
        #    {"text": "California", "label": "LOC"}]
    """

    def __init__(self):
        print("⏳ Loading NER model (this may take a moment)...")
        # aggregation_strategy="simple" merges multi-word entities like "Tim Cook"
        self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        print("✅ NER model loaded.")

    def extract_entities(self, article_text):
        """
        Extract all named entities from an article.

        Args:
            article_text: A news article as a string

        Returns:
            A list of dictionaries with "text" and "label" for each entity
        """
        if not article_text or not article_text.strip():
            return []

        try:
            ner_results = self.ner_pipeline(article_text)
            entities = [
                {
                    "text": ent["word"],
                    "label": ent["entity_group"],
                    "confidence": round(float(ent["score"]), 3)
                }
                for ent in ner_results
            ]
            return entities

        except Exception as e:
            return [{"error": str(e)}]

    def extract_relationships(self, article_text):
        """
        Create simple relationships between consecutive entities.

        For example:
        "Apple CEO Tim Cook visited California" →
        "Apple → Tim Cook", "Tim Cook → California"

        Args:
            article_text: A news article as a string

        Returns:
            A list of relationship strings
        """
        entities = self.extract_entities(article_text)

        if len(entities) < 2:
            return []

        relationships = []
        for i in range(len(entities) - 1):
            if "error" not in entities[i] and "error" not in entities[i + 1]:
                relationships.append(
                    f"{entities[i]['text']} ({entities[i]['label']}) "
                    f"→ {entities[i+1]['text']} ({entities[i+1]['label']})"
                )
        return relationships

    def build_knowledge_graph(self, articles):
        """
        Build a simple knowledge graph from multiple articles.

        Each article maps to the entities found in it.

        Args:
            articles: A list of article text strings

        Returns:
            A dictionary mapping article previews to their entities
        """
        graph = {}
        for i, article in enumerate(articles):
            entities = self.extract_entities(article)
            key = f"article_{i}: {article[:50]}..."
            graph[key] = entities
        return graph

    def find_entity_connections(self, entity1, entity2, articles):
        """
        Find articles that mention both entities — showing a connection.

        Args:
            entity1: First entity name (e.g. "Apple")
            entity2: Second entity name (e.g. "Tim Cook")
            articles: List of article strings to search through

        Returns:
            List of articles that mention both entities
        """
        connected = []
        for article in articles:
            if entity1.lower() in article.lower() and entity2.lower() in article.lower():
                connected.append(article[:100] + "...")
        return connected
