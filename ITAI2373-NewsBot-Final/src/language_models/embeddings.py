"""
embeddings.py — Semantic Search Engine for NewsBot 2.0

This module finds articles that are SIMILAR IN MEANING to a query,
even if they don't share the exact same words.

Example: "automobile accident" would match articles about "car crash"
because the MEANING is similar, not just the words.

This works by converting text into "embeddings" — lists of numbers
that represent the meaning of the text in a way computers understand.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearchEngine:
    """
    Finds semantically similar articles using sentence embeddings.

    Uses the 'all-MiniLM-L6-v2' model from Hugging Face — a fast,
    accurate model for measuring text similarity.

    Example:
        engine = SemanticSearchEngine()
        engine.encode_documents(list_of_articles)
        results = engine.find_similar_articles("climate change policy")
        # → [{"article": "...", "similarity": 0.89}, ...]
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("⏳ Loading semantic search model...")
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = None
        self.documents = []
        print("✅ Semantic search model loaded.")

    def encode_documents(self, documents):
        """
        Convert a list of articles into embedding vectors.
        Must be called before searching.

        Args:
            documents: List of article text strings

        Returns:
            The embedding array (also stored internally)
        """
        self.documents = documents
        self.document_embeddings = self.model.encode(documents)
        print(f"✅ Encoded {len(documents)} documents.")
        return self.document_embeddings

    def find_similar_articles(self, query_text, top_k=5):
        """
        Find the most similar articles to a query text.

        Args:
            query_text: The search query (e.g., "AI technology breakthroughs")
            top_k: How many results to return (default: 5)

        Returns:
            List of dictionaries with "article" text and "similarity" score
        """
        if self.document_embeddings is None:
            return [{"error": "No documents encoded yet. Call encode_documents() first."}]

        query_embedding = self.model.encode([query_text])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]

        # Get the top_k most similar indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "article_preview": self.documents[idx][:150] + "...",
                "similarity": round(float(similarities[idx]), 4)
            })

        return results

    def semantic_search(self, query_text, article_database):
        """
        Search a new set of articles without pre-encoding.
        Convenience method that encodes and searches in one step.

        Args:
            query_text: The search query string
            article_database: List of article strings to search through

        Returns:
            Top similar articles
        """
        self.encode_documents(article_database)
        return self.find_similar_articles(query_text)

    def cluster_similar_content(self, articles, threshold=0.75):
        """
        Group articles that are very similar to each other.

        Args:
            articles: List of article strings
            threshold: Similarity score above which articles are considered "similar"

        Returns:
            A similarity matrix (2D array of scores between all article pairs)
        """
        embeddings = self.model.encode(articles)
        similarity_matrix = cosine_similarity(embeddings)

        # Build simple groups
        groups = []
        used = set()

        for i in range(len(articles)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(articles)):
                if similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    used.add(j)
            groups.append(group)
            used.add(i)

        return {
            "groups": groups,
            "similarity_matrix_shape": similarity_matrix.shape,
            "total_articles": len(articles)
        }
