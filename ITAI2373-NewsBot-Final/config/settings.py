"""
settings.py — Configuration management for NewsBot 2.0

All system settings are stored here so they are easy to find and change
without having to dig through the code.
"""


class NewsBot2Config:
    """
    Central configuration class for NewsBot 2.0.

    Change settings here to customize how the system behaves.
    """

    def __init__(self):

        # ----- General Settings -----
        self.model_name = "NewsBot2.0"
        self.version = "2.0.0"

        # ----- API Keys (leave blank — add your own keys locally) -----
        self.api_key = ""  # Do NOT put real keys here

        # ----- File Paths -----
        self.data_path = "data/"
        self.raw_data_path = "data/raw/"
        self.processed_data_path = "data/processed/"
        self.models_path = "data/models/"
        self.results_path = "data/results/"

        # ----- Processing Limits -----
        self.max_articles = 100          # Max articles to process at once
        self.max_article_length = 5000   # Max characters per article

        # ----- Classification Settings -----
        self.confidence_threshold = 0.80  # Minimum confidence to accept a prediction

        # ----- Topic Modeling Settings -----
        self.num_topics = 10             # Number of topics to discover
        self.topic_method = "lda"        # Options: "lda" or "nmf"

        # ----- Summarization Settings -----
        self.summary_sentences = 3       # Default number of sentences in a summary

        # ----- Multilingual Settings -----
        self.default_language = "en"     # Default output language
        self.supported_languages = ["en", "es", "fr", "de", "zh", "ar"]

        # ----- Semantic Search Settings -----
        self.embedding_model = "all-MiniLM-L6-v2"
        self.top_k_results = 5           # How many similar articles to return

    def display(self):
        """Print all current settings."""
        print("=" * 40)
        print(f"  {self.model_name} v{self.version} — Configuration")
        print("=" * 40)
        print(f"  Max Articles:        {self.max_articles}")
        print(f"  Topics to Find:      {self.num_topics}")
        print(f"  Topic Method:        {self.topic_method}")
        print(f"  Summary Sentences:   {self.summary_sentences}")
        print(f"  Default Language:    {self.default_language}")
        print(f"  Confidence Min:      {self.confidence_threshold}")
        print("=" * 40)
