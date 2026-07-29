"""
data_validator.py — Data Quality Checks for NewsBot 2.0

This module checks if news articles are valid and good quality
before we try to analyze them.

Think of it like a bouncer at a club — it decides what gets in.
"""


class DataValidator:
    """
    Validates news article data before processing.

    Checks for:
    - Empty or missing text
    - Text that is too short or too long
    - Non-string inputs
    - Duplicate articles
    - Language validity

    Example:
        validator = DataValidator(min_length=50, max_length=10000)
        result = validator.validate_article("This is a news article...")
        # → {"valid": True, "checks": {...}}
    """

    def __init__(self, min_length=20, max_length=50000):
        """
        Args:
            min_length: Minimum number of characters required
            max_length: Maximum number of characters allowed
        """
        self.min_length = min_length
        self.max_length = max_length
        self.seen_articles = set()  # For duplicate detection

    def validate_article(self, text):
        """
        Run all validation checks on a single article.

        Args:
            text: The article text to validate

        Returns:
            Dictionary with "valid" (True/False) and details of each check
        """
        checks = {}

        # Check 1: Is it a string?
        checks["is_string"] = isinstance(text, str)

        # Check 2: Is it not empty?
        checks["not_empty"] = bool(text and text.strip())

        # Check 3: Is it long enough?
        checks["min_length"] = len(text.strip()) >= self.min_length if text else False

        # Check 4: Is it not too long?
        checks["max_length"] = len(text.strip()) <= self.max_length if text else False

        # Check 5: Does it contain actual words (not just symbols)?
        checks["has_words"] = bool(text and any(c.isalpha() for c in text))

        # Check 6: Is it a duplicate?
        text_hash = hash(text.strip()) if text else None
        checks["not_duplicate"] = text_hash not in self.seen_articles
        if text_hash and checks["not_duplicate"]:
            self.seen_articles.add(text_hash)

        # Overall result — all checks must pass
        all_valid = all(checks.values())

        return {
            "valid": all_valid,
            "checks": checks,
            "text_length": len(text) if text else 0,
            "issues": [k for k, v in checks.items() if not v]
        }

    def validate_batch(self, texts):
        """
        Validate a list of articles and return only the valid ones.

        Args:
            texts: List of article strings

        Returns:
            Dictionary with valid articles and a summary report
        """
        results = []
        valid_articles = []
        invalid_count = 0

        for i, text in enumerate(texts):
            result = self.validate_article(text)
            results.append(result)

            if result["valid"]:
                valid_articles.append(text)
            else:
                invalid_count += 1

        return {
            "total": len(texts),
            "valid_count": len(valid_articles),
            "invalid_count": invalid_count,
            "valid_articles": valid_articles,
            "validation_details": results
        }

    def generate_quality_report(self, texts):
        """
        Generate a quality report for a collection of articles.

        Args:
            texts: List of article strings

        Returns:
            A summary of data quality metrics
        """
        batch_result = self.validate_batch(texts)

        lengths = [len(t) for t in texts if t]
        avg_length = round(sum(lengths) / max(len(lengths), 1), 1)

        return {
            "total_articles": batch_result["total"],
            "valid_articles": batch_result["valid_count"],
            "invalid_articles": batch_result["invalid_count"],
            "quality_rate": round(
                batch_result["valid_count"] / max(batch_result["total"], 1) * 100, 1
            ),
            "average_length_chars": avg_length,
            "shortest_article": min(lengths) if lengths else 0,
            "longest_article": max(lengths) if lengths else 0,
            "recommendation": (
                "Data quality is good!" if batch_result["valid_count"] / max(batch_result["total"], 1) > 0.9
                else "Consider cleaning the dataset — more than 10% of articles are invalid."
            )
        }

    def reset_duplicates(self):
        """Clear the duplicate tracking history."""
        self.seen_articles = set()
        print("🔄 Duplicate tracking reset.")
