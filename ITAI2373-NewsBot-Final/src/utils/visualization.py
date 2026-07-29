"""
visualization.py — Visualization Tools for NewsBot 2.0

This module creates charts and graphs to help visualize
news analysis results in a clear, readable way.

Think of it like turning your data into pictures
so patterns are easier to spot.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
from collections import Counter


class NewsVisualizer:
    """
    Creates visualizations for NewsBot 2.0 analysis results.

    Chart types:
    - Sentiment distribution bar chart
    - Topic word clouds (text-based)
    - Article length histogram
    - Sentiment timeline line chart
    - Category pie chart

    Example:
        viz = NewsVisualizer()
        viz.plot_sentiment_distribution(["Positive", "Negative", "Positive", "Neutral"])
    """

    def __init__(self, style="dark_background"):
        """
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except Exception:
            pass  # Use default style if not available
        self.colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0"]

    def plot_sentiment_distribution(self, sentiments, save_path=None):
        """
        Create a bar chart showing the distribution of sentiments.

        Args:
            sentiments: List of sentiment strings ("Positive", "Negative", "Neutral")
            save_path: Optional file path to save the chart

        Returns:
            Path to saved chart or None
        """
        if not sentiments:
            print("No sentiment data to plot.")
            return None

        counts = Counter(sentiments)
        labels = list(counts.keys())
        values = list(counts.values())

        colors = {
            "Positive": "#4CAF50",
            "Negative": "#F44336",
            "Neutral": "#2196F3"
        }
        bar_colors = [colors.get(label, "#9E9E9E") for label in labels]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(value),
                ha="center", va="bottom", fontsize=12, fontweight="bold"
            )

        ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Sentiment", fontsize=11)
        ax.set_ylabel("Number of Articles", fontsize=11)
        ax.set_ylim(0, max(values) * 1.2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Chart saved to {save_path}")
            return save_path

        plt.show()
        plt.close()
        return None

    def plot_topic_keywords(self, topics_dict, save_path=None):
        """
        Create a horizontal bar chart showing top words per topic.

        Args:
            topics_dict: Dictionary like {0: ["word1", "word2", ...], 1: [...]}
            save_path: Optional file path to save the chart
        """
        if not topics_dict:
            print("No topic data to plot.")
            return None

        num_topics = min(len(topics_dict), 5)  # Show max 5 topics
        fig, axes = plt.subplots(1, num_topics, figsize=(4 * num_topics, 4))

        if num_topics == 1:
            axes = [axes]

        for i, (topic_id, words) in enumerate(list(topics_dict.items())[:num_topics]):
            ax = axes[i]
            words_to_show = words[:8]
            y_pos = range(len(words_to_show))

            ax.barh(
                y_pos,
                range(len(words_to_show), 0, -1),
                color=self.colors[i % len(self.colors)],
                alpha=0.8
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words_to_show, fontsize=9)
            ax.set_title(f"Topic {topic_id}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Importance", fontsize=9)

        plt.suptitle("Discovered Topics — Top Keywords", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Chart saved to {save_path}")
            return save_path

        plt.show()
        plt.close()
        return None

    def plot_sentiment_timeline(self, timeline_data, save_path=None):
        """
        Create a line chart showing sentiment over time.

        Args:
            timeline_data: List of {"date": "...", "compound_score": 0.5} dicts
            save_path: Optional file path to save the chart
        """
        if not timeline_data:
            print("No timeline data to plot.")
            return None

        dates = [item.get("date", f"Article {i}") for i, item in enumerate(timeline_data)]
        scores = [item.get("compound_score", 0) for item in timeline_data]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(dates, scores, color="#2196F3", linewidth=2, marker="o", markersize=5)
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.3, linewidth=1)
        ax.fill_between(
            range(len(scores)),
            scores,
            0,
            where=[s > 0 for s in scores],
            color="#4CAF50", alpha=0.2, label="Positive"
        )
        ax.fill_between(
            range(len(scores)),
            scores,
            0,
            where=[s < 0 for s in scores],
            color="#F44336", alpha=0.2, label="Negative"
        )

        ax.set_title("Sentiment Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Article / Date", fontsize=11)
        ax.set_ylabel("Sentiment Score (-1 to +1)", fontsize=11)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.set_ylim(-1.1, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Chart saved to {save_path}")
            return save_path

        plt.show()
        plt.close()
        return None

    def plot_category_distribution(self, categories, save_path=None):
        """
        Create a pie chart showing article category distribution.

        Args:
            categories: List of category strings
            save_path: Optional file path to save the chart
        """
        if not categories:
            print("No category data to plot.")
            return None

        counts = Counter(categories)
        labels = list(counts.keys())
        values = list(counts.values())

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            colors=self.colors[:len(labels)],
            startangle=140,
            pctdistance=0.85
        )

        for text in autotexts:
            text.set_fontsize(9)
            text.set_fontweight("bold")

        ax.set_title("Article Category Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Chart saved to {save_path}")
            return save_path

        plt.show()
        plt.close()
        return None
