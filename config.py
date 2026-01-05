"""
Configuration file for sentiment analysis project.
Modify topics, subreddits, and collection settings here.
"""

# Political topics to analyze
TOPICS = {
    "climate_change": {
        "keywords": ["climate change", "global warming", "climate crisis", "carbon emissions", "greenhouse gas", "renewable energy", "fossil fuels", "climate action", "net zero"]
    },
    "immigration": {
        "keywords": ["immigration", "immigrant", "refugee", "border", "ice", "asylum", "deportation", "visa", "migration", "illegal immigration"]
    },
    "ai_bubble": {
        "keywords": ["ai bubble", "artificial intelligence bubble", "ai crash", "tech bubble", "ai hype", "ai stock", "ai market", "chatgpt", "llm", "ai valuation"]
    },
    "cost_of_living": {
        "keywords": ["cost of living", "inflation", "living costs", "economic crisis", "price increase", "rent", "housing costs", "wages", "purchasing power", "affordability"]
    },
    "ukraine": {
        "keywords": ["ukraine", "zelensky", "putin", "kyiv"]
    }
}

# Reddit subreddit groups to compare
REDDIT_SUBREDDITS_LEFT = [
    "politics",
    "socialism",
    "LateStageCapitalism"
]

REDDIT_SUBREDDITS_RIGHT = [
    "Conservative",
    "Republican",
    "libertarian"
]

# Data collection settings
DATA_COLLECTION = {
    "reddit": {
        "posts_per_topic": 50,  # Number of top posts to collect per topic per subreddit (not per group)
        "comments_per_post": 10,  # Number of top comments to collect per post
        "post_score_threshold": 50,  # Minimum score (upvotes) for posts to be included
        "comment_score_threshold": 5,  # Minimum score for comments to be included
        "time_filter": "year",  # Options: "hour", "day", "week", "month", "year", "all" (used for initial Reddit API query)
        # Date range filter (applied after collection)
        # Set to None to disable date filtering, or specify dates in "YYYY-MM-DD" format
        "start_date": "2025-01-01",  # Inclusive start date (YYYY-MM-DD format)
        "end_date": "2025-12-31"     # Inclusive end date (YYYY-MM-DD format)
    }
}

# Sentiment analysis settings
SENTIMENT_MODEL = "roberta"  # Options: "vader", "roberta"

# Output settings
OUTPUT_DIR = "data"
RESULTS_DIR = "results"

