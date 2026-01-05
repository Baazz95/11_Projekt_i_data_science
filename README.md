# Sentiment Analysis Project: Reddit Subreddit Groups

This project collects data from left-leaning and right-leaning Reddit subreddits, performs sentiment analysis, and compares sentiment extremity and polarization across subreddit groups.

## Project Overview

The project analyzes sentiment for the following topics:
- Climate change
- Immigration
- AI Bubble
- Cost of Living
- War in Ukraine

### Goals

This project aims to measure and compare sentiment extremity and polarization across left-leaning and right-leaning subreddit groups on Reddit. The goal is to quantify differences in how emotionally charged discussions are in each group by analyzing sentiment scores, calculating extremity metrics (distance from neutral), measuring polarization (variance and distribution shape), and performing statistical tests to determine if observed differences are significant. This is achieved through automated data collection from Reddit's API across different subreddit groups, sentiment analysis using transformer models, and comprehensive statistical comparison including effect size calculations.

## Project Structure

```
Project/
├── src/
│   ├── data_collection/
│   │   └── reddit_collector.py    # Reddit data collection
│   ├── sentiment_analysis/
│   │   └── analyzer.py            # Sentiment analysis models
│   └── analysis/
│       └── comparison.py           # Cross-platform comparison
├── data/                           # Collected data (CSV files)
├── results/                        # Analysis results and visualizations
├── config.py                       # Configuration settings
├── main.py                         # Main execution script
├── requirements.txt                # Python dependencies
├── setup.bat                       # Windows setup script
└── README.md                       # This file
```

## Setup Instructions

### 1. Create Virtual Environment

**Windows (Quick Setup):**
```bash
setup.bat
```

**Windows (Manual Setup):**
```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure API Credentials

1. Create a `.env` file in the project root
2. Get Reddit API credentials from https://www.reddit.com/prefs/apps
3. Add credentials to `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USER_AGENT=your_user_agent_here
   ```

### 3. Configure Topics and Settings

Edit `config.py` to modify topics, subreddits, collection limits, and sentiment model.

### Run Complete Analysis

```bash
python main.py
```

This executes:
1. Data collection from left-leaning and right-leaning Reddit subreddits (via API)
2. Sentiment analysis using configured model (RoBERTa or VADER)
3. Extremity and polarization calculations
4. Statistical tests comparing subreddit groups
5. Visualization generation
6. Report generation with results

## Data Collection

### Date Scope

The project collects data within a specified date range. The date scope is configured in `config.py`:

- **Start Date**: January 1, 2025 (`start_date: "2025-01-01"`)
- **End Date**: December 31, 2025 (`end_date: "2025-12-31"`)

Only posts created within this date range (inclusive) are included in the analysis. The date filtering is applied after initial data collection to ensure accurate filtering based on post creation timestamps.

To modify the date range, edit the `start_date` and `end_date` fields in `config.py` using the "YYYY-MM-DD" format. Set either value to `None` to disable that boundary filter.

**Note**: The `time_filter` parameter is still used for the initial Reddit API query to improve collection efficiency, but the explicit date range filter ensures only posts from the specified period are included.

### Reddit Subreddit Groups

Uses PRAW library to search subreddits for posts containing topic keywords. The project collects data from two subreddit groups:

**Left-leaning subreddits:**
- r/politics
- r/socialism
- r/LateStageCapitalism

**Right-leaning subreddits:**
- r/Conservative
- r/Republican
- r/libertarian

Configure subreddits in `config.py`. Collects post metadata and optionally comments.

## Sentiment Analysis

### Models

- **VADER**: Fast, rule-based model
- **RoBERTa**: Transformer-based model (recommended)

Configure in `config.py`:
```python
SENTIMENT_MODEL = "roberta"  # or "vader"
```

### Metrics Calculated

**Sentiment:**
- Mean/median sentiment scores
- % positive, negative, neutral
- Standard deviation

**Extremity:**
- Mean extremity (|sentiment_score|)
- % highly extreme (>0.7), moderate (0.4-0.7), neutral (≤0.4)

**Polarization:**
- Variance and standard deviation
- Skewness and kurtosis
- Bimodality coefficient

**Statistical Tests:**
- Mann-Whitney U test
- Levene's test (variance)
- t-test (extremity)
- Cohen's d (effect size)
- Kolmogorov-Smirnov test

## Output Files

### Data Files (`data/`)
- `left_[topic].csv`: Raw left-leaning subreddit data per topic
- `right_[topic].csv`: Raw right-leaning subreddit data per topic
- `left_all.csv`: Combined left-leaning subreddit data
- `right_all.csv`: Combined right-leaning subreddit data
- `left_analyzed.csv`: Left-leaning data with sentiment scores
- `right_analyzed.csv`: Right-leaning data with sentiment scores

### Results Files (`results/`)
- `sentiment_comparison.png`: Basic sentiment comparison charts
- `distribution_extremity_analysis.png`: Distribution plots and extremity analysis
- `extremity_categories.png`: Extremity category breakdown
- `subreddit_group_comparison.csv`: Comparison statistics with metrics
- `comprehensive_analysis.json`: Complete statistical test results
- `extremity_analysis_report.txt`: Detailed analysis report

## Customization

### Add New Topics

Edit `config.py`:
```python
TOPICS = {
    "your_topic": {
        "keywords": ["keyword1", "keyword2"]
    }
}
```

### Modify Subreddits

Edit `config.py`:
```python
REDDIT_SUBREDDITS_LEFT = ["news", "worldnews", "your_subreddit"]
REDDIT_SUBREDDITS_RIGHT = ["conservative", "libertarian", "your_subreddit"]
```

### Adjust Collection Limits

Edit `config.py`:
```python
DATA_COLLECTION = {
    "reddit": {
        "posts_per_topic": 200,
        "comments_per_post": 20,
        "comment_score_threshold": 5,
        "time_filter": "year",  # Used for initial API query
        "start_date": "2025-01-01",  # Date range filter (YYYY-MM-DD)
        "end_date": "2025-12-31"
    }
}
```

### Modify Date Range

To change the date scope of collected data, edit `config.py`:
```python
DATA_COLLECTION = {
    "reddit": {
        # ... other settings ...
        "start_date": "2025-01-01",  # Set to None to remove start date filter
        "end_date": "2025-12-31"     # Set to None to remove end date filter
    }
}
```

## Troubleshooting

### Reddit API Errors
- Verify credentials in `.env`
- Check user agent format
- Ensure rate limits aren't exceeded

### Data Collection Issues
- Verify Reddit API credentials in `.env`
- Check that subreddit names are correct (no r/ prefix needed)
- Some subreddits may have restricted access

### Sentiment Analysis Errors
- VADER works out of the box
- RoBERTa requires transformers library
- Script falls back to VADER if model fails to load

## Limitations

- Sample sizes and subreddit selection may not represent full political spectrum
- Sentiment model accuracy is ~70-80% on political text
- Sarcasm and context may be missed
- Subreddit groups may not perfectly represent left/right political leanings
- Ensure compliance with Reddit API Terms of Service
