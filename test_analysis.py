"""
Test Analysis Script
A simplified version to test the Reddit connection and run a small analysis.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

def test_reddit_connection():
    """Test Reddit API connection."""
    print("\nTesting Reddit API connection...")
    try:
        from src.data_collection.reddit_collector import RedditCollector
        collector = RedditCollector()
        # Try to access Reddit
        subreddit = collector.reddit.subreddit("news")
        print(f"✅ Successfully connected to Reddit!")
        print(f"   Testing with r/news...")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Reddit: {e}")
        return False

def run_small_test():
    """Run a small test analysis on top posts."""
    print("\n" + "="*80)
    print("RUNNING SMALL TEST ANALYSIS")
    print("="*80)
    
    from src.data_collection.reddit_collector import RedditCollector
    from src.sentiment_analysis.analyzer import SentimentAnalyzer
    import pandas as pd
    
    # Collect top 15 posts directly (no keyword filtering)
    collector = RedditCollector()
    print("\nCollecting top 15 posts from r/news (past week)...")
    
    # Get top posts directly
    subreddit = collector.reddit.subreddit("news")
    posts = []
    for post in subreddit.top(time_filter="week", limit=15):
        try:
            post_data = {
                "platform": "reddit",
                "subreddit": "news",
                "post_id": post.id,
                "title": post.title,
                "text": post.selftext,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "url": post.url,
                "author": str(post.author) if post.author else "[deleted]",
                "full_text": f"{post.title} {post.selftext}".strip(),
                "topic": "test",
                "content_type": "post"
            }
            posts.append(post_data)
        except Exception as e:
            print(f"Error processing post {post.id}: {e}")
            continue
    
    if len(posts) == 0:
        print("❌ No posts collected. Please try again.")
        return
    
    df = pd.DataFrame(posts)
    print(f"✅ Collected {len(df)} posts")
    
    # Analyze sentiment
    print("\nAnalyzing sentiment...")
    analyzer = SentimentAnalyzer(model_type="vader")
    df_analyzed = analyzer.analyze_dataframe(df, text_column="full_text")
    
    # Show results
    summary = analyzer.get_sentiment_summary(df_analyzed)
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total posts analyzed: {summary['total_texts']}")
    print(f"Mean sentiment score: {summary['mean_sentiment_score']:.3f}")
    print(f"Positive: {summary['positive_percentage']:.1f}%")
    print(f"Negative: {summary['negative_percentage']:.1f}%")
    print(f"Neutral: {summary['neutral_percentage']:.1f}%")
    
    # Show sample posts
    print("\nSample posts:")
    for idx, row in df_analyzed.head(3).iterrows():
        print(f"\n[{row['sentiment_label'].upper()}] Score: {row['sentiment_score']:.3f}")
        print(f"Title: {row['title'][:100]}...")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEST ANALYSIS - SENTIMENT ANALYSIS PROJECT")
    print("="*80)
    
    if not test_reddit_connection():
        print("\nPlease check your Reddit API credentials.")
        sys.exit(1)
    
    # Run test analysis
    run_small_test()

