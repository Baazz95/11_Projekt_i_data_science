"""
Reddit Data Collector
Collects posts and comments from specified subreddits based on topic search queries.
"""

import praw
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
from datetime import datetime
import logging

load_dotenv()


class RedditCollector:
    """Collects data from Reddit using PRAW."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize Reddit API client.
        
        Args:
            log_file: Optional path to log file for collection statistics. If None, logging is disabled.
        """
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment_analysis_project/1.0")
        )
        
        # Set up logging for collection statistics
        self.logger = None
        if log_file:
            # Create logger
            self.logger = logging.getLogger(f"RedditCollector_{id(self)}")
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()  # Clear any existing handlers
            
            # Create file handler
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
        
    def search_subreddit(self, subreddit_name: str, search_query: str, 
                        limit: int = 100, time_filter: str = "month",
                        start_date: Optional[str] = None, end_date: Optional[str] = None,
                        min_score: int = 0, exclude_post_ids: Optional[set] = None) -> Tuple[List[Dict], Dict]:
        """
        Search for posts in a subreddit using Reddit's search API.
        
        Args:
            subreddit_name: Name of the subreddit (without r/)
            search_query: Search query string (topic name)
            limit: Maximum number of posts to retrieve
            time_filter: Time filter for posts ("hour", "day", "week", "month", "year", "all")
            start_date: Start date filter in "YYYY-MM-DD" format (inclusive). If None, no start filter.
            end_date: End date filter in "YYYY-MM-DD" format (inclusive). If None, no end filter.
            min_score: Minimum score (upvotes) for posts to be included
            exclude_post_ids: Set of post IDs to exclude (to avoid duplicates)
            exclude_post_ids: Set of post IDs to exclude (to avoid duplicates)
        
        Returns:
            Tuple of (list of dictionaries containing post data, statistics dictionary)
        """
        if exclude_post_ids is None:
            exclude_post_ids = set()
        
        posts = []
        stats = {
            "checked": 0,
            "filtered_by_date": 0,
            "filtered_by_score": 0,
            "filtered_by_duplicate": 0,
            "collected": 0
        }
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # Convert date strings to Unix timestamps if provided
        start_timestamp = None
        end_timestamp = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_timestamp = int(start_dt.timestamp())
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            # Set to end of day (23:59:59)
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
            end_timestamp = int(end_dt.timestamp())
        
        try:
            # Use Reddit's search API to find posts matching the topic query, sorted by top
            # Fetch more than limit to account for date/score filtering
            fetch_limit = min(limit * 3, 1000)  # Fetch up to 3x target or Reddit's max of 1000
            searched_posts = subreddit.search(
                query=search_query,
                sort="top",
                time_filter=time_filter,
                limit=fetch_limit
            )
            
            # Filter posts by date range and score
            for post in tqdm(searched_posts, desc=f"Searching r/{subreddit_name} for '{search_query}'"):
                try:
                    stats["checked"] += 1
                    
                    # Filter by date range first (more efficient)
                    if start_timestamp and post.created_utc < start_timestamp:
                        stats["filtered_by_date"] += 1
                        continue  # Post is before start date, skip
                    if end_timestamp and post.created_utc > end_timestamp:
                        stats["filtered_by_date"] += 1
                        continue  # Post is after end date, skip
                    
                    # Filter by score
                    if post.score < min_score:
                        stats["filtered_by_score"] += 1
                        continue  # Post score too low, skip
                    
                    # Filter by duplicate (exclude already collected posts)
                    if post.id in exclude_post_ids:
                        stats["filtered_by_duplicate"] += 1
                        continue  # Post already collected, skip
                    
                    # All remaining posts match search query, date range, score, and are not duplicates
                    post_data = {
                        "platform": "reddit",
                        "subreddit": subreddit_name,
                        "post_id": post.id,
                        "title": post.title,
                        "text": post.selftext,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc,
                        "url": post.url,
                        "author": str(post.author) if post.author else "[deleted]",
                        "full_text": f"{post.title} {post.selftext}".strip()
                    }
                    posts.append(post_data)
                    stats["collected"] += 1
                    
                    # Stop once we have enough posts
                    if len(posts) >= limit:
                        break
                        
                except Exception as e:
                    print(f"Error processing post {post.id}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error searching r/{subreddit_name}: {e}")
            
        return posts, stats
    
    def get_post_comments(self, post_id: str, limit: int = 10, min_score: int = 0) -> List[Dict]:
        """
        Get top comments from a specific post, filtered by score threshold.
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to retrieve
            min_score: Minimum score threshold for comments (default: 0)
        
        Returns:
            List of dictionaries containing comment data, sorted by score (highest first)
        """
        comments = []
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments" placeholders
            
            # Collect all comments with their scores
            all_comments = []
            for comment in submission.comments.list():
                if hasattr(comment, 'body') and comment.body != "[deleted]":
                    comment_data = {
                        "platform": "reddit",
                        "post_id": post_id,
                        "comment_id": comment.id,
                        "text": comment.body,
                        "score": comment.score,
                        "created_utc": comment.created_utc,
                        "author": str(comment.author) if comment.author else "[deleted]",
                        "full_text": comment.body
                    }
                    all_comments.append(comment_data)
            
            # Filter by score threshold and sort by score (descending)
            filtered_comments = [c for c in all_comments if c["score"] >= min_score]
            filtered_comments.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top N comments
            comments = filtered_comments[:limit]
            
        except Exception as e:
            print(f"Error fetching comments for post {post_id}: {e}")
            
        return comments
    
    def collect_topic_data(self, topic_name: str, fallback_keywords: Optional[List[str]] = None,
                          subreddits: List[str] = None, posts_per_subreddit: int = 50,
                          comments_per_post: int = 10, time_filter: str = "year",
                          comment_score_threshold: int = 5, post_score_threshold: int = 0,
                          start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Collect all data for a specific topic across multiple subreddits.
        First tries the topic name, then falls back to keywords if not enough posts are found.
        
        Args:
            topic_name: Name of the topic (underscores will be converted to spaces for search)
            fallback_keywords: Optional list of backup keywords to search if topic name doesn't yield enough posts
            subreddits: List of subreddit names to search
            posts_per_subreddit: Number of top posts to collect per subreddit
            comments_per_post: Number of top comments to collect per post
            time_filter: Time filter for posts (used for initial Reddit API query)
            comment_score_threshold: Minimum score for comments to be included
            post_score_threshold: Minimum score for posts to be included
            start_date: Start date filter in "YYYY-MM-DD" format (inclusive). If None, no start filter.
            end_date: End date filter in "YYYY-MM-DD" format (inclusive). If None, no end filter.
        
        Returns:
            DataFrame containing all collected data
        """
        if subreddits is None:
            subreddits = []
        if fallback_keywords is None:
            fallback_keywords = []
        
        all_data = []
        
        # Convert topic name to search query (replace underscores with spaces)
        primary_search_query = topic_name.replace("_", " ")
        
        for subreddit in subreddits:
            print(f"\nCollecting data from r/{subreddit} for topic: {topic_name}")
            
            # First, try searching with the topic name
            collected_post_ids = set()
            all_posts = []
            cumulative_stats = {
                "checked": 0,
                "filtered_by_date": 0,
                "filtered_by_score": 0,
                "filtered_by_duplicate": 0,
                "collected": 0
            }
            
            # Primary search with topic name
            posts, stats = self.search_subreddit(
                subreddit_name=subreddit,
                search_query=primary_search_query,
                limit=posts_per_subreddit,
                time_filter=time_filter,
                start_date=start_date,
                end_date=end_date,
                min_score=post_score_threshold,
                exclude_post_ids=collected_post_ids
            )
            
            # Update cumulative stats and collected post IDs
            for key in cumulative_stats:
                cumulative_stats[key] += stats.get(key, 0)
            for post in posts:
                collected_post_ids.add(post["post_id"])
                all_posts.append(post)
            
            # If we don't have enough posts, try fallback keywords
            if len(all_posts) < posts_per_subreddit and fallback_keywords:
                remaining_needed = posts_per_subreddit - len(all_posts)
                print(f"  Only found {len(all_posts)}/{posts_per_subreddit} posts. Trying fallback keywords...")
                
                for keyword in fallback_keywords:
                    if len(all_posts) >= posts_per_subreddit:
                        break
                    
                    # Skip the keyword if it's the same as the primary search query
                    if keyword.lower() == primary_search_query.lower():
                        continue
                    
                    remaining_needed = posts_per_subreddit - len(all_posts)
                    keyword_posts, keyword_stats = self.search_subreddit(
                        subreddit_name=subreddit,
                        search_query=keyword,
                        limit=remaining_needed,
                        time_filter=time_filter,
                        start_date=start_date,
                        end_date=end_date,
                        min_score=post_score_threshold,
                        exclude_post_ids=collected_post_ids
                    )
                    
                    # Update cumulative stats
                    for key in cumulative_stats:
                        cumulative_stats[key] += keyword_stats.get(key, 0)
                    
                    # Add new posts (avoiding duplicates)
                    for post in keyword_posts:
                        if post["post_id"] not in collected_post_ids:
                            collected_post_ids.add(post["post_id"])
                            all_posts.append(post)
                            if len(all_posts) >= posts_per_subreddit:
                                break
                    
                    # Rate limiting between keyword searches
                    time.sleep(0.5)
            
            # Now collect comments for all posts and calculate statistics
            comments_collected = 0
            total_comment_upvotes = 0
            
            for post in all_posts:
                post["topic"] = topic_name
                post["content_type"] = "post"
                all_data.append(post)
                
                # Collect comments if requested
                if comments_per_post > 0:
                    comments = self.get_post_comments(
                        post["post_id"], 
                        limit=comments_per_post,
                        min_score=comment_score_threshold
                    )
                    for comment in comments:
                        comment["topic"] = topic_name
                        comment["content_type"] = "comment"
                        all_data.append(comment)
                        total_comment_upvotes += comment.get("score", 0)
                    comments_collected += len(comments)
                    
                    # Rate limiting - be respectful to Reddit API
                    time.sleep(0.5)
            
            # Calculate average upvotes for collected posts
            avg_post_upvotes = 0.0
            if all_posts:
                total_post_upvotes = sum(post.get("score", 0) for post in all_posts)
                avg_post_upvotes = total_post_upvotes / len(all_posts)
            
            # Calculate average comment upvotes per post
            avg_comment_upvotes = 0.0
            if comments_collected > 0:
                avg_comment_upvotes = total_comment_upvotes / comments_collected
            
            # Print detailed statistics
            print(f"  📊 Collection Statistics for r/{subreddit}:")
            print(f"     Posts checked: {cumulative_stats['checked']}")
            print(f"     Filtered by date: {cumulative_stats['filtered_by_date']}")
            print(f"     Filtered by score (<{post_score_threshold}): {cumulative_stats['filtered_by_score']}")
            if cumulative_stats.get('filtered_by_duplicate', 0) > 0:
                print(f"     Filtered by duplicate: {cumulative_stats['filtered_by_duplicate']}")
            print(f"     ✅ Posts collected: {len(all_posts)} (target: {posts_per_subreddit})")
            print(f"     💬 Comments collected: {comments_collected}")
            print(f"     📈 Average upvotes per post: {avg_post_upvotes:.2f}")
            print(f"     💬 Average upvotes per comment: {avg_comment_upvotes:.2f}")
            
            if len(all_posts) < posts_per_subreddit:
                print(f"     ⚠️  Warning: Only collected {len(all_posts)}/{posts_per_subreddit} posts.")
            
            # Log statistics to file if logger is configured
            if self.logger:
                self.logger.info("=" * 80)
                self.logger.info(f"Topic: {topic_name} | Subreddit: r/{subreddit}")
                self.logger.info("-" * 80)
                self.logger.info(f"Posts checked: {cumulative_stats['checked']}")
                self.logger.info(f"Filtered by date: {cumulative_stats['filtered_by_date']}")
                self.logger.info(f"Filtered by score (<{post_score_threshold}): {cumulative_stats['filtered_by_score']}")
                if cumulative_stats.get('filtered_by_duplicate', 0) > 0:
                    self.logger.info(f"Filtered by duplicate: {cumulative_stats['filtered_by_duplicate']}")
                self.logger.info(f"Posts collected: {len(all_posts)} (target: {posts_per_subreddit})")
                self.logger.info(f"Comments collected: {comments_collected}")
                self.logger.info(f"Average upvotes per post: {avg_post_upvotes:.2f}")
                self.logger.info(f"Average upvotes per comment: {avg_comment_upvotes:.2f}")
                if len(all_posts) < posts_per_subreddit:
                    self.logger.info(f"WARNING: Only collected {len(all_posts)}/{posts_per_subreddit} posts")
                self.logger.info("")
            
            # Rate limiting between subreddits
            time.sleep(1)
        
        df = pd.DataFrame(all_data)
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str, output_dir: str = "data"):
        """Save collected data to CSV file."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
