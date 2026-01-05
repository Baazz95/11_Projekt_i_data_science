"""
Main script for sentiment analysis project.
Run this script to collect data, analyze sentiment, and compare subreddit groups.
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import TOPICS, REDDIT_SUBREDDITS_LEFT, REDDIT_SUBREDDITS_RIGHT, DATA_COLLECTION, SENTIMENT_MODEL, OUTPUT_DIR, RESULTS_DIR
from src.data_collection.reddit_collector import RedditCollector
from src.sentiment_analysis.analyzer import SentimentAnalyzer
from src.analysis.comparison import SubredditGroupComparator

load_dotenv()


def compile_collection_stats_log():
    """Compile left and right collection statistics into a master comparison log."""
    left_log = os.path.join(RESULTS_DIR, "collection_stats_left.log")
    right_log = os.path.join(RESULTS_DIR, "collection_stats_right.log")
    master_log = os.path.join(RESULTS_DIR, "collection_stats_master.log")
    
    if not os.path.exists(left_log) or not os.path.exists(right_log):
        print("Warning: One or both collection log files not found. Skipping master log compilation.")
        return
    
    # Parse log files into structured data
    def parse_log_file(log_path, group_label):
        """Parse a log file and return structured statistics."""
        stats_dict = {}
        current_topic = None
        current_subreddit = None
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for topic/subreddit header
                    if "Topic:" in line and "Subreddit:" in line:
                        parts = line.split("|")
                        topic_part = [p for p in parts if "Topic:" in p][0]
                        subreddit_part = [p for p in parts if "Subreddit:" in p][0]
                        current_topic = topic_part.split("Topic:")[1].strip()
                        current_subreddit = subreddit_part.split("Subreddit:")[1].strip().replace("r/", "")
                        
                        if current_topic not in stats_dict:
                            stats_dict[current_topic] = {}
                        if current_subreddit not in stats_dict[current_topic]:
                            stats_dict[current_topic][current_subreddit] = {}
                    
                    # Parse statistics lines
                    elif current_topic and current_subreddit:
                        if "Posts checked:" in line:
                            value = line.split(":", 1)[1].strip()
                            stats_dict[current_topic][current_subreddit]["posts_checked"] = int(value)
                        elif "Filtered by date:" in line:
                            value = line.split(":", 1)[1].strip()
                            stats_dict[current_topic][current_subreddit]["filtered_by_date"] = int(value)
                        elif "Filtered by score" in line:
                            # Format: "Filtered by score (<threshold>): value"
                            value = line.rsplit(":", 1)[1].strip()
                            stats_dict[current_topic][current_subreddit]["filtered_by_score"] = int(value)
                        elif "Posts collected:" in line:
                            # Format: "Posts collected: value (target: X)"
                            collected_part = line.split(":", 1)[1].strip().split("(")[0].strip()
                            stats_dict[current_topic][current_subreddit]["posts_collected"] = int(collected_part)
                        elif "Comments collected:" in line:
                            value = line.split(":", 1)[1].strip()
                            stats_dict[current_topic][current_subreddit]["comments_collected"] = int(value)
                        elif "Average upvotes per post:" in line:
                            value = line.split(":", 1)[1].strip()
                            stats_dict[current_topic][current_subreddit]["avg_upvotes"] = float(value)
                        elif "Average upvotes per comment:" in line:
                            value = line.split(":", 1)[1].strip()
                            stats_dict[current_topic][current_subreddit]["avg_comment_upvotes"] = float(value)
        except Exception as e:
            print(f"Error parsing log file {log_path}: {e}")
        
        return stats_dict
    
    # Parse both log files
    left_stats = parse_log_file(left_log, "left")
    right_stats = parse_log_file(right_log, "right")
    
    # Get all topics and subreddits
    all_topics = set(list(left_stats.keys()) + list(right_stats.keys()))
    
    # Compile master log
    with open(master_log, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("COLLECTION STATISTICS MASTER LOG - LEFT vs RIGHT COMPARISON\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for topic in sorted(all_topics):
            f.write("\n" + "=" * 120 + "\n")
            f.write(f"TOPIC: {topic.upper().replace('_', ' ')}\n")
            f.write("=" * 120 + "\n\n")
            
            # Calculate aggregate statistics for each group
            def aggregate_stats(stats_dict, topic_name):
                """Calculate aggregate statistics across all subreddits for a topic."""
                topic_data = stats_dict.get(topic_name, {})
                if not topic_data:
                    return {}
                
                agg = {
                    "posts_checked": 0,
                    "filtered_by_date": 0,
                    "filtered_by_score": 0,
                    "posts_collected": 0,
                    "comments_collected": 0,
                    "total_post_upvotes": 0,
                    "total_comment_upvotes": 0,
                    "subreddit_count": 0
                }
                
                for sub_data in topic_data.values():
                    agg["posts_checked"] += sub_data.get("posts_checked", 0)
                    agg["filtered_by_date"] += sub_data.get("filtered_by_date", 0)
                    agg["filtered_by_score"] += sub_data.get("filtered_by_score", 0)
                    agg["posts_collected"] += sub_data.get("posts_collected", 0)
                    agg["comments_collected"] += sub_data.get("comments_collected", 0)
                    # Calculate total upvotes for averages
                    posts_collected = sub_data.get("posts_collected", 0)
                    avg_post_upvotes = sub_data.get("avg_upvotes", 0)
                    if posts_collected > 0:
                        agg["total_post_upvotes"] += avg_post_upvotes * posts_collected
                    comments_collected = sub_data.get("comments_collected", 0)
                    avg_comment_upvotes = sub_data.get("avg_comment_upvotes", 0)
                    if comments_collected > 0:
                        agg["total_comment_upvotes"] += avg_comment_upvotes * comments_collected
                    agg["subreddit_count"] += 1
                
                # Calculate average upvotes
                if agg["posts_collected"] > 0:
                    agg["avg_upvotes"] = agg["total_post_upvotes"] / agg["posts_collected"]
                else:
                    agg["avg_upvotes"] = 0.0
                if agg["comments_collected"] > 0:
                    agg["avg_comment_upvotes"] = agg["total_comment_upvotes"] / agg["comments_collected"]
                else:
                    agg["avg_comment_upvotes"] = 0.0
                del agg["total_post_upvotes"]  # Remove intermediate values
                del agg["total_comment_upvotes"]
                
                return agg
            
            left_agg = aggregate_stats(left_stats, topic)
            right_agg = aggregate_stats(right_stats, topic)
            
            # Write aggregate comparison
            f.write("GROUP-LEVEL AGGREGATE COMPARISON\n")
            f.write("-" * 120 + "\n")
            metrics = [
                ("Posts checked (total)", "posts_checked", int),
                ("Filtered by date (total)", "filtered_by_date", int),
                ("Filtered by score (total)", "filtered_by_score", int),
                ("Posts collected (total)", "posts_collected", int),
                ("Comments collected (total)", "comments_collected", int),
                ("Avg upvotes/post", "avg_upvotes", float),
                ("Avg upvotes/comment", "avg_comment_upvotes", float),
                ("Subreddits in group", "subreddit_count", int)
            ]
            
            f.write(f"{'Metric':<35} {'LEFT GROUP':<30} {'RIGHT GROUP':<30} {'Difference':<20}\n")
            f.write("-" * 120 + "\n")
            
            for metric_name, metric_key, val_type in metrics:
                left_val = left_agg.get(metric_key, 0)
                right_val = right_agg.get(metric_key, 0)
                
                left_str = f"{left_val:.2f}" if val_type == float else str(left_val)
                right_str = f"{right_val:.2f}" if val_type == float else str(right_val)
                
                # Calculate difference
                diff = left_val - right_val
                if val_type == float:
                    diff_str = f"{diff:+.2f}"
                else:
                    diff_str = f"{diff:+d}"
                
                f.write(f"{metric_name:<35} {left_str:<30} {right_str:<30} {diff_str:<20}\n")
            
            f.write("\n" + "-" * 120 + "\n")
            f.write("DETAILED BREAKDOWN BY SUBREDDIT\n")
            f.write("-" * 120 + "\n\n")
            
            # Get all subreddits for this topic
            left_subs = set(left_stats.get(topic, {}).keys())
            right_subs = set(right_stats.get(topic, {}).keys())
            
            # Show left group subreddits
            if left_subs:
                f.write(f"\nLEFT GROUP SUBREDDITS:\n")
                for subreddit in sorted(left_subs):
                    sub_data = left_stats.get(topic, {}).get(subreddit, {})
                    f.write(f"  r/{subreddit}:\n")
                    f.write(f"    Posts checked: {sub_data.get('posts_checked', 'N/A')}\n")
                    f.write(f"    Filtered by date: {sub_data.get('filtered_by_date', 'N/A')}\n")
                    f.write(f"    Filtered by score: {sub_data.get('filtered_by_score', 'N/A')}\n")
                    f.write(f"    Posts collected: {sub_data.get('posts_collected', 'N/A')}\n")
                    f.write(f"    Comments collected: {sub_data.get('comments_collected', 'N/A')}\n")
                    avg_post = sub_data.get('avg_upvotes', 'N/A')
                    if avg_post != 'N/A':
                        f.write(f"    Avg upvotes/post: {avg_post:.2f}\n")
                    else:
                        f.write(f"    Avg upvotes/post: N/A\n")
                    avg_comment = sub_data.get('avg_comment_upvotes', 'N/A')
                    if avg_comment != 'N/A':
                        f.write(f"    Avg upvotes/comment: {avg_comment:.2f}\n")
                    else:
                        f.write(f"    Avg upvotes/comment: N/A\n")
                    f.write("\n")
            
            # Show right group subreddits
            if right_subs:
                f.write(f"\nRIGHT GROUP SUBREDDITS:\n")
                for subreddit in sorted(right_subs):
                    sub_data = right_stats.get(topic, {}).get(subreddit, {})
                    f.write(f"  r/{subreddit}:\n")
                    f.write(f"    Posts checked: {sub_data.get('posts_checked', 'N/A')}\n")
                    f.write(f"    Filtered by date: {sub_data.get('filtered_by_date', 'N/A')}\n")
                    f.write(f"    Filtered by score: {sub_data.get('filtered_by_score', 'N/A')}\n")
                    f.write(f"    Posts collected: {sub_data.get('posts_collected', 'N/A')}\n")
                    f.write(f"    Comments collected: {sub_data.get('comments_collected', 'N/A')}\n")
                    avg_post = sub_data.get('avg_upvotes', 'N/A')
                    if avg_post != 'N/A':
                        f.write(f"    Avg upvotes/post: {avg_post:.2f}\n")
                    else:
                        f.write(f"    Avg upvotes/post: N/A\n")
                    avg_comment = sub_data.get('avg_comment_upvotes', 'N/A')
                    if avg_comment != 'N/A':
                        f.write(f"    Avg upvotes/comment: {avg_comment:.2f}\n")
                    else:
                        f.write(f"    Avg upvotes/comment: N/A\n")
                    f.write("\n")
            
            f.write("\n")
        
        f.write("=" * 120 + "\n")
        f.write("END OF MASTER LOG\n")
        f.write("=" * 120 + "\n")
    
    print(f"Master collection statistics log compiled: {master_log}")


def collect_reddit_data(subreddits, group_label):
    """Collect data from Reddit subreddits."""
    print(f"\n{'='*80}")
    print(f"COLLECTING REDDIT DATA: {group_label.upper()} SUBREDDITS")
    print(f"{'='*80}")
    print(f"Subreddits: {', '.join(subreddits)}")
    
    # Set up log file for collection statistics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = os.path.join(RESULTS_DIR, f"collection_stats_{group_label}.log")
    # Clear the log file at the start of collection
    if os.path.exists(log_file):
        os.remove(log_file)
    
    collector = RedditCollector(log_file=log_file)
    all_data = []
    
    for topic_name, topic_config in TOPICS.items():
        print(f"\n{'='*80}")
        print(f"Topic: {topic_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        df = collector.collect_topic_data(
            topic_name=topic_name,
            fallback_keywords=topic_config.get("keywords", []),
            subreddits=subreddits,
            posts_per_subreddit=DATA_COLLECTION["reddit"]["posts_per_topic"],  # Now means posts per subreddit (not total per group)
            comments_per_post=DATA_COLLECTION["reddit"]["comments_per_post"],
            time_filter=DATA_COLLECTION["reddit"]["time_filter"],
            comment_score_threshold=DATA_COLLECTION["reddit"]["comment_score_threshold"],
            post_score_threshold=DATA_COLLECTION["reddit"].get("post_score_threshold", 0),
            start_date=DATA_COLLECTION["reddit"].get("start_date"),
            end_date=DATA_COLLECTION["reddit"].get("end_date")
        )
        
        if len(df) > 0:
            # Add group label
            df['subreddit_group'] = group_label
            all_data.append(df)
            
            # Print summary per subreddit for this topic
            print(f"\n  📈 Summary for topic '{topic_name}' ({group_label} group):")
            for subreddit in subreddits:
                subreddit_data = df[df['subreddit'] == subreddit]
                posts_count = len(subreddit_data[subreddit_data['content_type'] == 'post'])
                comments_count = len(subreddit_data[subreddit_data['content_type'] == 'comment'])
                print(f"     r/{subreddit}: {posts_count} posts, {comments_count} comments")
            
            # Save individual topic data
            filename = f"{group_label}_{topic_name}.csv"
            collector.save_data(df, filename, OUTPUT_DIR)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        collector.save_data(combined_df, f"{group_label}_all.csv", OUTPUT_DIR)
        return combined_df
    else:
        print(f"No data collected for {group_label} group.")
        return pd.DataFrame()


def analyze_sentiment(df: pd.DataFrame, group_label: str):
    """Analyze sentiment for a dataset."""
    print(f"\n{'='*80}")
    print(f"ANALYZING SENTIMENT: {group_label.upper()}")
    print(f"{'='*80}")
    
    if len(df) == 0:
        print(f"No data to analyze for {group_label}")
        return df
    
    analyzer = SentimentAnalyzer(model_type=SENTIMENT_MODEL)
    df_analyzed = analyzer.analyze_dataframe(df, text_column="full_text")
    
    # Save analyzed data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{group_label}_analyzed.csv")
    df_analyzed.to_csv(filepath, index=False)
    print(f"Analyzed data saved to {filepath}")
    
    # Print summary
    summary = analyzer.get_sentiment_summary(df_analyzed)
    print("\nSentiment Summary:")
    print(f"  Total texts: {summary['total_texts']}")
    print(f"  Mean sentiment score: {summary['mean_sentiment_score']:.3f}")
    print(f"  Positive: {summary['positive_percentage']:.1f}%")
    print(f"  Negative: {summary['negative_percentage']:.1f}%")
    print(f"  Neutral: {summary['neutral_percentage']:.1f}%")
    
    return df_analyzed


def compare_subreddit_groups(left_df: pd.DataFrame, right_df: pd.DataFrame):
    """Compare sentiment between subreddit groups with comprehensive extremity analysis."""
    print(f"\n{'='*80}")
    print("COMPARING SUBREDDIT GROUPS: EXTREMITY & POLARIZATION ANALYSIS")
    print(f"{'='*80}")
    
    comparator = SubredditGroupComparator(results_dir=RESULTS_DIR)
    
    # Basic comparison
    comparison_df = comparator.compare_by_topic(left_df, right_df)
    
    # Save comparison data
    filepath = os.path.join(RESULTS_DIR, "subreddit_group_comparison.csv")
    comparison_df.to_csv(filepath, index=False)
    print(f"Comparison data saved to {filepath}")
    
    # Comprehensive analysis with statistical tests
    print("\nPerforming comprehensive extremity and polarization analysis...")
    analysis_results = comparator.comprehensive_analysis(left_df, right_df)
    
    # Save detailed analysis results
    import json
    analysis_filepath = os.path.join(RESULTS_DIR, "comprehensive_analysis.json")
    # Convert to JSON-serializable format
    analysis_for_json = {}
    for key, value in analysis_results.items():
        if isinstance(value, dict):
            analysis_for_json[key] = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    analysis_for_json[key][k] = {str(kk): float(vv) if isinstance(vv, (int, float, np.integer, np.floating)) else str(vv) 
                                                 for kk, vv in v.items() if not isinstance(vv, (pd.DataFrame, pd.Series))}
                else:
                    analysis_for_json[key][k] = str(v) if not isinstance(v, (int, float)) else v
    
    with open(analysis_filepath, 'w') as f:
        json.dump(analysis_for_json, f, indent=2)
    print(f"Detailed analysis saved to {analysis_filepath}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    print("  1. Basic comparison charts...")
    comparator.plot_sentiment_comparison(comparison_df, save=True)
    
    print("  2. Distribution and extremity analysis...")
    comparator.plot_distribution_comparison(left_df, right_df, save=True)
    
    print("  3. Extremity category breakdown...")
    comparator.plot_extremity_categories(left_df, right_df, save=True)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = comparator.generate_comprehensive_report(left_df, right_df, analysis_results, save=True)
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    left_overall = analysis_results['overall']['left']
    right_overall = analysis_results['overall']['right']
    stats_overall = analysis_results['statistical_tests']['overall']
    
    print(f"\nExtremity Comparison:")
    print(f"  Left-leaning subreddits mean extremity: {left_overall['mean_extremity']:.3f}")
    print(f"  Right-leaning subreddits mean extremity: {right_overall['mean_extremity']:.3f}")
    if stats_overall['extremity_ttest']['significant']:
        print(f"  → Statistically significant difference (p={stats_overall['extremity_ttest']['p_value']:.4f})")
        print(f"  → Effect size: {stats_overall['effect_size_interpretation']}")
    else:
        print(f"  → No significant difference (p={stats_overall['extremity_ttest']['p_value']:.4f})")
    
    print(f"\nPolarization (Variance) Comparison:")
    print(f"  Left-leaning std dev: {left_overall['sentiment_std']:.3f}")
    print(f"  Right-leaning std dev: {right_overall['sentiment_std']:.3f}")
    if stats_overall['levene_variance']['significant']:
        print(f"  → Groups differ in polarization (p={stats_overall['levene_variance']['p_value']:.4f})")
    else:
        print(f"  → Similar polarization levels (p={stats_overall['levene_variance']['p_value']:.4f})")
    
    print(f"\nHighly Extreme Content (>0.7):")
    print(f"  Left-leaning: {left_overall['pct_highly_extreme']:.1f}%")
    print(f"  Right-leaning: {right_overall['pct_highly_extreme']:.1f}%")
    
    return comparison_df, analysis_results


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS PROJECT")
    print("Comparing Left-Leaning vs Right-Leaning Reddit Subreddits")
    print("="*80)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Collect data from both subreddit groups
    print("\n[STEP 1] Data Collection")
    left_df = collect_reddit_data(REDDIT_SUBREDDITS_LEFT, "left")
    right_df = collect_reddit_data(REDDIT_SUBREDDITS_RIGHT, "right")
    
    # Step 2: Compile master collection statistics log
    print("\n[STEP 2] Compiling Master Collection Statistics Log")
    compile_collection_stats_log()
    
    # Step 3: Analyze sentiment
    print("\n[STEP 3] Sentiment Analysis")
    left_analyzed = analyze_sentiment(left_df, "left")
    right_analyzed = analyze_sentiment(right_df, "right")
    
    # Step 4: Compare subreddit groups with extremity analysis
    if len(left_analyzed) > 0 and len(right_analyzed) > 0:
        print("\n[STEP 4] Subreddit Group Comparison: Extremity & Polarization Analysis")
        comparison_df, analysis_results = compare_subreddit_groups(left_analyzed, right_analyzed)
    else:
        print("\nSkipping comparison - insufficient data from one or both subreddit groups")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved in:")
    print(f"  - Data: {OUTPUT_DIR}/")
    print(f"  - Results: {RESULTS_DIR}/")
    print(f"  - Collection statistics logs:")
    print(f"    * Individual: {RESULTS_DIR}/collection_stats_left.log, {RESULTS_DIR}/collection_stats_right.log")
    print(f"    * Master comparison: {RESULTS_DIR}/collection_stats_master.log")


if __name__ == "__main__":
    main()

