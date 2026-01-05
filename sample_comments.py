"""
Sample Comments by Extremity Script
Extracts sample comments from each extremity bracket for qualitative analysis.
Organized by topic and subreddit group.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import OUTPUT_DIR, RESULTS_DIR, TOPICS


def load_analyzed_data(group_label: str) -> pd.DataFrame:
    """Load existing analyzed data from CSV file."""
    filepath = os.path.join(OUTPUT_DIR, f"{group_label}_analyzed.csv")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    return df


def get_extremity_bracket(extremity: float) -> str:
    """Categorize extremity score into brackets."""
    if abs(extremity) < 0.01:
        return "Neutral (~0)"
    elif 0.01 <= abs(extremity) <= 0.4:
        return "Low (0.01-0.4)"
    elif 0.4 < abs(extremity) <= 0.7:
        return "Moderate (0.4-0.7)"
    else:  # > 0.7
        return "High (0.7+)"


def sample_comments_by_extremity(df: pd.DataFrame, n_per_bracket: int = 5) -> pd.DataFrame:
    """
    Sample comments from each extremity bracket.
    
    Args:
        df: DataFrame with comment data (must have sentiment_score and content_type='comment')
        n_per_bracket: Number of comments to sample from each bracket
    
    Returns:
        DataFrame with sampled comments
    """
    # Filter to comments only
    comments_df = df[df['content_type'] == 'comment'].copy()
    
    if len(comments_df) == 0:
        return pd.DataFrame()
    
    # Calculate extremity (absolute value of sentiment score)
    comments_df['extremity'] = comments_df['sentiment_score'].abs()
    
    # Add extremity bracket
    comments_df['extremity_bracket'] = comments_df['extremity'].apply(get_extremity_bracket)
    
    # Sample from each bracket
    sampled_comments = []
    
    # Define brackets with their ranges
    brackets = [
        ("Neutral (~0)", lambda x: abs(x) < 0.01),
        ("Low (0.01-0.4)", lambda x: 0.01 <= abs(x) <= 0.4),
        ("Moderate (0.4-0.7)", lambda x: 0.4 < abs(x) <= 0.7),
        ("High (0.7+)", lambda x: abs(x) > 0.7)
    ]
    
    for bracket_name, bracket_filter in brackets:
        bracket_comments = comments_df[comments_df['sentiment_score'].apply(bracket_filter)]
        
        if len(bracket_comments) > 0:
            # Sample randomly, but use more for neutral bracket
            n_sample = n_per_bracket if bracket_name != "Neutral (~0)" else n_per_bracket
            n_sample = min(n_sample, len(bracket_comments))
            
            sampled = bracket_comments.sample(n=n_sample, random_state=42)
            sampled_comments.append(sampled)
    
    if sampled_comments:
        result_df = pd.concat(sampled_comments, ignore_index=True)
        # Sort by topic, then extremity bracket, then extremity (descending)
        bracket_order = {"Neutral (~0)": 0, "Low (0.01-0.4)": 1, "Moderate (0.4-0.7)": 2, "High (0.7+)": 3}
        result_df['bracket_order'] = result_df['extremity_bracket'].map(bracket_order)
        result_df = result_df.sort_values(['topic', 'bracket_order', 'extremity'], ascending=[True, True, False])
        result_df = result_df.drop('bracket_order', axis=1)
        return result_df
    else:
        return pd.DataFrame()


def format_comment_sample(df: pd.DataFrame, output_file: str):
    """Format and save comment samples to a readable text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("SAMPLE COMMENTS BY EXTREMITY\n")
        f.write("Qualitative Examples for Each Extremity Bracket\n")
        f.write("=" * 100 + "\n\n")
        
        # Group by topic and subreddit_group
        for topic in sorted(df['topic'].unique()):
            topic_df = df[df['topic'] == topic]
            
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"TOPIC: {topic.upper().replace('_', ' ')}\n")
            f.write("=" * 100 + "\n\n")
            
            for group in sorted(topic_df['subreddit_group'].unique()):
                group_df = topic_df[topic_df['subreddit_group'] == group]
                
                f.write(f"\n{'-' * 100}\n")
                f.write(f"{group.upper()}-LEANING SUBREDDITS\n")
                f.write(f"{'-' * 100}\n\n")
                
                # Group by extremity bracket
                for bracket in ["Neutral (~0)", "Low (0.01-0.4)", "Moderate (0.4-0.7)", "High (0.7+)"]:
                    bracket_df = group_df[group_df['extremity_bracket'] == bracket]
                    
                    if len(bracket_df) > 0:
                        f.write(f"\n{bracket.upper()} EXTREMITY:\n")
                        f.write("-" * 100 + "\n")
                        
                        for idx, row in bracket_df.iterrows():
                            f.write(f"\nComment #{idx + 1}:\n")
                            f.write(f"  Extremity Score: {row['extremity']:.4f} (Sentiment: {row['sentiment_score']:.4f})\n")
                            f.write(f"  From Post ID: {row.get('post_id', 'N/A')}\n")
                            f.write(f"  Subreddit: r/{row.get('subreddit', 'N/A')}\n")
                            if 'title' in row and pd.notna(row.get('title')):
                                # Try to get post title from the data - might need to join
                                f.write(f"  Post Title: {row.get('title', 'N/A')[:100]}...\n")
                            f.write(f"  Comment Text: {row.get('text', row.get('full_text', 'N/A'))[:500]}\n")
                            if len(str(row.get('text', row.get('full_text', '')))) > 500:
                                f.write("    ... (truncated)\n")
                            f.write("\n")
                        
                        f.write("\n")
            
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF SAMPLES\n")
        f.write("=" * 100 + "\n")


def get_post_titles(df_comments: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    """Join comment data with post titles using post_id."""
    # Get posts from the full dataframe
    posts_df = df_all[df_all['content_type'] == 'post'][['post_id', 'title', 'subreddit']].copy()
    posts_df = posts_df.drop_duplicates(subset=['post_id'])
    
    # Merge to get post titles
    df_with_titles = df_comments.merge(
        posts_df,
        on='post_id',
        how='left',
        suffixes=('', '_post')
    )
    
    # Use the post subreddit if comment subreddit is missing
    if 'subreddit_post' in df_with_titles.columns:
        df_with_titles['subreddit'] = df_with_titles['subreddit'].fillna(df_with_titles['subreddit_post'])
    
    return df_with_titles


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SAMPLE COMMENTS BY EXTREMITY")
    print("=" * 80)
    
    # Load analyzed data
    print("\n[STEP 1] Loading analyzed data...")
    left_df = load_analyzed_data("left")
    right_df = load_analyzed_data("right")
    
    if len(left_df) == 0 or len(right_df) == 0:
        print("❌ Error: Could not load analyzed data. Please run main.py first.")
        return
    
    # Combine for processing
    all_df = pd.concat([left_df, right_df], ignore_index=True)
    
    print(f"Loaded {len(all_df)} total records")
    print(f"  Comments: {len(all_df[all_df['content_type'] == 'comment'])}")
    print(f"  Posts: {len(all_df[all_df['content_type'] == 'post'])}")
    
    # Sample comments for each topic and group
    print("\n[STEP 2] Sampling comments by extremity bracket...")
    
    all_samples = []
    
    for topic in sorted(all_df['topic'].unique()):
        topic_df = all_df[all_df['topic'] == topic]
        
        for group in sorted(topic_df['subreddit_group'].unique()):
            group_df = topic_df[topic_df['subreddit_group'] == group]
            
            print(f"  Processing {topic} - {group} group...")
            samples = sample_comments_by_extremity(group_df, n_per_bracket=5)
            
            if len(samples) > 0:
                all_samples.append(samples)
    
    if len(all_samples) == 0:
        print("❌ No comments found to sample.")
        return
    
    # Combine all samples
    samples_df = pd.concat(all_samples, ignore_index=True)
    
    # Get post titles by merging with post data
    print("\n[STEP 3] Joining with post information...")
    samples_df = get_post_titles(samples_df, all_df)
    
    # Create output directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save as CSV
    csv_file = os.path.join(RESULTS_DIR, "sample_comments_by_extremity.csv")
    
    # Select relevant columns for CSV
    output_columns = [
        'topic', 'subreddit_group', 'subreddit', 'extremity_bracket',
        'extremity', 'sentiment_score', 'sentiment_label',
        'post_id', 'title', 'text',
        'score', 'comment_id'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in samples_df.columns]
    samples_df[available_columns].to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n✅ CSV file saved: {csv_file}")
    
    # Save as formatted text file
    txt_file = os.path.join(RESULTS_DIR, "sample_comments_by_extremity.txt")
    format_comment_sample(samples_df, txt_file)
    print(f"✅ Formatted text file saved: {txt_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print(f"\nTotal sampled comments: {len(samples_df)}")
    print(f"\nBreakdown by extremity bracket:")
    for bracket in ["Neutral (~0)", "Low (0.01-0.4)", "Moderate (0.4-0.7)", "High (0.7+)"]:
        count = len(samples_df[samples_df['extremity_bracket'] == bracket])
        print(f"  {bracket}: {count} comments")
    
    print(f"\nBreakdown by group:")
    for group in sorted(samples_df['subreddit_group'].unique()):
        count = len(samples_df[samples_df['subreddit_group'] == group])
        print(f"  {group}: {count} comments")
    
    print(f"\nOutput files:")
    print(f"  - CSV: {csv_file}")
    print(f"  - Formatted text: {txt_file}")


if __name__ == "__main__":
    main()
