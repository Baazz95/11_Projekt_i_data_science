"""
Separate Analysis Script for Posts vs Comments
Loads existing analyzed data and generates separate visualizations for posts and comments.
This script does NOT recollect data - it uses existing analyzed CSV files.
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import OUTPUT_DIR, RESULTS_DIR
from src.analysis.comparison import SubredditGroupComparator


def load_analyzed_data(group_label: str) -> pd.DataFrame:
    """Load existing analyzed data from CSV file."""
    filepath = os.path.join(OUTPUT_DIR, f"{group_label}_analyzed.csv")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        print("Please run main.py first to generate analyzed data.")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def filter_by_content_type(df: pd.DataFrame, content_type: str) -> pd.DataFrame:
    """Filter dataframe by content type (post or comment)."""
    filtered = df[df['content_type'] == content_type].copy()
    print(f"  Filtered to {len(filtered)} {content_type}s")
    return filtered


def run_separate_analysis(left_df: pd.DataFrame, right_df: pd.DataFrame, 
                          content_type: str, suffix: str = ""):
    """Run complete analysis for posts or comments separately."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {content_type.upper()}{suffix}")
    print(f"{'='*80}")
    
    # Filter by content type
    left_filtered = filter_by_content_type(left_df, content_type)
    right_filtered = filter_by_content_type(right_df, content_type)
    
    if len(left_filtered) == 0 or len(right_filtered) == 0:
        print(f"⚠️  Insufficient data for {content_type} analysis. Skipping...")
        return None, None
    
    # Create separate results directory for this content type
    content_results_dir = os.path.join(RESULTS_DIR, content_type)
    os.makedirs(content_results_dir, exist_ok=True)
    
    comparator = SubredditGroupComparator(results_dir=content_results_dir)
    
    # Basic comparison
    comparison_df = comparator.compare_by_topic(left_filtered, right_filtered)
    
    # Save comparison data
    filepath = os.path.join(content_results_dir, f"subreddit_group_comparison_{content_type}.csv")
    comparison_df.to_csv(filepath, index=False)
    print(f"Comparison data saved to {filepath}")
    
    # Comprehensive analysis with statistical tests
    print(f"\nPerforming comprehensive extremity and polarization analysis for {content_type}...")
    analysis_results = comparator.comprehensive_analysis(left_filtered, right_filtered)
    
    # Save detailed analysis results
    analysis_filepath = os.path.join(content_results_dir, f"comprehensive_analysis_{content_type}.json")
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
    
    # Generate visualizations (each saved separately)
    print(f"\nGenerating visualizations for {content_type}...")
    print("  1. Basic comparison charts...")
    comparator.plot_sentiment_comparison(comparison_df, save=True)
    
    print("  2. Distribution and extremity analysis...")
    comparator.plot_distribution_comparison(left_filtered, right_filtered, save=True)
    
    print("  3. Extremity category breakdown...")
    comparator.plot_extremity_categories(left_filtered, right_filtered, save=True)
    
    # Generate comprehensive report
    print(f"\nGenerating comprehensive report for {content_type}...")
    report = comparator.generate_comprehensive_report(left_filtered, right_filtered, analysis_results, save=True)
    
    # Print key findings
    print("\n" + "="*80)
    print(f"KEY FINDINGS SUMMARY - {content_type.upper()}")
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
    
    print(f"\nSample Sizes:")
    print(f"  Left-leaning {content_type}s: {stats_overall['sample_sizes']['left']}")
    print(f"  Right-leaning {content_type}s: {stats_overall['sample_sizes']['right']}")
    
    return comparison_df, analysis_results


def main():
    """Main execution function for separate post/comment analysis."""
    print("\n" + "="*80)
    print("SEPARATE ANALYSIS: POSTS vs COMMENTS")
    print("Using existing analyzed data - NO data collection")
    print("="*80)
    
    # Check if analyzed data exists
    left_file = os.path.join(OUTPUT_DIR, "left_analyzed.csv")
    right_file = os.path.join(OUTPUT_DIR, "right_analyzed.csv")
    
    if not os.path.exists(left_file) or not os.path.exists(right_file):
        print("\n❌ Error: Analyzed data files not found!")
        print(f"   Looking for: {left_file}")
        print(f"   Looking for: {right_file}")
        print("\n   Please run main.py first to generate analyzed data.")
        return
    
    # Load existing analyzed data
    print("\n[STEP 1] Loading analyzed data...")
    left_df = load_analyzed_data("left")
    right_df = load_analyzed_data("right")
    
    if len(left_df) == 0 or len(right_df) == 0:
        print("❌ Error: Could not load analyzed data. Please run main.py first.")
        return
    
    # Show data breakdown
    print(f"\nData breakdown:")
    print(f"  Left group - Posts: {len(left_df[left_df['content_type'] == 'post'])}, Comments: {len(left_df[left_df['content_type'] == 'comment'])}")
    print(f"  Right group - Posts: {len(right_df[right_df['content_type'] == 'post'])}, Comments: {len(right_df[right_df['content_type'] == 'comment'])}")
    
    # Analyze posts separately
    print("\n" + "="*80)
    print("[STEP 2] ANALYZING POSTS")
    print("="*80)
    posts_comparison, posts_results = run_separate_analysis(left_df, right_df, "post")
    
    # Analyze comments separately
    print("\n" + "="*80)
    print("[STEP 3] ANALYZING COMMENTS")
    print("="*80)
    comments_comparison, comments_results = run_separate_analysis(left_df, right_df, "comment")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved in:")
    print(f"  - Posts analysis: {RESULTS_DIR}/post/")
    print(f"  - Comments analysis: {RESULTS_DIR}/comment/")
    print(f"\nEach content type has:")
    print(f"  - Individual visualizations (01-09_*.png)")
    print(f"  - Comparison CSV files")
    print(f"  - Comprehensive analysis JSON")
    print(f"  - Detailed report")


if __name__ == "__main__":
    main()

