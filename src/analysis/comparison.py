import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
import os


class SubredditGroupComparator:
    """Compare sentiment across subreddit groups."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize comparator.
        
        Args:
            results_dir: Directory to save comparison results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def _calculate_bimodality_coefficient(self, data: pd.Series) -> float:
        """
        Calculate bimodality coefficient.
        Values > 0.555 suggest bimodal distribution (polarization).
        
        Args:
            data: Series of sentiment scores
        
        Returns:
            Bimodality coefficient
        """
        n = len(data)
        if n < 3:
            return 0.0
        
        skew = data.skew()
        kurt = data.kurtosis()
        
        # Bimodality coefficient formula
        bc = (skew**2 + 1) / (kurt + 3 * ((n-1)**2) / ((n-2)*(n-3)))
        return bc
    
    def analyze_extremity(self, df: pd.DataFrame, group_label: str) -> Dict:
        """
        Analyze sentiment extremity and polarization metrics.
        
        Args:
            df: DataFrame with sentiment scores
            group_label: Subreddit group label (left/right) for labeling
        
        Returns:
            Dictionary with extremity metrics
        """
        sentiment_scores = df['sentiment_score']
        extremity_scores = sentiment_scores.abs()
        
        return {
            'subreddit_group': group_label,
            'count': len(df),
            # Basic sentiment stats
            'mean_sentiment': sentiment_scores.mean(),
            'median_sentiment': sentiment_scores.median(),
            'sentiment_std': sentiment_scores.std(),
            'sentiment_variance': sentiment_scores.var(),
            'sentiment_range': sentiment_scores.max() - sentiment_scores.min(),
            # Extremity metrics (how far from neutral)
            'mean_extremity': extremity_scores.mean(),
            'median_extremity': extremity_scores.median(),
            'extremity_std': extremity_scores.std(),
            'pct_highly_extreme': (extremity_scores > 0.7).mean() * 100,
            'pct_moderately_extreme': ((extremity_scores > 0.4) & (extremity_scores <= 0.7)).mean() * 100,
            'pct_neutral': (extremity_scores <= 0.4).mean() * 100,
            # Distribution shape (polarization indicators)
            'skewness': sentiment_scores.skew(),
            'kurtosis': sentiment_scores.kurtosis(),
            'bimodality_coefficient': self._calculate_bimodality_coefficient(sentiment_scores),
            # Sentiment categories
            'positive_pct': (df['sentiment_label'] == 'positive').mean() * 100,
            'negative_pct': (df['sentiment_label'] == 'negative').mean() * 100,
            'neutral_pct': (df['sentiment_label'] == 'neutral').mean() * 100,
        }
    
    def statistical_comparison(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                               topic: str = "overall") -> Dict:
        """
        Perform statistical tests comparing subreddit groups.
        
        Args:
            left_df: Left-leaning subreddit data
            right_df: Right-leaning subreddit data
            topic: Topic name for reporting
        
        Returns:
            Dictionary with test results
        """
        left_sentiment = left_df['sentiment_score']
        right_sentiment = right_df['sentiment_score']
        left_extremity = left_sentiment.abs()
        right_extremity = right_sentiment.abs()
        
        # Test 1: Mann-Whitney U (non-parametric test for central tendency)
        mw_stat, mw_p = stats.mannwhitneyu(left_sentiment, right_sentiment, alternative='two-sided')
        
        # Test 2: Levene's test (variance equality - tests polarization difference)
        levene_stat, levene_p = stats.levene(left_sentiment, right_sentiment)
        
        # Test 3: Kolmogorov-Smirnov (distribution shape difference)
        ks_stat, ks_p = stats.ks_2samp(left_sentiment, right_sentiment)
        
        # Test 4: T-test for extremity scores
        extremity_t, extremity_p = stats.ttest_ind(left_extremity, right_extremity)
        
        # Effect sizes
        # Cohen's d for sentiment difference
        pooled_std = np.sqrt((left_sentiment.std()**2 + right_sentiment.std()**2) / 2)
        cohens_d = (left_sentiment.mean() - right_sentiment.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Cohen's d for extremity difference
        pooled_std_ext = np.sqrt((left_extremity.std()**2 + right_extremity.std()**2) / 2)
        cohens_d_extremity = (left_extremity.mean() - right_extremity.mean()) / pooled_std_ext if pooled_std_ext > 0 else 0
        
        return {
            'topic': topic,
            'sample_sizes': {'left': len(left_df), 'right': len(right_df)},
            # Sentiment central tendency test
            'mann_whitney_u': {
                'statistic': mw_stat,
                'p_value': mw_p,
                'significant': mw_p < 0.05,
                'interpretation': 'Groups differ in sentiment' if mw_p < 0.05 else 'No significant sentiment difference'
            },
            # Variance test (polarization)
            'levene_variance': {
                'statistic': levene_stat,
                'p_value': levene_p,
                'significant': levene_p < 0.05,
                'interpretation': 'Groups differ in polarization' if levene_p < 0.05 else 'Similar polarization levels'
            },
            # Distribution shape test
            'kolmogorov_smirnov': {
                'statistic': ks_stat,
                'p_value': ks_p,
                'significant': ks_p < 0.05,
                'interpretation': 'Different sentiment distributions' if ks_p < 0.05 else 'Similar distributions'
            },
            # Extremity test
            'extremity_ttest': {
                'statistic': extremity_t,
                'p_value': extremity_p,
                'significant': extremity_p < 0.05,
                'interpretation': 'Groups differ in extremity' if extremity_p < 0.05 else 'Similar extremity levels'
            },
            # Effect sizes
            'cohens_d_sentiment': cohens_d,
            'cohens_d_extremity': cohens_d_extremity,
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d_extremity))
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def compare_by_topic(self, left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare sentiment between subreddit groups for each topic.
        
        Args:
            left_df: DataFrame with left-leaning subreddit sentiment data
            right_df: DataFrame with right-leaning subreddit sentiment data
        
        Returns:
            DataFrame with comparison statistics
        """
        comparison_results = []
        
        # Get unique topics
        topics = set(left_df['topic'].unique()) | set(right_df['topic'].unique())
        
        for topic in topics:
            left_topic = left_df[left_df['topic'] == topic]
            right_topic = right_df[right_df['topic'] == topic]
            
            if len(left_topic) == 0 or len(right_topic) == 0:
                continue
            
            # Comparison with extremity metrics
            left_metrics = self.analyze_extremity(left_topic, "left")
            left_metrics['topic'] = topic
            comparison_results.append(left_metrics)
            
            right_metrics = self.analyze_extremity(right_topic, "right")
            right_metrics['topic'] = topic
            comparison_results.append(right_metrics)
        
        comparison_df = pd.DataFrame(comparison_results)
        return comparison_df
    
    def comprehensive_analysis(self, left_df: pd.DataFrame, right_df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive cross-group analysis with statistical testing.
        
        Args:
            left_df: Left-leaning subreddit sentiment data
            right_df: Right-leaning subreddit sentiment data
        
        Returns:
            Dictionary with all analysis results
        """
        results = {
            'overall': {},
            'by_topic': {},
            'statistical_tests': {}
        }
        
        # Overall analysis
        results['overall']['left'] = self.analyze_extremity(left_df, 'left')
        results['overall']['right'] = self.analyze_extremity(right_df, 'right')
        results['statistical_tests']['overall'] = self.statistical_comparison(left_df, right_df, 'overall')
        
        # Per-topic analysis
        topics = set(left_df['topic'].unique()) | set(right_df['topic'].unique())
        
        for topic in topics:
            left_topic = left_df[left_df['topic'] == topic]
            right_topic = right_df[right_df['topic'] == topic]
            
            if len(left_topic) > 0 and len(right_topic) > 0:
                results['by_topic'][topic] = {
                    'left': self.analyze_extremity(left_topic, 'left'),
                    'right': self.analyze_extremity(right_topic, 'right'),
                    'statistical_tests': self.statistical_comparison(left_topic, right_topic, topic)
                }
        
        return results
    
    def plot_sentiment_comparison(self, comparison_df: pd.DataFrame, save: bool = True):
        """
        Create visualization comparing sentiment across platforms.
        Saves each plot separately for easy use in documents.
        
        Args:
            comparison_df: DataFrame with comparison statistics
            save: Whether to save the plot
        """
        saved_files = []
        
        # 1. Mean sentiment by topic and subreddit group
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        pivot_mean = comparison_df.pivot(index='topic', columns='subreddit_group', values='mean_sentiment')
        pivot_mean.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Mean Sentiment Score by Topic and Subreddit Group', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Topic', fontsize=12)
        ax1.set_ylabel('Mean Sentiment Score', fontsize=12)
        ax1.legend(title='Subreddit Group')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '01_mean_sentiment_by_topic.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 1 saved to {filepath}")
        plt.close(fig1)
        
        # 2. Sentiment distribution (positive/negative/neutral)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        topics = comparison_df['topic'].unique()
        x = np.arange(len(topics))
        width = 0.35
        
        left_data = comparison_df[comparison_df['subreddit_group'] == 'left']
        right_data = comparison_df[comparison_df['subreddit_group'] == 'right']
        
        left_pos = [left_data[left_data['topic'] == t]['positive_pct'].values[0] if len(left_data[left_data['topic'] == t]) > 0 else 0 for t in topics]
        right_pos = [right_data[right_data['topic'] == t]['positive_pct'].values[0] if len(right_data[right_data['topic'] == t]) > 0 else 0 for t in topics]
        
        ax2.bar(x - width/2, left_pos, width, label='Left-Leaning Positive', color='#95E1D3', alpha=0.8)
        ax2.bar(x + width/2, right_pos, width, label='Right-Leaning Positive', color='#F38181', alpha=0.8)
        ax2.set_title('Positive Sentiment Percentage by Topic', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Topic', fontsize=12)
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(topics, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '02_positive_sentiment_by_topic.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 2 saved to {filepath}")
        plt.close(fig2)
        
        # 3. Volume comparison
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        pivot_count = comparison_df.pivot(index='topic', columns='subreddit_group', values='count')
        pivot_count.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('Discussion Volume by Topic and Subreddit Group', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Topic', fontsize=12)
        ax3.set_ylabel('Number of Items', fontsize=12)
        ax3.legend(title='Subreddit Group')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '03_discussion_volume_by_topic.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 3 saved to {filepath}")
        plt.close(fig3)
        
        # 4. Sentiment distribution pie chart (overall)
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        overall_sentiment = {
            'Positive': comparison_df['positive_pct'].mean(),
            'Negative': comparison_df['negative_pct'].mean(),
            'Neutral': comparison_df['neutral_pct'].mean()
        }
        colors = ['#95E1D3', '#F38181', '#AA96DA']
        ax4.pie(overall_sentiment.values(), labels=overall_sentiment.keys(), autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax4.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '04_overall_sentiment_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 4 saved to {filepath}")
        plt.close(fig4)
        
        if save:
            print(f"\nAll sentiment comparison plots saved separately to {self.results_dir}/")
            print(f"Files: {len(saved_files)} plots saved")
    
    def plot_distribution_comparison(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                                     save: bool = True):
        """
        Create detailed distribution visualizations showing extremity and polarization.
        Saves each plot separately for easy use in documents.
        
        Args:
            left_df: Left-leaning subreddit sentiment data
            right_df: Right-leaning subreddit sentiment data
            save: Whether to save the plot
        """
        saved_files = []
        
        # 1. Distribution comparison (violin plots)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        data_for_violin = pd.DataFrame({
            'Sentiment': list(left_df['sentiment_score']) + list(right_df['sentiment_score']),
            'Group': ['Left-Leaning']*len(left_df) + ['Right-Leaning']*len(right_df)
        })
        sns.violinplot(data=data_for_violin, x='Group', y='Sentiment', ax=ax1, 
                      palette={'Left-Leaning': '#FF6B6B', 'Right-Leaning': '#4ECDC4'})
        ax1.set_title('Sentiment Distribution by Subreddit Group (Violin Plot)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Sentiment Score', fontsize=12)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '05_sentiment_distribution_violin.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 5 saved to {filepath}")
        plt.close(fig1)
        
        # 2. Extremity comparison (absolute values)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        left_extremity = left_df['sentiment_score'].abs()
        right_extremity = right_df['sentiment_score'].abs()
        
        ax2.hist(left_extremity, bins=30, alpha=0.6, label='Left-Leaning', color='#FF6B6B', density=True)
        ax2.hist(right_extremity, bins=30, alpha=0.6, label='Right-Leaning', color='#4ECDC4', density=True)
        ax2.axvline(x=left_extremity.mean(), color='#FF6B6B', linestyle='--', 
                   linewidth=2, label=f'Left Mean: {left_extremity.mean():.3f}')
        ax2.axvline(x=right_extremity.mean(), color='#4ECDC4', linestyle='--', 
                   linewidth=2, label=f'Right Mean: {right_extremity.mean():.3f}')
        ax2.set_title('Extremity Distribution (|Sentiment|)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Extremity Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '06_extremity_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 6 saved to {filepath}")
        plt.close(fig2)
        
        # 3. Sentiment distribution by topic
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        topics = left_df['topic'].unique()
        
        left_extremity_by_topic = [left_df[left_df['topic']==t]['sentiment_score'].abs().mean() 
                                     for t in topics]
        right_extremity_by_topic = [right_df[right_df['topic']==t]['sentiment_score'].abs().mean() 
                                      for t in topics if t in right_df['topic'].unique()]
        
        x = np.arange(len(topics))
        width = 0.35
        
        ax3.bar(x - width/2, left_extremity_by_topic, width, label='Left-Leaning', 
               color='#FF6B6B', alpha=0.8)
        ax3.bar(x + width/2, right_extremity_by_topic, width, label='Right-Leaning', 
               color='#4ECDC4', alpha=0.8)
        ax3.set_title('Mean Extremity by Topic', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Topic', fontsize=12)
        ax3.set_ylabel('Mean Extremity Score', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels([t.replace('_', ' ').title() for t in topics], 
                           rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '07_mean_extremity_by_topic.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 7 saved to {filepath}")
        plt.close(fig3)
        
        # 4. Polarization indicators (variance comparison)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        left_std_by_topic = [left_df[left_df['topic']==t]['sentiment_score'].std() 
                               for t in topics]
        right_std_by_topic = [right_df[right_df['topic']==t]['sentiment_score'].std() 
                               for t in topics if t in right_df['topic'].unique()]
        
        ax4.bar(x - width/2, left_std_by_topic, width, label='Left-Leaning', 
               color='#FF6B6B', alpha=0.8)
        ax4.bar(x + width/2, right_std_by_topic, width, label='Right-Leaning', 
               color='#4ECDC4', alpha=0.8)
        ax4.set_title('Sentiment Variance by Topic (Polarization Indicator)', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Topic', fontsize=12)
        ax4.set_ylabel('Standard Deviation', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels([t.replace('_', ' ').title() for t in topics], 
                           rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '08_sentiment_variance_by_topic.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Plot 8 saved to {filepath}")
        plt.close(fig4)
        
        if save:
            print(f"\nAll distribution comparison plots saved separately to {self.results_dir}/")
            print(f"Files: {len(saved_files)} plots saved")
    
    def plot_extremity_categories(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                                  save: bool = True):
        """
        Plot extremity categories showing % highly extreme, moderate, and neutral.
        
        Args:
            left_df: Left-leaning subreddit sentiment data
            right_df: Right-leaning subreddit sentiment data
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate extremity categories
        def categorize_extremity(df):
            extremity = df['sentiment_score'].abs()
            return {
                'Highly Extreme\n(>0.7)': (extremity > 0.7).mean() * 100,
                'Moderately Extreme\n(0.4-0.7)': ((extremity > 0.4) & (extremity <= 0.7)).mean() * 100,
                'Neutral\n(≤0.4)': (extremity <= 0.4).mean() * 100
            }
        
        left_cats = categorize_extremity(left_df)
        right_cats = categorize_extremity(right_df)
        
        categories = list(left_cats.keys())
        left_values = list(left_cats.values())
        right_values = list(right_cats.values())
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, left_values, width, label='Left-Leaning', 
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, right_values, width, label='Right-Leaning', 
                      color='#4ECDC4', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Extremity Category Distribution by Subreddit Group', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Percentage of Posts (%)', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(left_values + right_values) * 1.15)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_dir, '09_extremity_categories.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot 9 saved to {filepath}")
        
        plt.close()
    
    def generate_comprehensive_report(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                                     analysis_results: Dict, save: bool = True) -> str:
        """
        Generate comprehensive report including extremity and statistical analysis.
        
        Args:
            left_df: Left-leaning subreddit sentiment data
            right_df: Right-leaning subreddit sentiment data
            analysis_results: Results from comprehensive_analysis()
            save: Whether to save the report
        
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SENTIMENT EXTREMITY & POLARIZATION ANALYSIS REPORT")
        report_lines.append("Subreddit Group Comparison: Left-Leaning vs Right-Leaning")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("Research Focus: Measuring sentiment extremity and polarization as potential")
        report_lines.append("indicators of algorithmic amplification and discourse radicalization.")
        report_lines.append("")
        
        # Overall Analysis
        report_lines.append("\n" + "=" * 80)
        report_lines.append("OVERALL SUBREDDIT GROUP COMPARISON")
        report_lines.append("=" * 80)
        
        left_overall = analysis_results['overall']['left']
        right_overall = analysis_results['overall']['right']
        stats_overall = analysis_results['statistical_tests']['overall']
        
        report_lines.append(f"\nLEFT-LEANING SUBREDDITS (n={left_overall['count']}):")
        report_lines.append(f"  Mean Sentiment: {left_overall['mean_sentiment']:.3f}")
        report_lines.append(f"  Sentiment Std Dev: {left_overall['sentiment_std']:.3f} (polarization indicator)")
        report_lines.append(f"  Mean Extremity: {left_overall['mean_extremity']:.3f}")
        report_lines.append(f"  Highly Extreme Posts (>0.7): {left_overall['pct_highly_extreme']:.1f}%")
        report_lines.append(f"  Bimodality Coefficient: {left_overall['bimodality_coefficient']:.3f}")
        if left_overall['bimodality_coefficient'] > 0.555:
            report_lines.append(f"    → Suggests POLARIZED distribution")
        
        report_lines.append(f"\nRIGHT-LEANING SUBREDDITS (n={right_overall['count']}):")
        report_lines.append(f"  Mean Sentiment: {right_overall['mean_sentiment']:.3f}")
        report_lines.append(f"  Sentiment Std Dev: {right_overall['sentiment_std']:.3f} (polarization indicator)")
        report_lines.append(f"  Mean Extremity: {right_overall['mean_extremity']:.3f}")
        report_lines.append(f"  Highly Extreme Posts (>0.7): {right_overall['pct_highly_extreme']:.1f}%")
        report_lines.append(f"  Bimodality Coefficient: {right_overall['bimodality_coefficient']:.3f}")
        if right_overall['bimodality_coefficient'] > 0.555:
            report_lines.append(f"    → Suggests POLARIZED distribution")
        
        report_lines.append("\n" + "-" * 80)
        report_lines.append("STATISTICAL SIGNIFICANCE TESTS")
        report_lines.append("-" * 80)
        
        report_lines.append(f"\n1. Mann-Whitney U Test (Sentiment Difference):")
        report_lines.append(f"   p-value: {stats_overall['mann_whitney_u']['p_value']:.4f}")
        report_lines.append(f"   Result: {stats_overall['mann_whitney_u']['interpretation']}")
        
        report_lines.append(f"\n2. Levene's Test (Variance/Polarization Difference):")
        report_lines.append(f"   p-value: {stats_overall['levene_variance']['p_value']:.4f}")
        report_lines.append(f"   Result: {stats_overall['levene_variance']['interpretation']}")
        
        report_lines.append(f"\n3. Extremity T-Test:")
        report_lines.append(f"   p-value: {stats_overall['extremity_ttest']['p_value']:.4f}")
        report_lines.append(f"   Cohen's d: {stats_overall['cohens_d_extremity']:.3f} ({stats_overall['effect_size_interpretation']} effect)")
        report_lines.append(f"   Result: {stats_overall['extremity_ttest']['interpretation']}")
        
        # Per-topic analysis
        if 'by_topic' in analysis_results and analysis_results['by_topic']:
            report_lines.append("\n\n" + "=" * 80)
            report_lines.append("TOPIC-SPECIFIC ANALYSIS")
            report_lines.append("=" * 80)
            
            for topic, topic_data in analysis_results['by_topic'].items():
                report_lines.append(f"\n{'-' * 80}")
                report_lines.append(f"Topic: {topic.upper().replace('_', ' ')}")
                report_lines.append(f"{'-' * 80}")
                
                left_topic = topic_data['left']
                right_topic = topic_data['right']
                stats_topic = topic_data['statistical_tests']
                
                report_lines.append(f"\nLeft-Leaning (n={left_topic['count']}):")
                report_lines.append(f"  Sentiment: {left_topic['mean_sentiment']:.3f} ± {left_topic['sentiment_std']:.3f}")
                report_lines.append(f"  Extremity: {left_topic['mean_extremity']:.3f}")
                report_lines.append(f"  Highly Extreme: {left_topic['pct_highly_extreme']:.1f}%")
                report_lines.append(f"  Distribution: {left_topic['positive_pct']:.1f}% pos, {left_topic['negative_pct']:.1f}% neg, {left_topic['neutral_pct']:.1f}% neutral")
                
                report_lines.append(f"\nRight-Leaning (n={right_topic['count']}):")
                report_lines.append(f"  Sentiment: {right_topic['mean_sentiment']:.3f} ± {right_topic['sentiment_std']:.3f}")
                report_lines.append(f"  Extremity: {right_topic['mean_extremity']:.3f}")
                report_lines.append(f"  Highly Extreme: {right_topic['pct_highly_extreme']:.1f}%")
                report_lines.append(f"  Distribution: {right_topic['positive_pct']:.1f}% pos, {right_topic['negative_pct']:.1f}% neg, {right_topic['neutral_pct']:.1f}% neutral")
                
                report_lines.append(f"\nStatistical Tests:")
                report_lines.append(f"  Extremity difference p-value: {stats_topic['extremity_ttest']['p_value']:.4f}")
                report_lines.append(f"  Effect size (Cohen's d): {stats_topic['cohens_d_extremity']:.3f}")
                report_lines.append(f"  Variance difference p-value: {stats_topic['levene_variance']['p_value']:.4f}")
                
                if stats_topic['extremity_ttest']['significant']:
                    extremity_diff = left_topic['mean_extremity'] - right_topic['mean_extremity']
                    if extremity_diff > 0:
                        report_lines.append(f"  → Left-leaning subreddits show SIGNIFICANTLY MORE extremity")
                    else:
                        report_lines.append(f"  → Right-leaning subreddits show SIGNIFICANTLY MORE extremity")
        
        # Interpretation section
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("INTERPRETATION & IMPLICATIONS")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("This analysis examines sentiment extremity and polarization as indicators")
        report_lines.append("consistent with theories of algorithmic amplification and discourse radicalization.")
        report_lines.append("")
        report_lines.append("Key findings to consider:")
        
        # Determine which group shows more extremity
        if left_overall['mean_extremity'] > right_overall['mean_extremity']:
            diff_pct = ((left_overall['mean_extremity'] / right_overall['mean_extremity']) - 1) * 100
            report_lines.append(f"- Left-leaning subreddits exhibit {diff_pct:.1f}% higher mean extremity than right-leaning")
        else:
            diff_pct = ((right_overall['mean_extremity'] / left_overall['mean_extremity']) - 1) * 100
            report_lines.append(f"- Right-leaning subreddits exhibit {diff_pct:.1f}% higher mean extremity than left-leaning")
        
        if left_overall['sentiment_std'] > right_overall['sentiment_std']:
            report_lines.append(f"- Left-leaning subreddits show greater variance (more polarized discussions)")
        else:
            report_lines.append(f"- Right-leaning subreddits show greater variance (more polarized discussions)")
        
        report_lines.append("")
        report_lines.append("LIMITATIONS:")
        report_lines.append("- Observational data cannot establish causation")
        report_lines.append("- Selection bias may affect subreddit group differences")
        report_lines.append("- Subreddit selection may not represent full political spectrum")
        report_lines.append("- Sentiment model accuracy on political topics requires validation")
        
        report_text = "\n".join(report_lines)
        
        if save:
            filepath = os.path.join(self.results_dir, 'extremity_analysis_report.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Comprehensive report saved to {filepath}")
        
        return report_text
    
    def generate_report(self, comparison_df: pd.DataFrame, save: bool = True) -> str:
        """
        Generate a basic text report of the comparison (legacy method).
        Use generate_comprehensive_report() for full analysis.
        
        Args:
            comparison_df: DataFrame with comparison statistics
            save: Whether to save the report
        
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SENTIMENT ANALYSIS COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        topics = comparison_df['topic'].unique()
        
        for topic in topics:
            report_lines.append(f"\nTopic: {topic.upper().replace('_', ' ')}")
            report_lines.append("-" * 80)
            
            topic_data = comparison_df[comparison_df['topic'] == topic]
            
            for group in ['left', 'right']:
                group_data = topic_data[topic_data['subreddit_group'] == group]
                if len(group_data) > 0:
                    data = group_data.iloc[0]
                    report_lines.append(f"\n{group.upper()}-LEANING:")
                    report_lines.append(f"  Total Posts: {data['count']}")
                    report_lines.append(f"  Mean Sentiment Score: {data['mean_sentiment']:.3f}")
                    report_lines.append(f"  Mean Extremity: {data.get('mean_extremity', 0):.3f}")
                    report_lines.append(f"  Positive: {data['positive_pct']:.1f}%")
                    report_lines.append(f"  Negative: {data['negative_pct']:.1f}%")
                    report_lines.append(f"  Neutral: {data['neutral_pct']:.1f}%")
            
            # Group difference
            left_data = topic_data[topic_data['subreddit_group'] == 'left']
            right_data = topic_data[topic_data['subreddit_group'] == 'right']
            
            if len(left_data) > 0 and len(right_data) > 0:
                diff = left_data.iloc[0]['mean_sentiment'] - right_data.iloc[0]['mean_sentiment']
                report_lines.append(f"\nDifference (Left - Right): {diff:.3f}")
                if abs(diff) > 0.1:
                    report_lines.append(f"  → Notable difference in sentiment between groups")
        
        report_text = "\n".join(report_lines)
        
        if save:
            filepath = os.path.join(self.results_dir, 'comparison_report.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to {filepath}")
        
        return report_text

