"""Script to format statistical test results into a readable table."""
import json
import ast
import re
from pathlib import Path


def parse_string_dict(s):
    """Parse string representation of dictionary with numpy types."""
    # Replace np.float64, np.True_, np.False_ with Python equivalents
    s = re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)
    s = re.sub(r'np\.True_', 'True', s)
    s = re.sub(r'np\.False_', 'False', s)
    s = re.sub(r'np\.int64\(([^)]+)\)', r'\1', s)
    return ast.literal_eval(s)


def format_p_value(p):
    """Format p-value for display."""
    if p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"


def format_cohens_d(d):
    """Format Cohen's d with interpretation."""
    abs_d = abs(d)
    if abs_d < 0.2:
        size = "negligible"
    elif abs_d < 0.5:
        size = "small"
    elif abs_d < 0.8:
        size = "medium"
    else:
        size = "large"
    return f"{d:.3f} ({size})"


def create_statistical_tables():
    """Create formatted tables from comprehensive analysis JSON."""
    json_path = Path("results/comprehensive_analysis.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_lines = []
    output_lines.append("# Statistical Test Results Summary")
    output_lines.append("")
    output_lines.append("## Overall Comparison (All Topics Combined)")
    output_lines.append("")
    
    # Overall statistics
    overall = data['statistical_tests']['overall']
    overall_stats = parse_string_dict(overall['sample_sizes'])
    
    overall_left = data['overall']['left']
    overall_right = data['overall']['right']
    
    output_lines.append("### Sample Sizes")
    output_lines.append(f"- **Left-leaning group:** {int(overall_left['count']):,} items")
    output_lines.append(f"- **Right-leaning group:** {int(overall_right['count']):,} items")
    output_lines.append("")
    
    output_lines.append("### Mean Sentiment Scores")
    output_lines.append(f"- **Left-leaning:** {overall_left['mean_sentiment']:.3f}")
    output_lines.append(f"- **Right-leaning:** {overall_right['mean_sentiment']:.3f}")
    output_lines.append(f"- **Difference:** {overall_left['mean_sentiment'] - overall_right['mean_sentiment']:.3f}")
    output_lines.append("")
    
    # Parse overall statistical tests
    mw_overall = parse_string_dict(overall['mann_whitney_u'])
    levene_overall = parse_string_dict(overall['levene_variance'])
    ks_overall = parse_string_dict(overall['kolmogorov_smirnov'])
    ext_overall = parse_string_dict(overall['extremity_ttest'])
    
    output_lines.append("### Statistical Tests")
    output_lines.append("")
    output_lines.append("| Test | p-value | Significant? | Interpretation |")
    output_lines.append("|------|---------|--------------|----------------|")
    output_lines.append(f"| **Mann-Whitney U** (Sentiment Difference) | {format_p_value(mw_overall['p_value'])} | {'✓ Yes' if mw_overall['significant'] else '✗ No'} | {mw_overall['interpretation']} |")
    output_lines.append(f"| **Levene's Test** (Variance/Polarization) | {format_p_value(levene_overall['p_value'])} | {'✓ Yes' if levene_overall['significant'] else '✗ No'} | {levene_overall['interpretation']} |")
    output_lines.append(f"| **Kolmogorov-Smirnov** (Distribution Shape) | {format_p_value(ks_overall['p_value'])} | {'✓ Yes' if ks_overall['significant'] else '✗ No'} | {ks_overall['interpretation']} |")
    output_lines.append(f"| **T-Test** (Extremity Difference) | {format_p_value(ext_overall['p_value'])} | {'✓ Yes' if ext_overall['significant'] else '✗ No'} | {ext_overall['interpretation']} |")
    output_lines.append("")
    
    output_lines.append("### Effect Sizes")
    output_lines.append("")
    output_lines.append("| Measure | Cohen's d | Interpretation |")
    output_lines.append("|---------|-----------|----------------|")
    output_lines.append(f"| **Sentiment Difference** | {format_cohens_d(overall['cohens_d_sentiment'])} | Effect size for sentiment difference |")
    output_lines.append(f"| **Extremity Difference** | {format_cohens_d(overall['cohens_d_extremity'])} | Effect size for extremity difference |")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    
    # Per-topic tables
    output_lines.append("## Per-Topic Comparison")
    output_lines.append("")
    
    topics_map = {
        'climate_change': 'Climate Change',
        'ukraine': 'Ukraine',
        'cost_of_living': 'Cost of Living',
        'immigration': 'Immigration',
        'ai_bubble': 'AI Bubble'
    }
    
    # Main per-topic table
    output_lines.append("### Summary Table: Statistical Tests by Topic")
    output_lines.append("")
    output_lines.append("| Topic | Sample Size (L/R) | Mann-Whitney U p-value | Significant? | Levene p-value | KS p-value | Extremity T-Test p-value | Cohen's d (Sentiment) |")
    output_lines.append("|-------|-------------------|------------------------|--------------|----------------|------------|-------------------------|---------------------|")
    
    for topic_key, topic_display in topics_map.items():
        if topic_key not in data['by_topic']:
            continue
            
        topic_data = data['by_topic'][topic_key]
        tests = parse_string_dict(topic_data['statistical_tests'])
        left_data = parse_string_dict(topic_data['left'])
        right_data = parse_string_dict(topic_data['right'])
        
        sample_sizes = tests['sample_sizes']
        mw = tests['mann_whitney_u']
        levene = tests['levene_variance']
        ks = tests['kolmogorov_smirnov']
        ext = tests['extremity_ttest']
        
        sig_mark = "✓" if mw['significant'] else "✗"
        cohens_d = tests['cohens_d_sentiment']
        
        output_lines.append(
            f"| {topic_display} | {sample_sizes['left']}/{sample_sizes['right']} | "
            f"{format_p_value(mw['p_value'])} | {sig_mark} | "
            f"{format_p_value(levene['p_value'])} | "
            f"{format_p_value(ks['p_value'])} | "
            f"{format_p_value(ext['p_value'])} | "
            f"{cohens_d:.3f} |"
        )
    
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    
    # Detailed per-topic breakdown
    output_lines.append("### Detailed Per-Topic Results")
    output_lines.append("")
    
    for topic_key, topic_display in sorted(topics_map.items()):
        if topic_key not in data['by_topic']:
            continue
            
        topic_data = data['by_topic'][topic_key]
        tests = parse_string_dict(topic_data['statistical_tests'])
        left_data = parse_string_dict(topic_data['left'])
        right_data = parse_string_dict(topic_data['right'])
        
        sample_sizes = tests['sample_sizes']
        mw = tests['mann_whitney_u']
        levene = tests['levene_variance']
        ks = tests['kolmogorov_smirnov']
        ext = tests['extremity_ttest']
        
        output_lines.append(f"#### {topic_display}")
        output_lines.append("")
        output_lines.append(f"**Sample Sizes:** Left: {sample_sizes['left']:,} | Right: {sample_sizes['right']:,}")
        output_lines.append("")
        output_lines.append(f"**Mean Sentiment:** Left: {left_data['mean_sentiment']:.3f} | Right: {right_data['mean_sentiment']:.3f}")
        output_lines.append("")
        output_lines.append("| Test | p-value | Significant? | Interpretation |")
        output_lines.append("|------|---------|--------------|----------------|")
        output_lines.append(f"| Mann-Whitney U | {format_p_value(mw['p_value'])} | {'✓ Yes' if mw['significant'] else '✗ No'} | {mw['interpretation']} |")
        output_lines.append(f"| Levene's Test | {format_p_value(levene['p_value'])} | {'✓ Yes' if levene['significant'] else '✗ No'} | {levene['interpretation']} |")
        output_lines.append(f"| Kolmogorov-Smirnov | {format_p_value(ks['p_value'])} | {'✓ Yes' if ks['significant'] else '✗ No'} | {ks['interpretation']} |")
        output_lines.append(f"| T-Test (Extremity) | {format_p_value(ext['p_value'])} | {'✓ Yes' if ext['significant'] else '✗ No'} | {ext['interpretation']} |")
        output_lines.append("")
        output_lines.append(f"**Effect Sizes:** Cohen's d (Sentiment) = {format_cohens_d(tests['cohens_d_sentiment'])} | Cohen's d (Extremity) = {format_cohens_d(tests['cohens_d_extremity'])}")
        output_lines.append("")
    
    # Write to file
    output_path = Path("results/statistical_tests_summary.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Statistical test summary table created: {output_path}")
    print(f"\nSummary:")
    print(f"- Overall: Mann-Whitney U p = {format_p_value(mw_overall['p_value'])} ({'Significant' if mw_overall['significant'] else 'Not significant'})")
    print(f"- Topics with significant differences: ", end="")
    sig_topics = []
    for topic_key, topic_display in topics_map.items():
        if topic_key in data['by_topic']:
            tests = parse_string_dict(data['by_topic'][topic_key]['statistical_tests'])
            if tests['mann_whitney_u']['significant']:
                sig_topics.append(topic_display)
    print(", ".join(sig_topics) if sig_topics else "None")


if __name__ == "__main__":
    create_statistical_tables()
