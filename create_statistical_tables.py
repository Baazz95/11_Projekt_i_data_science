"""Script to create matplotlib table visualizations from statistical test results."""
import json
import ast
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style to match existing visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def parse_string_dict(s):
    """Parse string representation of dictionary with numpy types."""
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


def create_overall_table(data):
    """Create overall statistical test results table."""
    overall = data['statistical_tests']['overall']
    overall_left = data['overall']['left']
    overall_right = data['overall']['right']
    
    # Parse test results
    mw = parse_string_dict(overall['mann_whitney_u'])
    levene = parse_string_dict(overall['levene_variance'])
    ks = parse_string_dict(overall['kolmogorov_smirnov'])
    ext = parse_string_dict(overall['extremity_ttest'])
    
    # Prepare table data
    headers = ['Test', 'p-value', 'Significant?', 'Interpretation']
    
    table_data = [
        ['Mann-Whitney U\n(Sentiment)', format_p_value(mw['p_value']), 
         'Yes' if mw['significant'] else 'No', mw['interpretation']],
        ["Levene's Test\n(Variance)", format_p_value(levene['p_value']), 
         'Yes' if levene['significant'] else 'No', levene['interpretation']],
        ['Kolmogorov-Smirnov\n(Distribution)', format_p_value(ks['p_value']), 
         'Yes' if ks['significant'] else 'No', ks['interpretation']],
        ['T-Test\n(Extremity)', format_p_value(ext['p_value']), 
         'Yes' if ext['significant'] else 'No', ext['interpretation']]
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='left',
                     loc='center', bbox=[0, 0, 1, 1])
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        table[(0, i)].set_height(0.15)
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_height(0.12)
            
            # Color code significance column
            if j == 2:  # Significant? column
                if table_data[i-1][2] == 'Yes':
                    cell.set_facecolor('#C5E0B4')  # Light green
                else:
                    cell.set_facecolor('#F4B084')  # Light orange
            
            # Left align text
            cell.set_text_props(fontsize=10)
    
    # Add title and summary info
    title_text = 'Overall Statistical Test Results'
    summary_text = (f"Sample Sizes: Left={int(overall_left['count']):,}, "
                   f"Right={int(overall_right['count']):,} | "
                   f"Mean Sentiment: Left={overall_left['mean_sentiment']:.3f}, "
                   f"Right={overall_right['mean_sentiment']:.3f} | "
                   f"Cohen's d (Sentiment)={overall['cohens_d_sentiment']:.3f}, "
                   f"Cohen's d (Extremity)={overall['cohens_d_extremity']:.3f}")
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.95)
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    output_path = Path("results/statistical_tests_overall_table.png")
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created: {output_path}")
    return output_path


def create_per_topic_table(data):
    """Create per-topic statistical test results table."""
    topics_map = {
        'climate_change': 'Climate Change',
        'ukraine': 'Ukraine',
        'cost_of_living': 'Cost of Living',
        'immigration': 'Immigration',
        'ai_bubble': 'AI Bubble'
    }
    
    # Prepare table data
    headers = ['Topic', 'Sample\n(L/R)', 'Mann-Whitney U\np-value', 'Sig?', 
               'Levene\np-value', 'KS\np-value', 'Extremity T\np-value', 
               "Cohen's d\n(Sentiment)"]
    
    table_data = []
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
        
        sig_mark = 'Yes' if mw['significant'] else 'No'
        cohens_d = tests['cohens_d_sentiment']
        
        table_data.append([
            topic_display,
            f"{sample_sizes['left']}/{sample_sizes['right']}",
            format_p_value(mw['p_value']),
            sig_mark,
            format_p_value(levene['p_value']),
            format_p_value(ks['p_value']),
            format_p_value(ext['p_value']),
            f"{cohens_d:.3f}"
        ])
    
    # Create figure - larger to accommodate more columns
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
        table[(0, i)].set_height(0.15)
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_height(0.12)
            cell.set_text_props(fontsize=9)
            
            # Color code significance column
            if j == 3:  # Sig? column
                if table_data[i-1][3] == 'Yes':
                    cell.set_facecolor('#C5E0B4')  # Light green
                else:
                    cell.set_facecolor('#F4B084')  # Light orange
    
    # Add title
    fig.suptitle('Statistical Test Results by Topic', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_path = Path("results/statistical_tests_per_topic_table.png")
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created: {output_path}")
    return output_path


def create_detailed_per_topic_tables(data):
    """Create detailed tables for each topic individually."""
    topics_map = {
        'climate_change': 'Climate Change',
        'ukraine': 'Ukraine',
        'cost_of_living': 'Cost of Living',
        'immigration': 'Immigration',
        'ai_bubble': 'AI Bubble'
    }
    
    output_paths = []
    
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
        
        # Prepare table data
        headers = ['Test', 'p-value', 'Significant?', 'Interpretation']
        
        table_data = [
            ['Mann-Whitney U\n(Sentiment)', format_p_value(mw['p_value']), 
             'Yes' if mw['significant'] else 'No', mw['interpretation']],
            ["Levene's Test\n(Variance)", format_p_value(levene['p_value']), 
             'Yes' if levene['significant'] else 'No', levene['interpretation']],
            ['Kolmogorov-Smirnov\n(Distribution)', format_p_value(ks['p_value']), 
             'Yes' if ks['significant'] else 'No', ks['interpretation']],
            ['T-Test\n(Extremity)', format_p_value(ext['p_value']), 
             'Yes' if ext['significant'] else 'No', ext['interpretation']]
        ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='left',
                         loc='center', bbox=[0, 0, 1, 1])
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
            table[(0, i)].set_height(0.15)
        
        # Style cells
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_height(0.12)
                
                # Color code significance column
                if j == 2:  # Significant? column
                    if table_data[i-1][2] == 'Yes':
                        cell.set_facecolor('#C5E0B4')  # Light green
                    else:
                        cell.set_facecolor('#F4B084')  # Light orange
                
                cell.set_text_props(fontsize=10)
        
        # Add title and summary
        title_text = f'Statistical Test Results: {topic_display}'
        summary_text = (f"Sample Sizes: Left={sample_sizes['left']:,}, "
                       f"Right={sample_sizes['right']:,} | "
                       f"Mean Sentiment: Left={left_data['mean_sentiment']:.3f}, "
                       f"Right={right_data['mean_sentiment']:.3f} | "
                       f"Cohen's d (Sentiment)={tests['cohens_d_sentiment']:.3f}, "
                       f"Cohen's d (Extremity)={tests['cohens_d_extremity']:.3f}")
        
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.95)
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        # Safe filename
        safe_topic = topic_key.replace('_', '_')
        output_path = Path(f"results/statistical_tests_{safe_topic}_table.png")
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        output_paths.append(output_path)
        print(f"Created: {output_path}")
    
    return output_paths


def main():
    """Main function to create all table visualizations."""
    json_path = Path("results/comprehensive_analysis.json")
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Creating statistical test table visualizations...")
    print()
    
    # Create overall table
    create_overall_table(data)
    
    # Create per-topic summary table
    create_per_topic_table(data)
    
    # Create detailed per-topic tables
    create_detailed_per_topic_tables(data)
    
    print()
    print("All table visualizations created successfully!")


if __name__ == "__main__":
    main()
