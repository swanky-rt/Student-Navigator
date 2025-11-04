"""
SPIDER GRAPH: Backdoor Attack Metrics Visualization
Shows ASR vs CA across different sample sizes
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def load_results(summary_path="./assignment-8/outputs/poison_records_summary.json"):
    """Load results from summary JSON"""
    if not os.path.exists(summary_path):
        print(f"Results file not found: {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def parse_results(results):
    """Parse results into sorted data"""
    if not results:
        return None
    
    data = {}
    for key, val in results.items():
        if 'records' in key:
            num = int(key.replace('records', ''))
            data[num] = {
                'asr': val.get('asr', 0) * 100,
                'ca': val.get('ca', 0) * 100,
            }
    
    return sorted(data.items())


def create_spider_chart(sorted_data):
    """Create spider chart for ASR and CA"""
    
    if not sorted_data:
        print("No data to plot")
        return
    
    # Extract data
    record_counts = [str(num) for num, _ in sorted_data]
    asr_values = [data['asr'] for _, data in sorted_data]
    ca_values = [data['ca'] for _, data in sorted_data]
    
    num_vars = len(record_counts)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    
    # Close the circle
    asr_values += asr_values[:1]
    ca_values += ca_values[:1]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Plot ASR and CA
    ax.plot(angles, asr_values, 'o-', linewidth=3, label='ASR (Attack Success)',
            color='#e74c3c', markersize=10)
    ax.fill(angles, asr_values, alpha=0.25, color='#e74c3c')
    
    ax.plot(angles, ca_values, 's-', linewidth=3, label='CA (Clean Accuracy)',
            color='#3498db', markersize=10)
    ax.fill(angles, ca_values, alpha=0.25, color='#3498db')
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(record_counts, size=12, weight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=10)
    ax.set_rlabel_position(0)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)
    plt.title('Backdoor Attack Metrics: ASR vs CA\nAcross Different Sample Sizes',
              size=14, weight='bold', pad=20)
    
    # Save
    output_path = "./assignment-8/outputs/backdoor_spider_chart.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Spider chart saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BACKDOOR METRICS SPIDER CHART")
    print("="*70)
    
    print("\n[LOADING RESULTS]")
    results = load_results()
    
    if results:
        print("[PARSING DATA]")
        sorted_data = parse_results(results)
        
        if sorted_data:
            print(f"Found {len(sorted_data)} data points\n")
            print("[GENERATING CHART]")
            create_spider_chart(sorted_data)
            print("\n" + "="*70 + "\n")
        else:
            print("Could not parse data")
    else:
        print("Could not load results")
