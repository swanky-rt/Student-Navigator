"""
LINE PLOT: Backdoor Attack Metrics Visualization
Shows ASR vs CA trends across different sample sizes
"""

import os
import json
import matplotlib.pyplot as plt

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

def create_line_plot(sorted_data):
    """Create line plot comparing ASR and CA trends"""
    if not sorted_data:
        print("No data to plot")
        return
    
    # Extract data
    record_counts = [num for num, _ in sorted_data]
    asr_values = [data['asr'] for _, data in sorted_data]
    ca_values = [data['ca'] for _, data in sorted_data]
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Plot ASR and CA
    plt.plot(record_counts, asr_values, 'o-', linewidth=3, label='ASR (Attack Success)',
             color='#e74c3c', markersize=10)
    plt.plot(record_counts, ca_values, 's-', linewidth=3, label='CA (Clean Accuracy)',
             color='#3498db', markersize=10)
    
    # Customize
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Poisoned Records', size=12, weight='bold')
    plt.ylabel('Percentage (%)', size=12, weight='bold')
    plt.ylim(0, 100)
    plt.xticks(record_counts, rotation=0)
    plt.yticks([0, 20, 40, 60, 80, 100])
    
    # Add value labels
    for i, (asr, ca) in enumerate(zip(asr_values, ca_values)):
        plt.annotate(f'{asr:.1f}%', (record_counts[i], asr), textcoords="offset points", 
                    xytext=(0,10), ha='center', color='#e74c3c')
        plt.annotate(f'{ca:.1f}%', (record_counts[i], ca), textcoords="offset points", 
                    xytext=(0,-15), ha='center', color='#3498db')
    
    # Legend and title
    plt.legend(fontsize=12, loc='center right')
    plt.title('Backdoor Attack Performance: ASR vs CA\nEffect of Poisoned Sample Size',
              size=14, weight='bold', pad=20)
    
    # Save
    output_path = "./assignment-8/outputs/backdoor_line_plot.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Line plot saved: {output_path}")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BACKDOOR METRICS LINE PLOT")
    print("="*70)
    
    print("\n[LOADING RESULTS]")
    results = load_results()
    
    if results:
        print("[PARSING DATA]")
        sorted_data = parse_results(results)
        
        if sorted_data:
            print(f"Found {len(sorted_data)} data points\n")
            print("[GENERATING LINE PLOT]")
            create_line_plot(sorted_data)
            print("\n" + "="*70 + "\n")
        else:
            print("Could not parse data")
    else:
        print("Could not load results")