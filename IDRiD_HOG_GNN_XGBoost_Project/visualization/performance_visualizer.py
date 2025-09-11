"""
Performance Visualization Script
Creates charts from benchmark results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_performance_charts(results_file="performance_benchmark_results.json"):
    """Create performance visualization charts"""

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Component Timing
    ax1 = axes[0, 0]
    inference_times = results['inference_times']

    components = []
    times = []

    component_mapping = {
        'data_loading': 'Data Loading',
        'hog_extraction': 'HOG Extraction',
        'graph_construction': 'Graph Construction',
        'gnn_inference': 'GNN Inference',
        'xgboost_dr': 'XGBoost DR',
        'xgboost_dme': 'XGBoost DME'
    }

    for key, name in component_mapping.items():
        if key in inference_times:
            comp = inference_times[key]
            time_val = comp.get('time', comp.get('mean_time', 0))
            if time_val > 0:
                components.append(name)
                times.append(time_val)

    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    bars = ax1.bar(components, times, color=colors)
    ax1.set_title('Component Processing Times', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{time:.4f}s', ha='center', va='bottom', fontsize=10)

    # Chart 2: Memory Usage
    ax2 = axes[0, 1]
    memory_components = []
    memory_usage = []

    for key, name in component_mapping.items():
        if key in inference_times and 'memory_gb' in inference_times[key]:
            memory_gb = inference_times[key]['memory_gb']
            if memory_gb > 0:
                memory_components.append(name)
                memory_usage.append(memory_gb * 1024)  # Convert to MB

    if memory_usage:
        bars2 = ax2.bar(memory_components, memory_usage, color='lightcoral')
        ax2.set_title('Memory Usage by Component', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Memory (MB)')
        ax2.tick_params(axis='x', rotation=45)

        for bar, mem in zip(bars2, memory_usage):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{mem:.1f}MB', ha='center', va='bottom', fontsize=10)

    # Chart 3: System Info
    ax3 = axes[1, 0]
    sys_info = results['system_info']

    # Create system info chart
    categories = ['CPU Cores', 'Logical Cores', 'Memory (GB)']
    values = [
        sys_info['cpu_count'],
        sys_info['cpu_count_logical'],
        sys_info['total_memory_gb']
    ]

    bars3 = ax3.bar(categories, values, color=['skyblue', 'lightgreen', 'gold'])
    ax3.set_title('System Specifications', fontsize=14, fontweight='bold')

    for bar, val in zip(bars3, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Chart 4: Throughput Analysis
    ax4 = axes[1, 1]

    if 'end_to_end' in inference_times:
        e2e = inference_times['end_to_end']
        throughput = e2e.get('throughput_images_per_second', 0)

        metrics = ['Images/Second', 'Images/Minute', 'Images/Hour']
        values = [throughput, throughput * 60, throughput * 3600]

        bars4 = ax4.bar(metrics, values, color=['orange', 'purple', 'green'])
        ax4.set_title('System Throughput', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Count')

        for bar, val in zip(bars4, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('performance_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("📊 Performance charts saved: performance_analysis_charts.png")


if __name__ == "__main__":
    create_performance_charts()
