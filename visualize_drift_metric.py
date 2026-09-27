import os
import json
import re
import matplotlib.pyplot as plt
from config import Config

def plot_drift_evolution():
    """
    Reads the drift metrics summary JSON (located via Config) 
    and plots the evolution of Mean Drift and Std Error (bps) 
    as the generator step increases.
    """
    cfg = Config()
    
    # Reconstruct the directory path exactly as done during evaluation
    run_name = getattr(cfg, 'eval_run_name', cfg.train.experiment_name)
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    summary_json_path = os.path.join(save_dir, "drift_metrics_summary.json")
    
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(f"Could not find summary file at: {summary_json_path}")
        
    # Ensure plots subfolder exists
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Loading drift metrics summary from: {summary_json_path}")
    with open(summary_json_path, 'r') as f:
        summary_data = json.load(f)
        
    steps = []
    mean_drifts = []
    std_errors_bps = []
    target_drift = None
    
    # Extract data and step numbers
    for ckpt_name, metrics in summary_data.items():
        # Look for digits in the checkpoint name (e.g., "generator_15000.pt" -> 15000)
        match = re.search(r'(\d+)', ckpt_name)
        if match:
            step = int(match.group(1))
            steps.append(step)
            mean_drifts.append(metrics.get("mean_drift", 0.0))
            std_errors_bps.append(metrics.get("std_error_bps", 0.0))
            
            # Capture the target drift once to plot a reference line
            if target_drift is None and "target_drift" in metrics:
                target_drift = metrics["target_drift"]
        else:
            print(f"Skipping {ckpt_name} in plot: Could not extract a step number.")
            
    if not steps:
        print("No valid numbered checkpoints found to plot.")
        return
        
    # Sort data by step number so the line plots connect correctly
    sorted_indices = sorted(range(len(steps)), key=lambda k: steps[k])
    steps = [steps[i] for i in sorted_indices]
    mean_drifts = [mean_drifts[i] for i in sorted_indices]
    std_errors_bps = [std_errors_bps[i] for i in sorted_indices]
    
    # Create a 1x2 subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Evolution of Model Drift Metrics over Training Steps", fontsize=16, y=1.05)
    
    # 1. Mean Drift Plot
    axes[0].plot(steps, mean_drifts, marker='o', linestyle='-', color='blue', linewidth=2, markersize=6)
    if target_drift is not None:
        axes[0].axhline(target_drift, color='black', linestyle='--', linewidth=2, label=f'Target ({target_drift:.4f})')
        axes[0].legend(loc='best')
    axes[0].set_title("Estimated Mean Drift", fontsize=13)
    axes[0].set_xlabel("Generator Step", fontsize=11)
    axes[0].set_ylabel("Annualized Drift", fontsize=11)
    axes[0].grid(True, alpha=0.4)
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # 2. Standard Error (bps) Plot
    axes[1].plot(steps, std_errors_bps, marker='o', linestyle='-', color='green', linewidth=2, markersize=6)
    axes[1].set_title("Standard Error of Drift", fontsize=13)
    axes[1].set_xlabel("Generator Step", fontsize=11)
    axes[1].set_ylabel("Std Error (Basis Points)", fontsize=11)
    axes[1].grid(True, alpha=0.4)
    axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "drift_metrics_evolution.pdf")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Successfully saved drift metrics evolution plot to: {plot_path}")

plot_drift_evolution()