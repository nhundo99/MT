import os
import json
import re
import matplotlib.pyplot as plt
from config import Config

def plot_metrics_evolution():
    """
    Reads the distributional metrics summary JSON (located via Config) 
    and plots the evolution of standard metrics (CVM, ACF, CCF, ES) 
    as well as Jump Behavior statistics as the generator step increases.
    """
    cfg = Config()
    
    # Reconstruct the directory path exactly as done during evaluation
    run_name = getattr(cfg, 'eval_run_name', cfg.train.experiment_name)
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    summary_json_path = os.path.join(save_dir, "distributional_metrics_summary.json")
    
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(f"Could not find summary file at: {summary_json_path}")
        
    # Ensure plots subfolder exists
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Loading metrics summary from: {summary_json_path}")
    with open(summary_json_path, 'r') as f:
        summary_data = json.load(f)
        
    steps = []
    
    # Standard metrics
    cvm_vals = []
    acf_vals = []
    ccf_vals = []
    es_vals = []
    
    # Jump metrics (Generated vs Real)
    gen_jumps = []
    real_jumps = []
    gen_means = []
    real_means = []
    gen_stds = []
    real_stds = []
    
    # Variables to hold the threshold info for the plot title
    threshold_multiplier = None
    threshold_value = None
    
    # Extract data and step numbers
    for ckpt_name, metrics in summary_data.items():
        # Look for digits in the checkpoint name (e.g., "generator_15000.pt" -> 15000)
        match = re.search(r'(\d+)', ckpt_name)
        if match:
            step = int(match.group(1))
            steps.append(step)
            
            # Extract Standard Metrics
            cvm_vals.append(metrics.get("CVM", 0))
            acf_vals.append(metrics.get("ACF", 0))
            ccf_vals.append(metrics.get("CCF", 0))
            es_vals.append(metrics.get("ES", 0))
            
            # Extract Jump Metrics (defaults to 0 if not found for older runs)
            j_metrics = metrics.get("jump_metrics", {})
            if j_metrics:
                gen_jumps.append(j_metrics.get("gen_jumps_per_year", 0))
                real_jumps.append(j_metrics.get("real_jumps_per_year", 0))
                gen_means.append(j_metrics.get("gen_mean", 0))
                real_means.append(j_metrics.get("real_mean", 0))
                gen_stds.append(j_metrics.get("gen_std", 0))
                real_stds.append(j_metrics.get("real_std", 0))
                
                # Capture the threshold parameters (they should be identical across steps, 
                # so capturing the last one or overwriting it is perfectly fine)
                threshold_multiplier = j_metrics.get("threshold_multiplier")
                threshold_value = j_metrics.get("threshold_value")
            else:
                # Fallback if jump metrics are missing in older checkpoints
                gen_jumps.append(0)
                real_jumps.append(0)
                gen_means.append(0)
                real_means.append(0)
                gen_stds.append(0)
                real_stds.append(0)
            
        else:
            print(f"Skipping {ckpt_name} in plot: Could not extract a step number.")
            
    if not steps:
        print("No valid numbered checkpoints found to plot.")
        return
        
    # Sort all data lists by step number so the line plots connect correctly
    sorted_indices = sorted(range(len(steps)), key=lambda k: steps[k])
    steps = [steps[i] for i in sorted_indices]
    
    cvm_vals = [cvm_vals[i] for i in sorted_indices]
    acf_vals = [acf_vals[i] for i in sorted_indices]
    ccf_vals = [ccf_vals[i] for i in sorted_indices]
    es_vals = [es_vals[i] for i in sorted_indices]
    
    gen_jumps = [gen_jumps[i] for i in sorted_indices]
    real_jumps = [real_jumps[i] for i in sorted_indices]
    gen_means = [gen_means[i] for i in sorted_indices]
    real_means = [real_means[i] for i in sorted_indices]
    gen_stds = [gen_stds[i] for i in sorted_indices]
    real_stds = [real_stds[i] for i in sorted_indices]
    
    # ---------------------------------------------------------
    # FIGURE 1: Standard Distributional Metrics
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle("Evolution of Distributional Metrics over Training Steps", fontsize=16, y=0.95)
    
    def plot_subplot(ax, x, y, title, ylabel, color):
        ax.plot(x, y, marker='o', linestyle='-', color=color, linewidth=2, markersize=6)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Generator Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    plot_subplot(axes1[0, 0], steps, cvm_vals, "Cramér-von Mises (CVM) Distance", "CVM", "blue")
    plot_subplot(axes1[0, 1], steps, acf_vals, "Autocorrelation Difference (ACF)", "ACF Diff", "green")
    plot_subplot(axes1[1, 0], steps, ccf_vals, "Cross-Correlation Difference (CCF)", "CCF Diff", "orange")
    plot_subplot(axes1[1, 1], steps, es_vals, "Expected Shortfall Difference (ES)", "ES Diff", "red")
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path_standard = os.path.join(plots_dir, "metrics_evolution.pdf")
    plt.savefig(plot_path_standard, format='pdf', bbox_inches='tight')
    plt.close(fig1)
    
    print(f"Successfully saved standard metrics evolution plot to: {plot_path_standard}")

    # ---------------------------------------------------------
    # FIGURE 2: Jump Behavior Statistics
    # ---------------------------------------------------------
    # Only plot if jump metrics were actually found in the JSON
    if any(real_jumps):
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        
        # Build the dynamic title with the threshold information
        jump_title = "Evolution of Jump Statistics over Training Steps"
        if threshold_multiplier is not None and threshold_value is not None:
            # Using \u00B1 or similar if needed, but here we just show the exact values
            jump_title += f"\n(Threshold: {threshold_multiplier}x Real Std. Dev. \u2248 {threshold_value:.4f})"
            
        fig2.suptitle(jump_title, fontsize=16, y=1.05)
        
        def plot_jump_subplot(ax, x, gen_y, real_y, title, ylabel, color):
            # Plot the generated metrics over time
            ax.plot(x, gen_y, marker='o', linestyle='-', color=color, linewidth=2, markersize=6, label='Generated')
            # Plot the real metrics as a target baseline (dashed black line)
            ax.plot(x, real_y, linestyle='--', color='black', linewidth=2, label='Real (Target)')
            
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Generator Step", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.4)
            ax.legend()
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
        # 1. Jump Intensity
        plot_jump_subplot(axes2[0], steps, gen_jumps, real_jumps, "Jump Intensity", "Jumps / Year", "purple")
        
        # 2. Conditional Jump Mean
        plot_jump_subplot(axes2[1], steps, gen_means, real_means, "Conditional Jump Mean", "Mean Jump Size", "teal")
        
        # 3. Conditional Jump Volatility (Std)
        plot_jump_subplot(axes2[2], steps, gen_stds, real_stds, "Conditional Jump Volatility", "Std of Jump Size", "brown")
        
        # Adjust layout so the two-line suptitle fits nicely without overlapping the plots
        plt.tight_layout(rect=[0, 0, 1, 0.90]) 
        
        plot_path_jump = os.path.join(plots_dir, "jump_metrics_evolution.pdf")
        plt.savefig(plot_path_jump, format='pdf', bbox_inches='tight')
        plt.close(fig2)
        
        print(f"Successfully saved jump metrics evolution plot to: {plot_path_jump}")

if __name__ == "__main__":
    plot_metrics_evolution()