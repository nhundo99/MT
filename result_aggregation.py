import os
import json
import re
import glob
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==========================================
# 1. Heatmap Aggregation (Fixed Horizon)
# ==========================================
def aggregate_final_metrics(base_dir, experiment_folders, output_csv="thesis_drift_aggregation.csv"):
    aggregated_data = []
    if not os.path.exists(base_dir):
        return pd.DataFrame()

    for folder in experiment_folders:
        folder_path = os.path.join(base_dir, folder)
        # Only parse the drift_reg folders for the heatmap
        match = re.search(r'drift_reg_(\d+)_(\d+)', folder)
        if not match:
            continue
            
        horizon, samples = int(match.group(1)), int(match.group(2))
        json_path = os.path.join(folder_path, "drift_metrics_summary.json")
        
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            metrics = json.load(f)
            
        final_metrics = metrics.get("generator_final.pt", {})
        if final_metrics:
            aggregated_data.append({
                "Run Name": folder,
                "MC Horizon": horizon,
                "MC Samples": samples,
                "Target Drift": final_metrics.get("target_drift"),
                "Mean Drift": final_metrics.get("mean_drift"),
                "Bias (bps)": final_metrics.get("bias_bps"),
                "Std Error (bps)": final_metrics.get("std_error_bps")
            })

    df = pd.DataFrame(aggregated_data)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Heatmap data saved to {output_csv}")
    return df

def plot_thesis_heatmap(df, metric="Bias (bps)", save_path="drift_bias_heatmap.pdf"):
    if df.empty: return
    pivot_df = df.pivot(index="MC Horizon", columns="MC Samples", values=metric)
    plt.figure(figsize=(8, 6))
    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    center_val = 0 if "Bias" in metric else None
    
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap=cmap, center=center_val,
                cbar_kws={'label': metric}, linewidths=.5)
    
    plt.title(f"Impact of Monte Carlo Hyperparameters on {metric}", fontsize=14, pad=15)
    plt.xlabel("Number of MC Samples", fontsize=12)
    plt.ylabel("MC Horizon (Steps)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

# ==========================================
# 2. Validation Curves (with Standard Error)
# ==========================================
def aggregate_validation_curves(base_dir, experiment_folders, output_csv="thesis_val_curves.csv"):
    aggregated_data = []
    for folder in experiment_folders:
        folder_path = os.path.join(base_dir, folder)
        match = re.search(r'drift_reg_(\d+)_(\d+)', folder)
        if not match: continue
            
        horizon, samples = int(match.group(1)), int(match.group(2))
        ckpt_paths = glob.glob(os.path.join(folder_path, "generator_step_*.pt"))
        
        for ckpt_path in ckpt_paths:
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if 'val_metrics_50_steps' in checkpoint and 'step' in checkpoint:
                    val_metrics = checkpoint['val_metrics_50_steps']
                    aggregated_data.append({
                        "Run Name": folder,
                        "MC Horizon": horizon,
                        "MC Samples": samples,
                        "Step": checkpoint['step'],
                        "Bias (bps)": val_metrics.get('bias_bps', 0.0),
                        "Std Error (bps)": val_metrics.get('stderr_drift', 0.0) * 10000
                    })
            except: pass

    df = pd.DataFrame(aggregated_data)
    if not df.empty:
        df = df.sort_values(by=["MC Samples", "MC Horizon", "Step"])
        df.to_csv(output_csv, index=False)
        print(f"Validation curves data saved to {output_csv}")
    return df

def plot_validation_curves(df, save_path="validation_bias_curves.pdf"):
    if df.empty: return
    sns.set_theme(style="whitegrid")
    
    samples = sorted(df['MC Samples'].unique())
    fig, axes = plt.subplots(1, len(samples), figsize=(12, 5), sharey=True)
    if len(samples) == 1: axes = [axes]
    
    for ax, sample_val in zip(axes, samples):
        sample_df = df[df['MC Samples'] == sample_val]
        horizons = sorted(sample_df['MC Horizon'].unique())
        palette = sns.color_palette("viridis", n_colors=len(horizons))
        
        for i, horizon in enumerate(horizons):
            h_df = sample_df[sample_df['MC Horizon'] == horizon]
            
            ax.plot(h_df['Step'], h_df['Bias (bps)'], label=f"H={horizon}", 
                    color=palette[i], marker='o', markersize=4, linewidth=2)
            
            ax.fill_between(h_df['Step'], 
                            h_df['Bias (bps)'] - h_df['Std Error (bps)'], 
                            h_df['Bias (bps)'] + h_df['Std Error (bps)'], 
                            color=palette[i], alpha=0.2)
            
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label="Target (0 Bias)")
        ax.set_title(f"MC Samples = {sample_val}", fontsize=13)
        ax.set_xlabel("Training Step", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Validation Bias (bps)", fontsize=12)
        ax.legend(title="MC Horizon", fontsize=10)
    
    fig.suptitle("Convergence of Drift Bias During Training ($\pm1$ Standard Error)", fontsize=15, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

# ==========================================
# 3. Curriculum vs Baseline Visual & Numerical Comparison
# ==========================================
def plot_curriculum_comparison(base_dir, baseline_run, curriculum_run_1, curriculum_run_2, save_path="curriculum_vs_baseline.pdf"):
    data = []
    
    runs_to_compare = [
        (baseline_run, "Baseline (H=1024, N=256)"),
        (curriculum_run_1, "Curriculum Learning (Full)"),
        (curriculum_run_2, "Curriculum Learning (Short)")
    ]
    
    for folder, label in runs_to_compare:
        folder_path = os.path.join(base_dir, folder)
        ckpt_paths = glob.glob(os.path.join(folder_path, "generator_step_*.pt"))
        
        for ckpt_path in ckpt_paths:
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if 'val_metrics_50_steps' in checkpoint and 'step' in checkpoint:
                    val_metrics = checkpoint['val_metrics_50_steps']
                    data.append({
                        "Model Setup": label,
                        "Step": checkpoint['step'],
                        "Bias (bps)": val_metrics.get('bias_bps', 0.0),
                        "Std Error (bps)": val_metrics.get('stderr_drift', 0.0) * 10000
                    })
            except Exception as e:
                print(f"Error reading {ckpt_path}: {e}")

    df = pd.DataFrame(data)
    if df.empty:
        print("No validation data found for curriculum comparison.")
        return
        
    df = df.sort_values(by=["Model Setup", "Step"])
    
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    
    models = df['Model Setup'].unique()
    palette = sns.color_palette("Set1", n_colors=len(models))
    
    for i, model in enumerate(models):
        model_df = df[df['Model Setup'] == model]
        plt.plot(model_df['Step'], model_df['Bias (bps)'], label=model, 
                 color=palette[i], marker='o', markersize=5, linewidth=2)
        plt.fill_between(model_df['Step'],
                         model_df['Bias (bps)'] - model_df['Std Error (bps)'],
                         model_df['Bias (bps)'] + model_df['Std Error (bps)'],
                         color=palette[i], alpha=0.2)
        
    plt.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Perfect Target (0 Bias)")
    
    plt.title("Drift Bias Convergence: Curriculum Learning vs. Fixed Horizon ($\pm1$ SE)", fontsize=14, pad=15)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Validation Bias (bps)", fontsize=12)
    plt.legend(title="Experiment Setup")
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Curriculum comparison plot saved to {save_path}")

def compare_baseline_and_curriculum_metrics(base_dir, baseline_run, curriculum_run_1, curriculum_run_2, output_csv="curriculum_metrics_comparison.csv"):
    runs_to_compare = {
        "Baseline (H=1024, N=256)": baseline_run,
        "Curriculum Learning (Full)": curriculum_run_1,
        "Curriculum Learning (Short)": curriculum_run_2
    }
    
    aggregated_data = []
    
    for label, folder in runs_to_compare.items():
        folder_path = os.path.join(base_dir, folder)
        json_path = os.path.join(folder_path, "drift_metrics_summary.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found for {label}")
            continue
            
        with open(json_path, 'r') as f:
            metrics = json.load(f)
            
        final_metrics = metrics.get("generator_final.pt", {})
        if final_metrics:
            aggregated_data.append({
                "Model Setup": label,
                "Target Drift": final_metrics.get("target_drift"),
                "Mean Drift": final_metrics.get("mean_drift"),
                "Bias (bps)": final_metrics.get("bias_bps"),
                "Std Error (bps)": final_metrics.get("std_error_bps")
            })

    df = pd.DataFrame(aggregated_data)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Numerical Comparison saved to {output_csv}")
    return df

# ==========================================
# 4. Extracting Training Times via TensorBoard
# ==========================================
def extract_training_times_tb(runs_base_dir, experiment_folders, output_json="training_times.json"):
    times_dict = {}
    
    for folder in experiment_folders:
        folder_path = os.path.join(runs_base_dir, folder)
        event_files = glob.glob(os.path.join(folder_path, "events.out.tfevents.*"))
        
        if not event_files:
            continue
            
        try:
            event_file = sorted(event_files)[-1]
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            if 'Loss/train_total' in ea.Tags().get('scalars', []):
                events = ea.Scalars('Loss/train_total')
                start_time = events[0].wall_time
                end_time = events[-1].wall_time
                
                duration_seconds = end_time - start_time
                hours, remainder = divmod(duration_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                times_dict[folder] = {
                    "start_step": events[0].step,
                    "end_step": events[-1].step,
                    "duration_seconds": duration_seconds,
                    "formatted_time": f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                }
        except Exception as e:
            print(f"Error processing TB logs for {folder}: {e}")

    with open(output_json, 'w') as f:
        json.dump(times_dict, f, indent=4)
        
    return times_dict

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    RESULTS_DIR = "../results/checkpoints"
    RUNS_DIR = "../results/runs"
    
    EXPERIMENTS = [
        "20260824_0811_GBM_wrong_drift_drift_reg_64_256",
        "20260824_1138_GBM_wrong_drift_drift_reg_128_256",
        "20260822_1935_GBM_wrong_drift_drift_reg_256_256",
        "20260823_0952_GBM_wrong_drift_drift_reg_512_256",
        "20260823_1307_GBM_wrong_drift_drift_reg_1024_256",
        "20260824_1339_GBM_wrong_drift_curriculum_learning_full",
        "20260919_1148_GBM_wrong_drift_curriculum_learning_short"
    ]

    print("--- 1. Aggregating Final Metrics for Heatmap ---")
    final_df = aggregate_final_metrics(RESULTS_DIR, EXPERIMENTS)
    plot_thesis_heatmap(final_df, metric="Bias (bps)")

    print("\n--- 2. Aggregating Validation Curves ---")
    val_df = aggregate_validation_curves(RESULTS_DIR, EXPERIMENTS)
    plot_validation_curves(val_df)
    
    print("\n--- 3. Analyzing Curriculum vs Baseline ---")
    baseline = "20260823_1307_GBM_wrong_drift_drift_reg_1024_256"
    curr_full = "20260824_1339_GBM_wrong_drift_curriculum_learning_full"
    curr_short = "20260919_1148_GBM_wrong_drift_curriculum_learning_short"
    
    # Generate the validation trajectory plot
    plot_curriculum_comparison(
        base_dir=RESULTS_DIR,
        baseline_run=baseline,
        curriculum_run_1=curr_full,
        curriculum_run_2=curr_short
    )
    
    # Generate and save the numerical final stats comparison to CSV
    compare_baseline_and_curriculum_metrics(
        base_dir=RESULTS_DIR,
        baseline_run=baseline,
        curriculum_run_1=curr_full,
        curriculum_run_2=curr_short
    )
    
    print("\n--- 4. Extracting Training Times (via TensorBoard) ---")
    times = extract_training_times_tb(RUNS_DIR, EXPERIMENTS)
    print(f"Training times exported to training_times.json")