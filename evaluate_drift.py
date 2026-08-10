import os
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOCK import build_generator
from config import Config
from utils import seed_everything

def evaluate_model_drift(generator_step = "generator_final.pt"):
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    run_name = getattr(cfg, 'eval_run_name', cfg.train.experiment_name)
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    
    # Identify checkpoints to evaluate
    if str(generator_step).lower() == "all":
        ckpt_pattern = os.path.join(save_dir, "*.pt")
        ckpt_paths = glob.glob(ckpt_pattern)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint files found matching pattern {ckpt_pattern}")
        ckpt_paths = sorted(ckpt_paths)
    else:
        ckpt_path = os.path.join(save_dir, generator_step)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Could not find model checkpoint at {ckpt_path}")
        ckpt_paths = [ckpt_path]

    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    test_paths = data_dict["test_paths"]
    dataset_config = data_dict.get("dataset_config", {})
    
    first_ckpt = torch.load(ckpt_paths[0], map_location=device)
    data_mean = first_ckpt['data_mean'].to(device)
    data_std = first_ckpt['data_std'].to(device)
    del first_ckpt

    n_paths = 100000
    years = 8.0
    trading_days_per_year = 252
    total_steps = int(years * trading_days_per_year)
    
    q = cfg.model.q_len
    T_chunk = cfg.model.T_len
    
    true_mu = dataset_config.get("mu", 0.0)
    true_sigma = dataset_config.get("sigma", 0.0)
    theoretical_target_drift = true_mu - (0.5 * (true_sigma ** 2))
    
    raw_init_context = test_paths[:n_paths, :q, :].to(device)
    current_context = (raw_init_context - data_mean) / data_std
    
    all_results = {}

    for ckpt_path in ckpt_paths:
        step_name = os.path.basename(ckpt_path)
        print(f"\nEvaluating checkpoint: {step_name}...")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        gen = build_generator(cfg.model).to(device)
        gen.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
        gen.eval()
        
        print(f"Generating {n_paths} paths over {years} years for {run_name}...")
        batch_size = 1000
        all_generated_returns = []
        
        with torch.no_grad():
            for b in range(0, n_paths, batch_size):
                b_context = current_context[b : b + batch_size]
                b_paths = []
                steps_done = 0
                
                while steps_done < total_steps:
                    next_chunk_scaled = gen(b_context, n_steps=T_chunk)
                    b_paths.append(next_chunk_scaled)
                    steps_done += T_chunk
                    b_context = next_chunk_scaled[:, -q:, :]
                    
                b_generated_scaled = torch.cat(b_paths, dim=1)[:, :total_steps, :]
                b_generated_returns = b_generated_scaled * data_std + data_mean
                all_generated_returns.append(b_generated_returns.cpu())
                
        full_returns = torch.cat(all_generated_returns, dim=0).numpy()
        path_annualized_drifts = full_returns[:, :, 0].mean(axis=1) * trading_days_per_year
        
        mean_estimated_drift = float(np.mean(path_annualized_drifts))
        sample_std_drift = float(np.std(path_annualized_drifts))
        std_error = float(sample_std_drift / np.sqrt(n_paths))
        bias = float(mean_estimated_drift - theoretical_target_drift)
        
        results = {
            "run_name": run_name,
            "n_paths": n_paths,
            "years": years,
            "target_drift": float(theoretical_target_drift),
            "mean_drift": mean_estimated_drift,
            "bias": bias,
            "bias_bps": bias * 10000,
            "std_error": std_error,
            "std_error_bps": std_error * 10000,
            "ci_lower": mean_estimated_drift - 1.96 * std_error,
            "ci_upper": mean_estimated_drift + 1.96 * std_error,
        }
        
        all_results[step_name] = results
        
        print("\n" + "=" * 50)
        print(f"      DRIFT EVALUATION: {run_name} ({step_name})")
        print("=" * 50)
        print(f"Target Drift:           {results['target_drift']:.6f}")
        print(f"Estimated Mean Drift:   {results['mean_drift']:.6f}")
        print(f"Drift Bias (bps):       {results['bias_bps']:.2f}")
        print(f"Standard Error (bps):   {results['std_error_bps']:.2f}")
        print(f"95% CI:                 [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
        print("=" * 50)
        
        base_name = os.path.splitext(step_name)[0]
        json_path = os.path.join(save_dir, f"drift_metrics_{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved statistical metrics to: {json_path}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(path_annualized_drifts, bins=100, alpha=0.6, color='blue', density=True, 
                 label=f'Model (Bias: {results["bias_bps"]:.1f} bps)')
        plt.axvline(theoretical_target_drift, color='black', linestyle='--', linewidth=2.5, 
                    label=f'Theoretical Target ({theoretical_target_drift:.4f})')
        plt.title(f"Distribution of Annualized Drift - {step_name}\n({n_paths} Paths, {years} Years Horizon)", fontsize=14)
        plt.xlabel("Annualized Path Drift", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f"mc_drift_histogram_{base_name}.pdf")
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f"Saved histogram plot to: {plot_path}\n")
        plt.close()

    if str(generator_step).lower() == "all":
        summary_path = os.path.join(save_dir, "drift_metrics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved aggregated summary metrics to: {summary_path}")

    return all_results

if __name__ == "__main__":
    evaluate_model_drift(generator_step="all")