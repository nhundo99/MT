import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOCK import build_generator
from config import Config
from utils import seed_everything

def evaluate_model_drift():
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    run_name = getattr(cfg, 'eval_run_name', cfg.train.experiment_name)
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    
    n_paths = 100000
    years = 8.0
    trading_days_per_year = 252
    total_steps = int(years * trading_days_per_year)
    
    q = cfg.model.q_len
    T_chunk = cfg.model.T_len
    
    ckpt_path = os.path.join(save_dir, "generator_final.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find model checkpoint at {ckpt_path}")
        
    print(f"Loading checkpoint: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    data_mean = checkpoint['data_mean'].to(device)
    data_std = checkpoint['data_std'].to(device)
    
    gen = build_generator(cfg.model).to(device)
    gen.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    gen.eval()
    
    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    test_paths = data_dict["test_paths"]
    dataset_config = data_dict.get("dataset_config", {})
    
    true_mu = dataset_config.get("mu", 0.0)
    true_sigma = dataset_config.get("sigma", 0.0)
    theoretical_target_drift = true_mu - (0.5 * (true_sigma ** 2))
    
    raw_init_context = test_paths[:n_paths, :q, :].to(device)
    current_context = (raw_init_context - data_mean) / data_std
    
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
    
    print("\n" + "=" * 50)
    print(f"      DRIFT EVALUATION: {run_name}")
    print("=" * 50)
    print(f"Target Drift:           {results['target_drift']:.6f}")
    print(f"Estimated Mean Drift:   {results['mean_drift']:.6f}")
    print(f"Drift Bias (bps):       {results['bias_bps']:.2f}")
    print(f"Standard Error (bps):   {results['std_error_bps']:.2f}")
    print(f"95% CI:                 [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    print("=" * 50)
    
    json_path = os.path.join(save_dir, "drift_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved statistical metrics to: {json_path}")
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(path_annualized_drifts, bins=100, alpha=0.6, color='blue', density=True, 
             label=f'Model (Bias: {results["bias_bps"]:.1f} bps)')
    
    plt.axvline(theoretical_target_drift, color='black', linestyle='--', linewidth=2.5, 
                label=f'Theoretical Target ({theoretical_target_drift:.4f})')
    
    plt.title(f"Distribution of Annualized Drift\n({n_paths} Paths, {years} Years Horizon)", fontsize=14)
    plt.xlabel("Annualized Path Drift", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "mc_drift_histogram.pdf")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved histogram plot to: {plot_path}\n")
    
    plt.show()
    return results

if __name__ == "__main__":
    evaluate_model_drift()