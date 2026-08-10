import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOCK import build_generator
from config import Config
from utils import seed_everything

def get_model_drift_stats(
    run_name: str, 
    n_paths: int, 
    total_steps: int,
    trading_days_per_year: int,
    device: torch.device
):
    """Generates paths for a single model and returns the drift statistics."""
    cfg = Config()
    seed_everything(cfg.seed)
    cfg.eval_run_name = run_name
    
    q = cfg.model.q_len
    T_chunk = cfg.model.T_len
    
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    ckpt_path = os.path.join(save_dir, "generator_final.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find model checkpoint at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    data_mean = checkpoint['data_mean'].to(device)
    data_std = checkpoint['data_std'].to(device)
    
    gen = build_generator(cfg.model).to(device)
    gen.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    gen.eval()
    
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    test_paths = data_dict["test_paths"]
    dataset_config = data_dict.get("dataset_config", {})
    
    true_mu = dataset_config.get("mu", 0.0)
    true_sigma = dataset_config.get("sigma", 0.0)
    theoretical_target_drift = true_mu - (0.5 * (true_sigma ** 2))
    
    raw_init_context = test_paths[:n_paths, :q, :].to(device)
    current_context = (raw_init_context - data_mean) / data_std
    
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
    
    mean_estimated_drift = np.mean(path_annualized_drifts)
    sample_std_drift = np.std(path_annualized_drifts)
    std_error = sample_std_drift / np.sqrt(n_paths)
    
    return {
        "run_name": run_name,
        "mean_drift": mean_estimated_drift,
        "std_error": std_error,
        "ci_lower": mean_estimated_drift - 1.96 * std_error,
        "ci_upper": mean_estimated_drift + 1.96 * std_error,
        "bias": mean_estimated_drift - theoretical_target_drift,
        "target": theoretical_target_drift,
        "all_drifts": path_annualized_drifts
    }


def run_comparison():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # CHANGE TO PICK COMPARISON
    baseline_run = "20260806_0920_GBM_wrong_drift_monte_carlo_not_regularized"
    regularized_run = "20260806_1601_GBM_wrong_drift_monte_carlo_regularized_50000_steps"
    
    n_paths = 100000
    years = 8.0
    trading_days = 252
    total_steps = int(years * trading_days)
    
    print(f"Generating {n_paths} paths over {years} years for baseline model...")
    stats_base = get_model_drift_stats(baseline_run, n_paths, total_steps, trading_days, device)
    
    print(f"Generating {n_paths} paths over {years} years for regularized model...")
    stats_reg = get_model_drift_stats(regularized_run, n_paths, total_steps, trading_days, device)
    
    target = stats_base["target"]
    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'Baseline (Unregularized)':<25} | {'MC Regularized':<25}")
    print("-" * 80)
    print(f"{'Target Drift':<25} | {target:.<25.6f} | {target:.<25.6f}")
    print(f"{'Estimated Mean Drift':<25} | {stats_base['mean_drift']:.<25.6f} | {stats_reg['mean_drift']:.<25.6f}")
    print(f"{'Drift Bias (bps)':<25} | {stats_base['bias']*10000:.<25.2f} | {stats_reg['bias']*10000:.<25.2f}")
    print(f"{'95% CI Lower':<25} | {stats_base['ci_lower']:.<25.6f} | {stats_reg['ci_lower']:.<25.6f}")
    print(f"{'95% CI Upper':<25} | {stats_base['ci_upper']:.<25.6f} | {stats_reg['ci_upper']:.<25.6f}")
    print(f"{'Standard Error (bps)':<25} | {stats_base['std_error']*10000:.<25.2f} | {stats_reg['std_error']*10000:.<25.2f}")
    print("="*80)
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(stats_base["all_drifts"], bins=100, alpha=0.5, color='red', density=True, label=f'Baseline (Bias: {stats_base["bias"]*10000:.1f} bps)')
    plt.hist(stats_reg["all_drifts"], bins=100, alpha=0.6, color='blue', density=True, label=f'Regularized (Bias: {stats_reg["bias"]*10000:.1f} bps)')
    
    plt.axvline(target, color='black', linestyle='--', linewidth=2.5, label=f'Theoretical Target ({target:.4f})')
    
    plt.title(f"Distribution of Annualized Drift ({n_paths} Paths, {years} Years Horizon)", fontsize=14)
    plt.xlabel("Annualized Path Drift", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "mc_drift_histogram_comparison.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\nSaved histogram plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_comparison()