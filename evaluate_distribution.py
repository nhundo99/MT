import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cramervonmises_2samp

from SOCK import *
from config import Config
from utils import seed_everything


# -------------------------------------------------------------------------
# 1. Metric Helper Functions
# -------------------------------------------------------------------------

def compute_cvm_distance(real_returns: np.ndarray, gen_returns: np.ndarray) -> float:
    """
    Cramér-von Mises (CVM) Distance:
    Measures 1D marginal fit by comparing real vs generated return ECDFs
    aggregated across time and paths for each channel.
    """
    d = real_returns.shape[-1]
    real_flat = real_returns.reshape(-1, d)
    gen_flat = gen_returns.reshape(-1, d)

    N = real_flat.shape[0]
    M = gen_flat.shape[0]
    scipy_scaling_factor = (N * M) / (N + M)
    
    cvm_scores = []
    for j in range(d):
        res = cramervonmises_2samp(real_flat[:, j], gen_flat[:, j])
        unscaled_distance = res.statistic / scipy_scaling_factor
        cvm_scores.append(unscaled_distance)
        
    return float(np.mean(cvm_scores))


def compute_acf_difference(real_returns: np.ndarray, gen_returns: np.ndarray, l_max: int) -> float:
    """
    Autocorrelation Function (ACF) Discrepancy:
    Calculates average absolute difference between channel-wise empirical autocorrelation
    up to lag l_max = floor(T / 3).
    """
    B, T, d = real_returns.shape
    
    def calc_panel_acf(data: np.ndarray) -> np.ndarray:
        # data shape: (B, T, d)
        acfs = np.zeros((d, l_max))
        for j in range(d):
            x = data[:, :, j]
            mean_j = np.mean(x)
            x_centered = x - mean_j
            denom = np.sum(x_centered ** 2) + 1e-12
            
            for k in range(1, l_max + 1):
                num = np.sum(x_centered[:, :-k] * x_centered[:, k:])
                acfs[j, k - 1] = num / denom
        return acfs

    acf_real = calc_panel_acf(real_returns)
    acf_gen = calc_panel_acf(gen_returns)

    abs_diff = np.abs(acf_real - acf_gen)
    return float(np.mean(abs_diff))


def compute_ccf_difference(real_returns: np.ndarray, gen_returns: np.ndarray) -> float:
    """
    Cross-Correlation Function (CCF) Discrepancy:
    Calculates average absolute discrepancy between Pearson cross-correlation matrices
    computed across aggregated observations.
    """
    d = real_returns.shape[-1]
    if d <= 1:
        return 0.0
    
    real_flat = real_returns.reshape(-1, d)
    gen_flat = gen_returns.reshape(-1, d)
    
    corr_real = np.corrcoef(real_flat, rowvar=False)
    corr_gen = np.corrcoef(gen_flat, rowvar=False)
    
    mask = ~np.eye(d, dtype=bool)
    diff = np.abs(corr_real[mask] - corr_gen[mask])
    
    return float(np.mean(diff))


def compute_es_difference(real_returns: np.ndarray, gen_returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Expected Shortfall (ES) Discrepancy:
    Calculates average relative absolute error in Expected Shortfall at tail level alpha (5%).
    """
    d = real_returns.shape[-1]
    real_flat = real_returns.reshape(-1, d)
    gen_flat = gen_returns.reshape(-1, d)
    
    es_diffs = []
    for j in range(d):
        q_real = np.percentile(real_flat[:, j], alpha * 100)
        tail_real = real_flat[real_flat[:, j] <= q_real, j]
        es_real = np.mean(tail_real)
        
        q_gen = np.percentile(gen_flat[:, j], alpha * 100)
        tail_gen = gen_flat[gen_flat[:, j] <= q_gen, j]
        es_gen = np.mean(tail_gen)
        
        rel_diff = np.abs(es_real - es_gen) / (np.abs(es_real) + 1e-8)
        es_diffs.append(rel_diff)
        
    return float(np.mean(es_diffs))

def plot_jump_diagnostics(real_returns: np.ndarray, gen_returns: np.ndarray):
    """
    Plots individual path realizations and log-scaled histograms 
    to diagnose jump-generation failures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(real_returns[0, :, 0], label='Real (Jump Only)', marker='o', linestyle='-', color='black')
    axes[0].plot(gen_returns[0, :, 0], label='Generated', marker='x', linestyle='--', color='red', alpha=0.7)
    axes[0].set_title("Step-by-Step Returns: Single Path")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(real_returns[:, :, 0].flatten(), bins=100, alpha=0.5, label='Real', color='black', log=True)
    axes[1].hist(gen_returns[:, :, 0].flatten(), bins=100, alpha=0.5, label='Generated', color='red', log=True)
    axes[1].set_title("Distribution of Returns (Log Scale)")
    axes[1].set_xlabel("Return Size")
    axes[1].set_ylabel("Frequency (Log Scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("jump_diagnostics.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def analyze_jump_behavior(real_returns: np.ndarray, gen_returns: np.ndarray, threshold: float = 0.03):
    """
    Analyzes the frequency and size of jumps in the returns data.
    Uses a threshold to filter out the 'fuzzy zeros' produced by continuous NNs.
    """
    print("\n" + "=" * 50)
    print(f"      JUMP BEHAVIOR ANALYSIS (Threshold = {threshold})")
    print("=" * 50)
    
    real_asset = real_returns[:, :, 0]
    gen_asset = gen_returns[:, :, 0]
    
    real_jump_mask = np.abs(real_asset) > threshold
    gen_jump_mask = np.abs(gen_asset) > threshold
    
    real_jumps_per_year = np.mean(real_jump_mask) * 252
    gen_jumps_per_year = np.mean(gen_jump_mask) * 252
    
    real_jump_sizes = real_asset[real_jump_mask]
    gen_jump_sizes = gen_asset[gen_jump_mask]
    
    real_mean, real_std = np.mean(real_jump_sizes), np.std(real_jump_sizes)
    
    if len(gen_jump_sizes) > 0:
        gen_mean, gen_std = np.mean(gen_jump_sizes), np.std(gen_jump_sizes)
    else:
        gen_mean, gen_std = 0.0, 0.0
        
    print(f"Jump Intensity (Jumps/Year): Real = {real_jumps_per_year:.2f} | Gen = {gen_jumps_per_year:.2f}")
    print(f"Conditional Jump Mean:       Real = {real_mean:.4f} | Gen = {gen_mean:.4f}")
    print(f"Conditional Jump Volatility: Real = {real_std:.4f} | Gen = {gen_std:.4f}")
    print("=" * 50)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(real_jump_sizes, bins=50, alpha=0.5, label='Real Jumps', color='black', density=True)
    
    if len(gen_jump_sizes) > 0:
        plt.hist(gen_jump_sizes, bins=50, alpha=0.5, label='Generated Jumps', color='red', density=True)
    
    plt.axvline(threshold, color='blue', linestyle='--', linewidth=1, label='+Threshold')
    plt.axvline(-threshold, color='blue', linestyle='--', linewidth=1, label='-Threshold')
    
    plt.title(f"Conditional Distribution of Jump Sizes (Returns > {threshold})")
    plt.xlabel("Return Size")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("jump_size_distribution.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------------
# 2. Main Evaluation Pipeline
# -------------------------------------------------------------------------

def evaluate_distributional_metrics(checkpoint_name: str = "generator_final.pt"):
    cfg = Config()
    seed_everything(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Running evaluation on device: {device}")
    
    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    
    # 1. Conditionally load and concatenate volatility for testing
    use_volatility = getattr(cfg.train, 'use_volatility', False)
    
    if use_volatility and "train_vol" in data_dict and "test_vols" in data_dict:
        print("Volatility usage ENABLED. Concatenating returns and volatility paths for evaluation...")
        train_data = torch.cat([data_dict["train_path"], data_dict["train_vol"]], dim=-1)
        test_data = torch.cat([data_dict["test_paths"], data_dict["test_vols"]], dim=-1)
    else:
        print("Using standard returns path...")
        train_data = data_dict["train_path"]
        test_data = data_dict["test_paths"]
    
    save_dir = os.path.join(cfg.train.model_base_dir, cfg.train.experiment_name)
    ckpt_path = os.path.join(save_dir, checkpoint_name)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    print(f"Loading checkpoint: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 2. Compute mean/std based on the joint train_data if missing
    if 'data_mean' in checkpoint and 'data_std' in checkpoint:
        data_mean_tensor = checkpoint['data_mean'].to(device)
        data_std_tensor = checkpoint['data_std'].to(device)
    else:
        print("Warning: data_mean/data_std missing in checkpoint. Calculating from train path...")
        data_mean_tensor = train_data.mean(dim=0, keepdim=True).to(device)
        data_std_tensor = train_data.std(dim=0, keepdim=True).to(device) + 1e-6
        
    data_mean_np = data_mean_tensor.cpu().numpy()
    data_std_np = data_std_tensor.cpu().numpy()
    
    N_len = test_data.size(1)
    q = cfg.model.q_len
    T = cfg.model.T_len
    
    all_raw_contexts = []
    all_real_data = []
    
    for t in range(0, N_len - q - T + 1, T):
        all_raw_contexts.append(test_data[:, t : t + q, :])
        all_real_data.append(test_data[:, t + q : t + q + T, :])
        
    raw_contexts = torch.cat(all_raw_contexts, dim=0).to(device)
    real_data = torch.cat(all_real_data, dim=0).numpy() # (B, T, 2d) if use_volatility
    scaled_contexts = (raw_contexts - data_mean_tensor) / data_std_tensor
    
    gen = build_generator(cfg.model).to(device)
    if 'generator_state_dict' in checkpoint:
        gen.load_state_dict(checkpoint['generator_state_dict'])
    else:
        gen.load_state_dict(checkpoint)
    gen.eval()
    
    print(f"Generating synthetic paths for {len(scaled_contexts)} evaluation segments...")
    batch_size = 2048
    generated_scaled_list = []
    
    with torch.no_grad():
        for i in range(0, len(scaled_contexts), batch_size):
            batch_contexts = scaled_contexts[i : i + batch_size]
            gen_batch = gen(batch_contexts, n_steps=T)
            generated_scaled_list.append(gen_batch)
            
    generated_scaled = torch.cat(generated_scaled_list, dim=0)
    gen_data = generated_scaled.cpu().numpy() * data_std_np + data_mean_np 
    
    d = cfg.model.d
    if use_volatility:
        real_returns = real_data[..., :d]
        gen_returns = gen_data[..., :d]
    else:
        real_returns = real_data
        gen_returns = gen_data

    print("\n" + "=" * 50)
    print("      DISTRIBUTIONAL METRICS EVALUATION")
    print("=" * 50)
    
    l_max = int(np.floor(T / 3))
    
    cvm_val = compute_cvm_distance(real_returns, gen_returns)
    acf_val = compute_acf_difference(real_returns, gen_returns, l_max=l_max)
    ccf_val = compute_ccf_difference(real_returns, gen_returns)
    es_val  = compute_es_difference(real_returns, gen_returns, alpha=0.05)
    
    results = {
        "CVM": round(cvm_val, 6),
        "ACF": round(acf_val, 6),
        "CCF": round(ccf_val, 6),
        "ES": round(es_val, 6),
        "num_segments_evaluated": len(real_returns),
        "rollout_horizon_T": T,
        "context_length_q": q,
    }
    
    print(f"Cramér-von Mises (CVM):         {results['CVM']:.6f}")
    print(f"Autocorrelation Diff (ACF):    {results['ACF']:.6f}")
    print(f"Cross-Correlation Diff (CCF):  {results['CCF']:.6f}")
    print(f"Expected Shortfall Diff (ES):  {results['ES']:.6f}")
    print("=" * 50)
    
    json_path = os.path.join(save_dir, "distributional_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved statistical metrics to: {json_path}\n")
    plot_jump_diagnostics(real_returns, gen_returns)
    analyze_jump_behavior(real_returns, gen_returns, threshold=0.03)
    return results




if __name__ == "__main__":
    evaluate_distributional_metrics()