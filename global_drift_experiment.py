import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOCK import build_generator
from config import Config
from utils import seed_everything

def get_expected_cumulative_trajectory(run_name: str, test_paths: torch.Tensor, device: torch.device):
    """
    Loads a specific model, generates thousands of futures, and calculates 
    the mean cumulative drift in the STANDARDIZED space (target = 0.0).
    """
    cfg = Config()
    cfg.eval_run_name = run_name 
    
    # 1. Load Model Checkpoint
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    ckpt_path = os.path.join(save_dir, "generator_final.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find model at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    data_mean_tensor = checkpoint['data_mean'].to(device)
    data_std_tensor = checkpoint['data_std'].to(device)
    
    gen = build_generator(cfg.model).to(device)
    gen.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    gen.eval()
    
    # 2. Extract Contexts from out-of-sample data
    N_len = test_paths.size(1)
    q = cfg.model.q_len
    T = cfg.model.T_len
    
    all_raw_contexts = []
    all_real_returns = []
    
    for t in range(0, N_len - q - T + 1, T):
        all_raw_contexts.append(test_paths[:, t : t + q, :])
        all_real_returns.append(test_paths[:, t + q : t + q + T, :])
        
    raw_contexts = torch.cat(all_raw_contexts, dim=0).to(device)
    real_returns = torch.cat(all_real_returns, dim=0).to(device)
    
    # Scale contexts
    scaled_contexts = (raw_contexts - data_mean_tensor) / data_std_tensor
    scaled_real = (real_returns - data_mean_tensor) / data_std_tensor
    
    # 3. Generate Fake Futures (Batched)
    batch_size = 2048
    generated_scaled_list = []
    
    with torch.no_grad():
        for i in range(0, len(scaled_contexts), batch_size):
            batch_contexts = scaled_contexts[i : i + batch_size]
            gen_batch = gen(batch_contexts, n_steps=T)
            generated_scaled_list.append(gen_batch)
            
    generated_scaled = torch.cat(generated_scaled_list, dim=0)
    
    # 4. Calculate Cumulative Trajectories (in Scaled Space)
    # We analyze Asset 0. Target is exactly 0.0 across all steps.
    gen_cum_returns = torch.cumsum(generated_scaled[:, :, 0], dim=1).cpu().numpy()
    real_cum_returns = torch.cumsum(scaled_real[:, :, 0], dim=1).cpu().numpy()
    
    # The "Expected Path" is the mean across all thousands of generated paths
    expected_gen_path = gen_cum_returns.mean(axis=0)
    expected_real_path = real_cum_returns.mean(axis=0)
    
    return expected_gen_path, expected_real_path


def plot_global_drift_comparison():
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # --- UPDATE THESE WITH YOUR ACTUAL FOLDER NAMES ---
    unregularized_run = "20260720_1512_GBM_v1_global_non_regularizedp" 
    regularized_run = "20260720_1524_GBM_v1_global_regularized"
    
    # Load raw dataset once
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    test_paths = data_dict["test_paths"]
    
    print("Evaluating Unregularized Model...")
    gen_unreg_path, real_path = get_expected_cumulative_trajectory(unregularized_run, test_paths, device)
    
    print("Evaluating Global Regularized Model...")
    gen_reg_path, _ = get_expected_cumulative_trajectory(regularized_run, test_paths, device)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(1, cfg.model.T_len + 1)
    
    # The perfect theoretical target in scaled space is a flat line at 0.0
    plt.axhline(0.0, color='black', linestyle='--', linewidth=2, label='Theoretical Target (0.0)')
    
    # Plot the Real Data's mean (It will fluctuate slightly around 0 due to finite sample size)
    plt.plot(time_steps, real_path, color='gray', alpha=0.6, linewidth=2, label='Real Data (Out-of-Sample Mean)')
    
    # Plot the Unregularized Model (This should drift visibly up or down)
    plt.plot(time_steps, gen_unreg_path, color='red', linewidth=2.5, label='Unregularized Model (Biased)')
    
    # Plot the Regularized Model (This should stay tightly clamped to the 0.0 line)
    plt.plot(time_steps, gen_reg_path, color='blue', linewidth=2.5, label='Global Regularized Model (Anchored)')
    
    plt.title("Compounding Global Drift: Unregularized vs. Regularized Generator", fontsize=14)
    plt.xlabel("Time Step (T)", fontsize=12)
    plt.ylabel("Expected Cumulative Return (Scaled Space)", fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "global_drift_comparison.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\nPlot successfully saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_global_drift_comparison()