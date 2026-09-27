import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOCK import build_generator
from config import Config
from utils import seed_everything

def evaluate_conditional_drift(run_name: str, checkpoint_name: str = "generator_final.pt"):
    cfg = Config()
    cfg.eval_run_name = run_name # Override the run name to load specific models
    seed_everything(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # 1. Load Data
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    test_paths = data_dict["test_paths"]
    
    # 2. Load Model
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    ckpt_path = os.path.join(save_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    data_mean_tensor = checkpoint['data_mean'].to(device)
    data_std_tensor = checkpoint['data_std'].to(device)
    
    gen = build_generator(cfg.model).to(device)
    gen.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    gen.eval()
    
    # 3. Extract Contexts and Real Futures (Non-overlapping)
    N_len = test_paths.size(1)
    q = cfg.model.q_len
    T = cfg.model.T_len
    
    all_raw_contexts = []
    all_real_returns = []
    
    for t in range(0, N_len - q - T + 1, T):
        all_raw_contexts.append(test_paths[:, t : t + q, :])
        all_real_returns.append(test_paths[:, t + q : t + q + T, :])
            
    raw_contexts = torch.cat(all_raw_contexts, dim=0).to(device)
    real_returns = torch.cat(all_real_returns, dim=0).numpy()
    scaled_contexts = (raw_contexts - data_mean_tensor) / data_std_tensor
    
    # 4. Generate Fake Futures
    batch_size = 2048
    generated_scaled_list = []
    
    with torch.no_grad():
        for i in range(0, len(scaled_contexts), batch_size):
            batch_contexts = scaled_contexts[i : i + batch_size]
            gen_batch = gen(batch_contexts, n_steps=T)
            generated_scaled_list.append(gen_batch)
            
    generated_scaled = torch.cat(generated_scaled_list, dim=0)
    gen_returns = generated_scaled.cpu().numpy() * data_std_tensor.cpu().numpy() + data_mean_tensor.cpu().numpy()
    
    # 5. Calculate Conditional Drifts (Mean over the T=64 dimension)
    # We will analyze Asset 0 for visualization
    real_conditional_drifts = real_returns[:, :, 0].mean(axis=1)
    gen_conditional_drifts = gen_returns[:, :, 0].mean(axis=1)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(real_conditional_drifts - gen_conditional_drifts))
    
    return real_conditional_drifts, gen_conditional_drifts, mae

def plot_ab_test():
    # YOU NEED TO REPLACE THESE WITH YOUR ACTUAL FOLDER NAMES
    baseline_run = "20260720_1055_GBM_v1_not_regularized" 
    regularized_run = "20260720_1132_GBM_v1_more_regularized"
    
    print("Evaluating Baseline Model...")
    real_base, gen_base, mae_base = evaluate_conditional_drift(baseline_run)
    
    print("Evaluating Regularized Model...")
    real_reg, gen_reg, mae_reg = evaluate_conditional_drift(regularized_run)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    
    # Calculate axis limits for the diagonal line
    min_val = min(real_base.min(), gen_base.min())
    max_val = max(real_base.max(), gen_base.max())
    
    # Baseline Plot
    axes[0].scatter(real_base, gen_base, alpha=0.3, color='red', s=10)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal (y=x)")
    axes[0].set_title(f"Baseline (No Reg) | MAE: {mae_base:.6f}")
    axes[0].set_xlabel("Real Conditional Drift (T=64)")
    axes[0].set_ylabel("Generated Conditional Drift (T=64)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Regularized Plot
    axes[1].scatter(real_reg, gen_reg, alpha=0.3, color='blue', s=10)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal (y=x)")
    axes[1].set_title(f"Regularized Model | MAE: {mae_reg:.6f}")
    axes[1].set_xlabel("Real Conditional Drift (T=64)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.suptitle("Impact of Conditional Drift Regularization on SOCK Generator")
    plt.tight_layout()
    plt.savefig("conditional_drift_experiment.pdf", format='pdf')
    plt.show()

if __name__ == "__main__":
    plot_ab_test()