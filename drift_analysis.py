import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from SOCK import Generator
from data_loader import FinancialTimeSeriesDataset
from config import Config
from utils import seed_everything

def analyze_cumulative_drift(checkpoints_to_plot=[10000, 50000, 100000]):
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. Load data
    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    train_path = data_dict["train_path"]
    test_paths = data_dict["test_paths"] # Shape: (J, N, d)
    
    dataset = FinancialTimeSeriesDataset(train_path, q=cfg.model.q_len, T=cfg.model.T_len)
    mean = dataset.mean.squeeze().cpu().numpy()
    std = dataset.std.squeeze().cpu().numpy()
    
    print("Calculating Ground Truth Quantiles from Continuations...")
    # Because we're predicting horizon T, we extract the first T steps of each out-of-sample path
    real_returns = test_paths[:, :cfg.model.T_len, :].numpy() # (2048, T, d)
    real_cum_returns = np.cumsum(real_returns, axis=1) # Cumulative sum = log prices
    
    # Ground Truth Quantiles
    real_q05 = np.percentile(real_cum_returns, 5, axis=0)
    real_q15 = np.percentile(real_cum_returns, 15, axis=0)
    real_q50 = np.percentile(real_cum_returns, 50, axis=0)
    real_q85 = np.percentile(real_cum_returns, 85, axis=0)
    real_q95 = np.percentile(real_cum_returns, 95, axis=0)

    # 2. Setup Generator
    gen = Generator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim).to(device)
    save_dir = os.path.join(cfg.train.model_base_dir, cfg.train.experiment_name)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    num_samples = test_paths.size(0) # J = 2048
    
    # Condition generation on the very end of the training data
    context = dataset.scaled_path[-cfg.model.q_len:].unsqueeze(0).to(device)
    batched_context = context.repeat(num_samples, 1, 1)

    checkpoints = [(step, f"generator_step_{step}.pt") for step in checkpoints_to_plot]
    checkpoints.append(("Final", "generator_final.pt"))

    for step_label, ckpt_name in checkpoints:
        ckpt_path = os.path.join(save_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"Analyzing drift for checkpoint {step_label}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['generator_state_dict'])
        else:
            gen.load_state_dict(checkpoint)
            
        gen.eval()
        
        # 3. Generate multiple futures
        with torch.no_grad():
            generated_scaled = gen(batched_context, n_steps=cfg.model.T_len)
            
        generated_returns = generated_scaled.cpu().numpy() * std + mean
        cum_log_returns = np.cumsum(generated_returns, axis=1) 
        
        mod_q05 = np.percentile(cum_log_returns, 5, axis=0)
        mod_q15 = np.percentile(cum_log_returns, 15, axis=0)
        mod_q50 = np.percentile(cum_log_returns, 50, axis=0)
        mod_q85 = np.percentile(cum_log_returns, 85, axis=0)
        mod_q95 = np.percentile(cum_log_returns, 95, axis=0)

        real_annualized_drift = (real_q50[-1, 0] / cfg.model.T_len) * 252
        model_annualized_drift = (mod_q50[-1, 0] / cfg.model.T_len) * 252
        drift_bias = model_annualized_drift - real_annualized_drift
        
        print(f"--- Drift Bias Analysis (Checkpoint {step_label}) ---")
        print(f"Real Annualized Drift:  {real_annualized_drift:.4f}")
        print(f"Model Annualized Drift: {model_annualized_drift:.4f}")
        print(f"Drift Bias (Model - Real): {drift_bias:.4f}")
        
        # 4. Plotting for Asset 1
        plt.figure(figsize=(8, 5))
        time_steps = np.arange(1, cfg.model.T_len + 1)
        
        plt.fill_between(time_steps, mod_q05[:, 0], mod_q95[:, 0], color='#4C72B0', alpha=0.2, label='Model $Q_{0.05} - Q_{0.95}$')
        plt.fill_between(time_steps, mod_q15[:, 0], mod_q85[:, 0], color='#4C72B0', alpha=0.4, label='Model $Q_{0.15} - Q_{0.85}$')
        
        plt.plot(time_steps, real_q05[:, 0], color='black', linestyle=':', linewidth=1.5, label='Real $Q_{0.05} - Q_{0.95}$')
        plt.plot(time_steps, real_q95[:, 0], color='black', linestyle=':', linewidth=1.5)
        plt.plot(time_steps, real_q15[:, 0], color='black', linestyle='--', linewidth=1.5, label='Real $Q_{0.15} - Q_{0.85}$')
        plt.plot(time_steps, real_q85[:, 0], color='black', linestyle='--', linewidth=1.5)

        plt.plot(time_steps, mod_q50[:, 0], color='red', linewidth=2.5, label='Model Median (Drift)')
        plt.plot(time_steps, real_q50[:, 0], color='black', linewidth=2.5, label='Real Median (Drift)')

        plt.title(f"Cumulative Log Returns Drift Analysis - Step {step_label}")
        plt.xlabel("Time $t$ (Trading Days)")
        plt.ylabel("Cum. Log Returns")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(plot_dir, f"drift_analysis_step_{step_label}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    analyze_cumulative_drift()