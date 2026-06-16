import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from SOCK import Generator
from data_loader import JumpDiffusionSimulator, FinancialTimeSeriesDataset
from config import Config
from utils import seed_everything

def analyze_cumulative_drift(checkpoints_to_plot=[10000, 50000, 100000]):
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. Prepare data and simulator
    sim = JumpDiffusionSimulator(d=cfg.model.d)
    hist_path = sim.simulate(H=2048)
    dataset = FinancialTimeSeriesDataset(hist_path, q=cfg.model.q_len, T=cfg.model.T_len)
    
    mean = dataset.mean.squeeze().cpu().numpy()
    std = dataset.std.squeeze().cpu().numpy()
    
    print("Calculating Ground Truth Quantiles...")
    # Extract all real future paths of length T from the dataset to calculate empirical real drift
    real_paths = []
    for i in range(len(dataset)):
        _, x_plus = dataset[i]
        real_paths.append(x_plus.cpu().numpy())
        
    real_paths = np.array(real_paths) # Shape: (N, T, d)
    real_returns = real_paths * std + mean
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
    
    num_samples = 5000 # Number of paths to simulate for a robust distribution
    
    # We will use the very first context in the dataset to condition our generated futures
    context = dataset.scaled_path[:cfg.model.q_len].unsqueeze(0).to(device)
    batched_context = context.repeat(num_samples, 1, 1)

    # Add the final model to the list of checkpoints to plot
    checkpoints = [(step, f"generator_step_{step}.pt") for step in checkpoints_to_plot]
    checkpoints.append(("Final", "generator_final.pt"))

    for step_label, ckpt_name in checkpoints:
        ckpt_path = os.path.join(save_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"Analyzing drift for checkpoint {step_label}...")
        
        # Handle the state_dict depending on if it's an intermediate dict or final weights
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['generator_state_dict'])
        else:
            gen.load_state_dict(checkpoint)
            
        gen.eval()
        
        # 3. Generate multiple futures for the same context
        with torch.no_grad():
            generated_scaled = gen(batched_context, n_steps=cfg.model.T_len) # (1000, T, d)
            
        generated_returns = generated_scaled.cpu().numpy() * std + mean
        cum_log_returns = np.cumsum(generated_returns, axis=1) # (1000, T, d)
        
        # Model Quantiles
        mod_q05 = np.percentile(cum_log_returns, 5, axis=0)
        mod_q15 = np.percentile(cum_log_returns, 15, axis=0)
        mod_q50 = np.percentile(cum_log_returns, 50, axis=0)
        mod_q85 = np.percentile(cum_log_returns, 85, axis=0)
        mod_q95 = np.percentile(cum_log_returns, 95, axis=0)

        # Calculate annualized drift (assuming 252 trading days per year)
        # Drift = (Cumulative Return at step T / T) * 252
        real_annualized_drift = (real_q50[-1, 0] / cfg.model.T_len) * 252
        model_annualized_drift = (mod_q50[-1, 0] / cfg.model.T_len) * 252
        drift_bias = model_annualized_drift - real_annualized_drift
        
        print(f"--- Drift Bias Analysis (Checkpoint {step_label}) ---")
        print(f"Real Annualized Drift:  {real_annualized_drift:.4f}")
        print(f"Model Annualized Drift: {model_annualized_drift:.4f}")
        print(f"Drift Bias (Model - Real): {drift_bias:.4f}")
        
        # 4. Plotting for Asset 1 (Index 0)
        plt.figure(figsize=(8, 5))
        time_steps = np.arange(1, cfg.model.T_len + 1)
        
        # Plot Model Bands
        plt.fill_between(time_steps, mod_q05[:, 0], mod_q95[:, 0], color='#4C72B0', alpha=0.2, label='Model $Q_{0.05} - Q_{0.95}$')
        plt.fill_between(time_steps, mod_q15[:, 0], mod_q85[:, 0], color='#4C72B0', alpha=0.4, label='Model $Q_{0.15} - Q_{0.85}$')
        
        # Plot Real Bands as lines
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
        print(f"Saved drift analysis plot to {save_path}")

if __name__ == "__main__":
    analyze_cumulative_drift()