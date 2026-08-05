import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from SOCK import *
from config import Config
from utils import seed_everything

def analyze_cumulative_drift(checkpoints_to_plot=[10000, 50000, 100000]):
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    train_path = data_dict["train_path"]
    test_paths = data_dict["test_paths"] # Shape: (J, N, d)

    gen = build_generator(cfg.model).to(device)
    save_dir = os.path.join(cfg.train.model_base_dir, cfg.train.experiment_name)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    num_samples = test_paths.size(0)
    
    checkpoints = [(step, f"generator_step_{step}.pt") for step in checkpoints_to_plot]
    checkpoints.append(("Final", "generator_final.pt"))

    for step_label, ckpt_name in checkpoints:
        ckpt_path = os.path.join(save_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"Analyzing drift for checkpoint {step_label}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if 'data_mean' in checkpoint and 'data_std' in checkpoint:
            data_mean_tensor = checkpoint['data_mean'].to(device)
            data_std_tensor = checkpoint['data_std'].to(device)
        else:
            print("Warning: data_mean and data_std not found in checkpoint. Falling back to dynamic calculation.")
            data_mean_tensor = train_path.mean(dim=0, keepdim=True).to(device)
            data_std_tensor = train_path.std(dim=0, keepdim=True).to(device) + 1e-6

        data_mean_np = data_mean_tensor.cpu().numpy()
        data_std_np = data_std_tensor.cpu().numpy()
        
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
        
        if 'generator_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['generator_state_dict'])
        else:
            gen.load_state_dict(checkpoint)
            
        gen.eval()
        
        batch_size = 2048
        generated_scaled_list = []
        
        with torch.no_grad():
            for i in range(0, len(scaled_contexts), batch_size):
                batch_contexts = scaled_contexts[i : i + batch_size]
                gen_batch = gen(batch_contexts, n_steps=T)
                generated_scaled_list.append(gen_batch)
                
        generated_scaled = torch.cat(generated_scaled_list, dim=0)
        
        generated_returns = generated_scaled.cpu().numpy() * data_std_np + data_mean_np
        
        cum_log_returns = np.cumsum(generated_returns, axis=1)
        real_cum_returns = np.cumsum(real_returns, axis=1) 
        
        real_returns = test_paths[:, cfg.model.q_len:cfg.model.q_len + cfg.model.T_len, :].numpy()
        real_cum_returns = np.cumsum(real_returns, axis=1)
        
        real_q05 = np.percentile(real_cum_returns, 5, axis=0)
        real_q15 = np.percentile(real_cum_returns, 15, axis=0)
        real_q50 = np.percentile(real_cum_returns, 50, axis=0)
        real_q85 = np.percentile(real_cum_returns, 85, axis=0)
        real_q95 = np.percentile(real_cum_returns, 95, axis=0)
        
        mod_q05 = np.percentile(cum_log_returns, 5, axis=0)
        mod_q15 = np.percentile(cum_log_returns, 15, axis=0)
        mod_q50 = np.percentile(cum_log_returns, 50, axis=0)
        mod_q85 = np.percentile(cum_log_returns, 85, axis=0)
        mod_q95 = np.percentile(cum_log_returns, 95, axis=0)

        real_mean_final = np.mean(real_cum_returns[:, -1, 0])
        model_mean_final = np.mean(cum_log_returns[:, -1, 0])

        real_annualized_drift = (real_mean_final / cfg.model.T_len) * 252
        model_annualized_drift = (model_mean_final / cfg.model.T_len) * 252
        drift_bias = model_annualized_drift - real_annualized_drift
        
        print(f"--- Drift Bias Analysis (Checkpoint {step_label}) ---")
        print(f"Real Annualized Drift (Mean):  {real_annualized_drift:.4f}")
        print(f"Model Annualized Drift (Mean): {model_annualized_drift:.4f}")
        print(f"Drift Bias (Model - Real): {drift_bias:.4f}")
        
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
        
        save_path = os.path.join(plot_dir, f"quantile_analysis_step_{step_label}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    analyze_cumulative_drift()