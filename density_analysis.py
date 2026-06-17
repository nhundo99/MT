import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

from SOCK import Generator
from data_loader import FinancialTimeSeriesDataset
from config import Config
from utils import seed_everything

def plot_probability_densities(checkpoints_to_plot=[10000, 50000, 100000]):
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. Load data
    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    train_path = data_dict["train_path"]
    test_paths = data_dict["test_paths"] # Shape: (J, N, d) = (2048, 2048, d)
    
    # We still use the dataset object to access the train path's standardizer
    dataset = FinancialTimeSeriesDataset(train_path, q=cfg.model.q_len, T=cfg.model.T_len)
    mean = dataset.mean.squeeze().cpu().numpy()
    std = dataset.std.squeeze().cpu().numpy()
    
    print("Extracting Ground Truth Returns from independent continuations...")
    # The true marginal is perfectly represented by flattening the out-of-sample test paths
    real_returns_flat = test_paths[:, :, 0].numpy().flatten()

    kde_real = gaussian_kde(real_returns_flat)
    x_grid = np.linspace(np.min(real_returns_flat), np.max(real_returns_flat), 1000)
    pdf_real = kde_real(x_grid)

    # 2. Setup Generator
    gen = Generator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim).to(device)
    save_dir = os.path.join(cfg.train.model_base_dir, cfg.train.experiment_name)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    num_samples = test_paths.size(0) # Exactly J=2048 to match the real continuations
    
    # To evaluate out-of-sample conditional generation correctly, we condition on the very end 
    # of the training path (the exact boundary before out-of-sample data begins).
    context = dataset.scaled_path[-cfg.model.q_len:].unsqueeze(0).to(device)
    batched_context = context.repeat(num_samples, 1, 1)

    checkpoints = [(step, f"generator_step_{step}.pt") for step in checkpoints_to_plot]
    checkpoints.append(("Final", "generator_final.pt"))

    for step_label, ckpt_name in checkpoints:
        ckpt_path = os.path.join(save_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"Plotting probability density for checkpoint {step_label}...")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['generator_state_dict'])
        else:
            gen.load_state_dict(checkpoint)
        gen.eval()
        
        # 3. Generate futures
        with torch.no_grad():
            generated_scaled = gen(batched_context, n_steps=cfg.model.T_len)
            
        generated_returns = generated_scaled.cpu().numpy() * std + mean
        generated_returns_flat = generated_returns[:, :, 0].flatten()
        
        kde_gen = gaussian_kde(generated_returns_flat)
        pdf_gen = kde_gen(x_grid) 

        # 4. Plotting
        plt.figure(figsize=(8, 5))
        plt.fill_between(x_grid, pdf_real, color='gray', alpha=0.3)
        plt.plot(x_grid, pdf_real, color='black', linestyle='--', linewidth=2, label='Real Data Density')
        plt.fill_between(x_grid, pdf_gen, color='#4C72B0', alpha=0.3)
        plt.plot(x_grid, pdf_gen, color='#4C72B0', linewidth=2, label=f'Model Density (Step {step_label})')

        plt.title(f"Marginal Return Probability Density - Step {step_label}")
        plt.xlabel("Log Returns (Asset 1)")
        plt.ylabel("Density")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(plot_dir, f"density_analysis_step_{step_label}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved density plot to {save_path}")

if __name__ == "__main__":
    plot_probability_densities()