import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from SOCK import Generator
from config import Config
from utils import seed_everything

def analyze_jump_frequency(threshold_multiplier=3.0):
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. Load Data
    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    train_path = data_dict["train_path"]
    test_paths = data_dict["test_paths"] # Shape: (J, N, d)
    
    # Extract Real Returns (Only looking at the prediction horizon T_len)
    real_returns = test_paths[:, :cfg.model.T_len, :].numpy() 
    num_samples = real_returns.shape[0]

    # 2. Load Checkpoint
    save_dir = os.path.join(cfg.train.model_base_dir, cfg.train.experiment_name)
    ckpt_path = os.path.join(save_dir, "generator_final.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Could not find checkpoint at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    data_mean_tensor = checkpoint['data_mean'].to(device)
    data_std_tensor = checkpoint['data_std'].to(device)
    data_mean_np = data_mean_tensor.cpu().numpy()
    data_std_np = data_std_tensor.cpu().numpy()

    # 3. Setup Generator
    gen = Generator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim).to(device)
    if 'generator_state_dict' in checkpoint:
        gen.load_state_dict(checkpoint['generator_state_dict'])
    else:
        gen.load_state_dict(checkpoint)
    gen.eval()

    # 4. Generate Fake Paths
    # Condition on the perfectly scaled end of the training path
    raw_context = train_path[-cfg.model.q_len:].unsqueeze(0).to(device)
    scaled_context = (raw_context - data_mean_tensor) / data_std_tensor
    batched_context = scaled_context.repeat(num_samples, 1, 1)

    print("Generating paths...")
    with torch.no_grad():
        generated_scaled = gen(batched_context, n_steps=cfg.model.T_len)
    
    # Unscale back to raw returns
    generated_returns = generated_scaled.cpu().numpy() * data_std_np + data_mean_np

    # 5. Jump Detection Logic (Using Robust Statistics)
    print("Calculating Jump Frequencies...")
    
    # Calculate the median and robust standard deviation (MAD) of the REAL returns
    # This ensures our threshold isn't artificially inflated by the real jumps
    median = np.median(real_returns, axis=(0, 1), keepdims=True)
    mad = np.median(np.abs(real_returns - median), axis=(0, 1), keepdims=True)
    robust_std = mad / 0.6745  # Standard conversion from MAD to Normal Std Dev

    # A jump is any return whose absolute deviation exceeds 'k' robust standard deviations
    threshold = threshold_multiplier * robust_std

    real_jumps = np.abs(real_returns - median) > threshold
    fake_jumps = np.abs(generated_returns - median) > threshold

    # We will analyze Asset 0 (the primary asset log-returns)
    asset_idx = 0
    real_jumps_asset = real_jumps[:, :, asset_idx]
    fake_jumps_asset = fake_jumps[:, :, asset_idx]

    # Sum the number of jumps across the T=64 time steps for each path
    real_jumps_per_path = real_jumps_asset.sum(axis=1)
    fake_jumps_per_path = fake_jumps_asset.sum(axis=1)

    print(f"\n--- Jump Frequency Analysis (Threshold: {threshold_multiplier}x Robust Std) ---")
    print(f"Continuous Volatility (Robust Std): {robust_std[0, 0, asset_idx]:.6f}")
    print(f"Jump Threshold Value:               {threshold[0, 0, asset_idx]:.6f}")
    print(f"-----------------------------------------------------------")
    print(f"Real Data - Total Jumps: {real_jumps_asset.sum()}, Avg per path: {real_jumps_per_path.mean():.2f}")
    print(f"Fake Data - Total Jumps: {fake_jumps_asset.sum()}, Avg per path: {fake_jumps_per_path.mean():.2f}")

    # 6. Plotting the Distributions
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    
    # Determine bins based on the max jumps observed
    max_jumps = max(real_jumps_per_path.max(), fake_jumps_per_path.max())
    bins = np.arange(-0.5, max_jumps + 1.5, 1)

    plt.hist(real_jumps_per_path, bins=bins, alpha=0.5, label='Real Jumps', color='black', density=True, rwidth=0.8)
    plt.hist(fake_jumps_per_path, bins=bins, alpha=0.5, label='Generated Jumps', color='#4C72B0', density=True, rwidth=0.8)

    plt.title(f"Distribution of Jump Counts per Path (T={cfg.model.T_len})\nThreshold = {threshold_multiplier}x $\sigma_{{robust}}$")
    plt.xlabel("Number of Jumps in 64-Day Path")
    plt.ylabel("Proportion of Paths")
    plt.xticks(np.arange(0, max_jumps + 1, 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(plot_dir, "jump_frequency_analysis.pdf")
    plt.savefig(save_path, format='pdf')
    print(f"\nSaved jump frequency histogram to {save_path}")

if __name__ == "__main__":
    # You can tweak this multiplier. 3.0 is a standard statistical anomaly threshold.
    analyze_jump_frequency(threshold_multiplier=3.0)