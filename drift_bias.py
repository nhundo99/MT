import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOCK import build_generator
from config import Config
from utils import seed_everything

def plot_3d_monte_carlo_drift_bias(experiment_name="no_regularization", n_paths=1024, H_gen=2048):
    cfg = Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the outlier dataset
    dataset_path = "data/GBM_wrong_drift.pt"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
    
    data_dict = torch.load(dataset_path, map_location="cpu")
    train_path = data_dict["train_path"]
    test_paths = data_dict["test_paths"]
    
    # Calculate theoretical geometric drift
    true_mu = 0.09
    true_sigma = 0.20
    theoretical_geom_drift = true_mu - (0.5 * true_sigma**2)  # 0.07 annualized
    dt = 1 / 252

    # 2. Load the unregularized model checkpoint
    model_dir = os.path.join(cfg.train.model_base_dir, experiment_name)
    ckpt_path = os.path.join(model_dir, "generator_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    gen = build_generator(cfg.model).to(device)
    gen.load_state_dict(checkpoint['generator_state_dict'])
    gen.eval()

    data_mean = checkpoint['data_mean'].to(device)
    data_std = checkpoint['data_std'].to(device)

    # 3. Autoregressively generate out-of-sample paths (Monte Carlo)
    q = cfg.model.q_len
    T_chunk = cfg.model.T_len
    
    # Prime the generator with N out-of-sample contexts
    raw_init_context = test_paths[:n_paths, :q, :].to(device)
    current_context = (raw_init_context - data_mean) / data_std

    generated_chunks = []
    steps_done = 0
    
    with torch.no_grad():
        while steps_done < H_gen:
            next_chunk = gen(current_context, n_steps=T_chunk)
            generated_chunks.append(next_chunk)
            steps_done += T_chunk
            current_context = next_chunk[:, -q:, :]
            
    generated_scaled = torch.cat(generated_chunks, dim=1)[:, :H_gen, :]
    generated_returns = generated_scaled * data_std + data_mean
    generated_returns = generated_returns.cpu().numpy()
    
    # 4. Compute Cumulative Returns & Means
    train_cum = np.cumsum(train_path.numpy(), axis=0)[:H_gen, :]
    gen_cum = np.cumsum(generated_returns, axis=1)
    mean_gen_cum = np.mean(gen_cum, axis=0) # Mean across the N paths
    
    time_steps = np.arange(1, H_gen + 1)
    theoretical_cum = theoretical_geom_drift * dt * time_steps

    # 5. Plot the results for all 3 assets
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f"The Drift Bias Problem: Mean Generated Trajectories ({n_paths} Paths) vs. Sample Drift", fontsize=16)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    empirical_drifts = [0.145, 0.160, -0.005] # Approximate values based on dataset setup

    for i in range(3):
        ax = axes[i]
        
        # Plot Theoretical Drift
        ax.plot(time_steps, theoretical_cum, label=f'Target Expectation (0.07)', 
                color='black', linestyle='--', linewidth=2.5)
        
        # Plot Empirical Training Path
        ax.plot(time_steps, train_cum[:, i], label=f'Training Path (≈ {empirical_drifts[i]})', 
                color=colors[i], alpha=0.5, linewidth=2.0)
        
        # Plot Mean Generated Path
        ax.plot(time_steps, mean_gen_cum[:, i], label=f'Mean Generated Path', 
                color='red', linestyle='-.', linewidth=2.5)
        
        ax.set_title(f"Asset {i+1}", fontsize=14)
        ax.set_xlabel("Time Steps (Trading Days)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Cumulative Log Return", fontsize=12)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "drift_bias_3d_monte_carlo.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Saved 3D drift bias illustration to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_3d_monte_carlo_drift_bias(experiment_name="20260906_1113_GBM_wrong_drift_no_regularization", n_paths=1024)