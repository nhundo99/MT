# numerical_drift_analysis.py
import torch
import numpy as np
import os

from SOCK import Generator
from data_loader import FinancialTimeSeriesDataset

def evaluate_numerical_drift(cfg, ckpt_step="final"):
    """
    Calculates numerical drift metrics (Theoretical, Train, Test, Generated)
    and returns them as a dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # 1. Load Data
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    train_path = data_dict["train_path"]
    test_paths = data_dict["test_paths"] # Shape: (J, N, d)
    
    # We evaluate on the first channel (Asset 1)
    d_idx = 0 
    
    # 2. Calculate Theoretical & Realized Ground Truth Drifts
    # From your simulator: return = (mu - 0.5*sigma^2)*dt + sigma*dW
    # The annualized theoretical expected log-return is:
    theoretical_annualized_drift = (cfg.data.mu - 0.5 * (cfg.data.sigma**2))
    
    train_returns_flat = train_path[:, d_idx].numpy()
    realized_train_drift = np.mean(train_returns_flat) * 252
    
    test_returns_flat = test_paths[:, :, d_idx].numpy().flatten()
    realized_test_drift = np.mean(test_returns_flat) * 252

    # 3. Setup Generator and load weights
    gen = Generator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim).to(device)
    save_dir = cfg.train.save_dir
    
    ckpt_name = "generator_final.pt" if ckpt_step == "final" else f"generator_step_{ckpt_step}.pt"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return None
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'generator_state_dict' in checkpoint:
        gen.load_state_dict(checkpoint['generator_state_dict'])
    else:
        gen.load_state_dict(checkpoint)
    gen.eval()
    
    # Extract dataset scalers
    dataset = FinancialTimeSeriesDataset(train_path, q=cfg.model.q_len, T=cfg.model.T_len)
    data_mean_tensor = dataset.mean.to(device)
    data_std_tensor = dataset.std.to(device)

    # 4. Generate Futures from out-of-sample contexts
    # We use independent out-of-sample continuations as contexts for a low-variance estimate
    raw_contexts = test_paths[:, :cfg.model.q_len, :].to(device) 
    scaled_contexts = (raw_contexts - data_mean_tensor) / data_std_tensor
    
    with torch.no_grad():
        generated_scaled = gen(scaled_contexts, n_steps=cfg.model.T_len)
        
    generated_returns = generated_scaled.cpu().numpy() * dataset.std.cpu().numpy() + dataset.mean.cpu().numpy()
    generated_returns_flat = generated_returns[:, :, d_idx].flatten()
    
    model_generated_drift = np.mean(generated_returns_flat) * 252

    # 5. Compile Results
    results = {
        "Target_Mu": cfg.data.mu,
        "Theoretical_Drift": theoretical_annualized_drift,
        "Realized_Train_Drift": realized_train_drift,
        "Realized_Test_Drift": realized_test_drift,
        "Model_Generated_Drift": model_generated_drift,
        "Bias_vs_Theoretical": model_generated_drift - theoretical_annualized_drift,
        "Bias_vs_Train": model_generated_drift - realized_train_drift
    }
    
    return results