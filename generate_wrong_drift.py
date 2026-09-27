from data_loader import (
    JumpDiffusionSimulator, 
    GeometricBrownianMotionSimulator, 
    HestonSimulator
)
from config import Config
from dataclasses import asdict
import torch
from utils import seed_everything
import os

def generate_and_save_dataset():
    cfg = Config()
    seed_everything(cfg.seed)
    
    asset_corr = torch.tensor(cfg.data.corr_matrix)
    
    if cfg.data.simulator == "JumpDiffusion":
        print("Initializing Jump Diffusion Simulator...")
        sim = JumpDiffusionSimulator(
            d=cfg.model.d, 
            mu=cfg.data.mu, sigma=cfg.data.sigma,
            jump_intensity=cfg.data.jump_intensity,
            jump_mean=cfg.data.jump_mean, jump_std=cfg.data.jump_std,
            corr_matrix=asset_corr
        )
    elif cfg.data.simulator == "GBM":
        print("Initializing Geometric Brownian Motion Simulator...")
        sim = GeometricBrownianMotionSimulator(
            d=cfg.model.d, 
            mu=cfg.data.mu, sigma=cfg.data.sigma,
            corr_matrix=asset_corr
        )
    elif cfg.data.simulator == "Heston":
        print("Initializing Heston Simulator...")
        sim = HestonSimulator(
            d=cfg.model.d, 
            mu=cfg.data.mu, 
            kappa=cfg.data.kappa, 
            theta_var=cfg.data.theta_var, 
            xi=cfg.data.xi, 
            rho=cfg.data.rho, 
            v0=cfg.data.v0,
            corr_matrix=asset_corr
        )
    else:
        raise ValueError(f"Unknown simulator type: {cfg.data.simulator}")
    
    print(f"Generating candidate training paths to isolate an outlier (H={cfg.data.H})...")
    num_candidates = 1000
    
    candidate_list = []
    vol_candidate_list = []
    
    for _ in range(num_candidates):
        if cfg.data.simulator == "Heston":
            returns, vol = sim.simulate(H=cfg.data.H)
            candidate_list.append(returns)
            # Track the volatility and apply the log transform immediately
            vol_candidate_list.append(torch.log(vol + 1e-8))
        else:
            candidate_list.append(sim.simulate(H=cfg.data.H))
    
    # Stack all candidate returns to calculate the outlier
    candidates = torch.stack(candidate_list)
    if cfg.data.simulator == "Heston":
        vol_candidates = torch.stack(vol_candidate_list)
    
    # --- Outlier selection logic based on returns ---
    realized_drifts = candidates.sum(dim=1)
    mean_drift = realized_drifts.mean(dim=0)
    dt = 1 / 252
    T = cfg.data.H * dt
    
    annualized_drift = mean_drift / T
    
    drift_distances = torch.norm(realized_drifts - annualized_drift, dim=1)
    drift_distances = torch.nan_to_num(drift_distances, nan=-1.0)
    
    outlier_idx = torch.argmax(drift_distances)
    
    # Select the specific outlier returns AND its corresponding volatility
    train_path = candidates[outlier_idx]
    if cfg.data.simulator == "Heston":
        train_vol = vol_candidates[outlier_idx]
    else:
        train_vol = None
    
    outlier_drift_formatted = [f"{x:.4f}" for x in realized_drifts[outlier_idx].tolist()]
    mean_drift_formatted = [f"{x:.4f}" for x in annualized_drift.tolist()]
    
    print(f"-> Selected path {outlier_idx} as train_path.")
    print(f"-> Drift from selected path: {outlier_drift_formatted}")
    print(f"-> Mean Drift: {mean_drift_formatted}")
    print(f"-> Distance from expected drift: {drift_distances[outlier_idx].item():.4f}")
    
    print(f"Generating {cfg.data.J} out-of-sample continuation paths (N={cfg.data.N})...")
    if cfg.data.simulator == "Heston":
        test_paths_flat, test_vol_flat = sim.simulate(H=cfg.data.J * cfg.data.N)
        test_paths = test_paths_flat.view(cfg.data.J, cfg.data.N, cfg.model.d)
        test_vol = torch.log(test_vol_flat + 1e-8).view(cfg.data.J, cfg.data.N, cfg.model.d)
    else:
        test_paths = sim.simulate(H=cfg.data.J * cfg.data.N).view(cfg.data.J, cfg.data.N, cfg.model.d) 
        test_vol = None
    
    os.makedirs("data", exist_ok=True)
    
    save_data = {
        "train_path": train_path,
        "test_paths": test_paths,
        "dataset_config": asdict(cfg.data) 
    }
    
    # Save the volatility fields if they exist (Heston model)
    if train_vol is not None:
        save_data["train_vol"] = train_vol
        save_data["test_vol"] = test_vol
        
    torch.save(save_data, cfg.train.dataset_path)
    
    print(f"Dataset successfully saved to {cfg.train.dataset_path}")

if __name__ == "__main__":
    generate_and_save_dataset()