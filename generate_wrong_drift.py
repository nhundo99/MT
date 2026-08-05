# generate_dataset.py
from data_loader import JumpDiffusionSimulator, GeometricBrownianMotionSimulator
from config import Config
from dataclasses import asdict
import torch
from utils import seed_everything
import os

def generate_and_save_dataset():
    cfg = Config()
    seed_everything(cfg.seed)
    
    # --- NEW: Convert config list to a tensor ---
    rho = torch.tensor(cfg.data.corr_matrix)
    
    # --- AUTOMATIC ROUTING BASED ON CONFIG ---
    if cfg.data.simulator == "JumpDiffusion":
        print("Initializing Jump Diffusion Simulator...")
        sim = JumpDiffusionSimulator(
            d=cfg.model.d, 
            mu=cfg.data.mu, sigma=cfg.data.sigma,
            jump_intensity=cfg.data.jump_intensity,
            jump_mean=cfg.data.jump_mean, jump_std=cfg.data.jump_std,
            corr_matrix=rho
        )
    elif cfg.data.simulator == "GBM":
        print("Initializing Geometric Brownian Motion Simulator...")
        sim = GeometricBrownianMotionSimulator(
            d=cfg.model.d, 
            mu=cfg.data.mu, sigma=cfg.data.sigma,
            corr_matrix=rho
        )
    else:
        raise ValueError(f"Unknown simulator type: {cfg.data.simulator}")
    
    print(f"Generating candidate training paths to isolate an outlier (H={cfg.data.H})...")
    print(f"Generating candidate training paths to isolate an outlier (H={cfg.data.H})...")
    num_candidates = 1000 
    
    candidate_list = []
    for _ in range(num_candidates):
        candidate_list.append(sim.simulate(H=cfg.data.H))
    
    candidates = torch.stack(candidate_list)
    
    start_vals = candidates[:, 0, :]
    end_vals = candidates[:, -1, :]
    
    if (candidates < 0).any():
        realized_drifts = end_vals - start_vals
    else:
        realized_drifts = torch.log((end_vals + 1e-8) / (start_vals + 1e-8))
    
    mean_drift = realized_drifts.mean(dim=0)
    
    drift_distances = torch.norm(realized_drifts - mean_drift, dim=1)
    
    drift_distances = torch.nan_to_num(drift_distances, nan=-1.0)
    
    outlier_idx = torch.argmax(drift_distances)
    train_path = candidates[outlier_idx]
    
    print(f"-> Selected path {outlier_idx} as train_path.")
    print(f"-> Distance from expected drift: {drift_distances[outlier_idx].item():.4f}")
    
    print(f"Generating {cfg.data.J} out-of-sample continuation paths (N={cfg.data.N})...")
    test_paths = sim.simulate(H=cfg.data.J * cfg.data.N).view(cfg.data.J, cfg.data.N, cfg.model.d) 
    
    os.makedirs("data", exist_ok=True)
    
    torch.save({
        "train_path": train_path,
        "test_paths": test_paths,
        "dataset_config": asdict(cfg.data) 
    }, cfg.train.dataset_path)
    
    print(f"Dataset successfully saved to {cfg.train.dataset_path}")

if __name__ == "__main__":
    generate_and_save_dataset()