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
    
    print(f"Generating training path (H={cfg.data.H})...")
    train_path = sim.simulate(H=cfg.data.H) 
    
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