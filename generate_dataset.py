from data_loader import JumpDiffusionSimulator, GeometricBrownianMotionSimulator, HestonSimulator
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
    
    print(f"Generating training path (H={cfg.data.H})...")
    if cfg.data.simulator == "Heston":
        train_path, train_vol = sim.simulate(H=cfg.data.H)
        train_vol = torch.log(train_vol + 1e-8)
    else:
        train_path = sim.simulate(H=cfg.data.H) 
        train_vol = None
    
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
    
    if train_vol is not None:
        save_data["train_vol"] = train_vol
        save_data["test_vol"] = test_vol
        
    torch.save(save_data, cfg.train.dataset_path)
    
    print(f"Dataset successfully saved to {cfg.train.dataset_path}")

if __name__ == "__main__":
    generate_and_save_dataset()