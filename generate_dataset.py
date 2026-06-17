import torch
import os
from data_loader import JumpDiffusionSimulator
from config import Config
from utils import seed_everything

def generate_and_save_dataset():
    cfg = Config()
    seed_everything(cfg.seed)
    
    # Paper specifications for synthetic data
    H = 2048  # Length of the single training path
    J = 2048  # Number of independent out-of-sample continuations
    N = 2048  # Length of each out-of-sample continuation

    rho = torch.tensor([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, -0.5],
        [0.3, -0.5, 1.0]
    ])
    
    sim = JumpDiffusionSimulator(d=cfg.model.d, corr_matrix=rho)
    
    print(f"Generating training path (H={H})...")
    train_path = sim.simulate(H=H)  # Shape: (2048, d)
    
    print(f"Generating {J} out-of-sample continuation paths (N={N})...")
    # Because jump diffusion returns are stationary, we can simulate all steps 
    # and reshape to get J independent paths of length N
    test_paths = sim.simulate(H=J * N).view(J, N, cfg.model.d) 
    
    os.makedirs("data", exist_ok=True)
    save_path = "data/jump_diffusion_data.pt"
    
    torch.save({
        "train_path": train_path,
        "test_paths": test_paths
    }, save_path)
    
    print(f"Dataset successfully saved to {save_path}")

if __name__ == "__main__":
    generate_and_save_dataset()