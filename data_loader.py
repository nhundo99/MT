import torch
from torch.utils.data import Dataset, DataLoader
import math

class JumpDiffusionSimulator:
    """Simulates a multivariate Merton-style Jump Diffusion process."""
    def __init__(self, d: int, mu: float = 0.05, sigma: float = 0.2, 
                 jump_intensity: float = 5.0, jump_mean: float = 0.0, jump_std: float = 0.3):
        self.d = d
        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def simulate(self, H: int, dt: float = 1/252) -> torch.Tensor:
        dW = torch.randn(H, self.d) * math.sqrt(dt)
        n_jumps = torch.poisson(torch.ones(H, self.d) * self.jump_intensity * dt)
        J = n_jumps * torch.randn(H, self.d) * self.jump_std + n_jumps * self.jump_mean
        
        # Log returns
        returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW + J
        
        # Return stationary log-returns, NOT log-prices
        return returns

class FinancialTimeSeriesDataset(Dataset):
    """Extracts sliding windows (context q, horizon T) and standardizes data."""
    def __init__(self, path: torch.Tensor, q: int = 5, T: int = 64):
        self.path = path
        self.q = q
        self.T = T
        self.H, self.d = path.shape
        
        # Standardize using the raw training path 
        self.mean = self.path.mean(dim=0, keepdim=True)
        self.std = self.path.std(dim=0, keepdim=True) + 1e-8
        self.scaled_path = (self.path - self.mean) / self.std

    def __len__(self) -> int:
        return self.H - self.q - self.T + 1

    def __getitem__(self, idx: int):
        # Context (x-) and Real Future (x+)
        x_minus = self.scaled_path[idx : idx + self.q]
        x_plus = self.scaled_path[idx + self.q : idx + self.q + self.T]
        return x_minus, x_plus