import torch
from torch.utils.data import Dataset
import math

class JumpDiffusionSimulator:
    """Simulates a multivariate correlated Merton-style Jump Diffusion process."""
    def __init__(self, d: int, mu: float = 0.05, sigma: float = 0.2, 
                 jump_intensity: float = 4.0, jump_mean: float = 0.0, jump_std: float = 0.1,
                 corr_matrix: torch.Tensor = None):
        self.d = d
        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        
        # 1. Setup correlation matrix (defaults to independent/Identity matrix)
        if corr_matrix is None:
            self.corr_matrix = torch.eye(d)
        else:
            assert corr_matrix.shape == (d, d), f"Correlation matrix must be {d}x{d}"
            # Ensure it's a valid correlation matrix (symmetric)
            assert torch.allclose(corr_matrix, corr_matrix.T), "Correlation matrix must be symmetric"
            self.corr_matrix = corr_matrix
            
        # 2. Cholesky decomposition: L @ L.T = corr_matrix
        # We add a tiny jitter (1e-6) to the diagonal to ensure numerical stability during Cholesky
        jitter = torch.eye(d) * 1e-6
        self.L = torch.linalg.cholesky(self.corr_matrix + jitter)

    def simulate(self, H: int, dt: float = 1/252) -> torch.Tensor:
        # --- Correlated Brownian Motion ---
        # Generate independent standard normal random variables
        Z = torch.randn(H, self.d) 
        
        # Induce correlation: Multiply by the Cholesky lower triangular matrix
        # Shape: (H, d) @ (d, d) -> (H, d)
        Z_corr = Z @ self.L.T 
        
        dW = Z_corr * math.sqrt(dt)
        
        # --- Jumps ---
        # Jumps are typically modeled as independent idiosyncratic shocks 
        # (unless you specifically want co-jumps/market-wide shocks)
        n_jumps = torch.poisson(torch.ones(H, self.d) * self.jump_intensity * dt)
        J = n_jumps * torch.randn(H, self.d) * self.jump_std + n_jumps * self.jump_mean
        
        # Log returns
        returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW + J
        
        # Return stationary log-returns, NOT log-prices
        return returns

class GeometricBrownianMotionSimulator:
    """Simulates a multivariate correlated Geometric Brownian Motion (GBM) process."""
    def __init__(self, d: int, mu: float = 0.05, sigma: float = 0.2, 
                 corr_matrix: torch.Tensor = None):
        self.d = d
        self.mu = mu
        self.sigma = sigma
        
        # 1. Setup correlation matrix (defaults to independent/Identity matrix)
        if corr_matrix is None:
            self.corr_matrix = torch.eye(d)
        else:
            assert corr_matrix.shape == (d, d), f"Correlation matrix must be {d}x{d}"
            # Ensure it's a valid correlation matrix (symmetric)
            assert torch.allclose(corr_matrix, corr_matrix.T), "Correlation matrix must be symmetric"
            self.corr_matrix = corr_matrix
            
        # 2. Cholesky decomposition: L @ L.T = corr_matrix
        # We add a tiny jitter (1e-6) to the diagonal to ensure numerical stability during Cholesky
        jitter = torch.eye(d) * 1e-6
        self.L = torch.linalg.cholesky(self.corr_matrix + jitter)

    def simulate(self, H: int, dt: float = 1/252) -> torch.Tensor:
        # --- Correlated Brownian Motion ---
        # Generate independent standard normal random variables
        Z = torch.randn(H, self.d) 
        
        # Induce correlation: Multiply by the Cholesky lower triangular matrix
        # Shape: (H, d) @ (d, d) -> (H, d)
        Z_corr = Z @ self.L.T 
        
        dW = Z_corr * math.sqrt(dt)
        
        # Log returns (Standard Ito calculus drift + diffusion)
        returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW
        
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