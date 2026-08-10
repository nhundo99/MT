import torch
from torch.utils.data import Dataset
import math

class JumpDiffusionSimulator:
    """Simulates a multivariate correlated Merton-style Jump Diffusion process."""
    def __init__(self, d: int, mu: float = 0.00, sigma: float = 0.0, 
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
            assert torch.allclose(corr_matrix, corr_matrix.T), "Correlation matrix must be symmetric"
            self.corr_matrix = corr_matrix
            
        # 2. Cholesky decomposition: L @ L.T = corr_matrix
        jitter = torch.eye(d) * 1e-6
        self.L = torch.linalg.cholesky(self.corr_matrix + jitter)

    def simulate(self, H: int, dt: float = 1/252) -> torch.Tensor:
        Z = torch.randn(H, self.d) 
        
        # Shape: (H, d) @ (d, d) -> (H, d)
        Z_corr = Z @ self.L.T 
        
        dW = Z_corr * math.sqrt(dt)
        
        # independent shocks
        n_jumps = torch.poisson(torch.ones(H, self.d) * self.jump_intensity * dt)
        J = torch.sqrt(n_jumps) * torch.randn(H, self.d) * self.jump_std + n_jumps * self.jump_mean
        
        returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW + J
        
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
            assert torch.allclose(corr_matrix, corr_matrix.T), "Correlation matrix must be symmetric"
            self.corr_matrix = corr_matrix
            
        # 2. Cholesky decomposition: L @ L.T = corr_matrix
        jitter = torch.eye(d) * 1e-6
        self.L = torch.linalg.cholesky(self.corr_matrix + jitter)

    def simulate(self, H: int, dt: float = 1/252) -> torch.Tensor:
        Z = torch.randn(H, self.d) 
        
        # Shape: (H, d) @ (d, d) -> (H, d)
        Z_corr = Z @ self.L.T 
        
        dW = Z_corr * math.sqrt(dt)
        
        returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW
        
        return returns

import math
import torch

class HestonSimulator:
    """Simulates a multivariate correlated Heston model using a Second-Order Alfonsi scheme for volatility."""
    def __init__(self, d: int, mu: float = 0.05, 
                 kappa: float = 2.0, theta_var: float = 0.04, xi: float = 0.2, 
                 rho: float = -0.5, v0: float = 0.04,
                 corr_matrix: torch.Tensor = None):
        self.d = d
        self.mu = mu
        self.kappa = kappa
        self.theta_var = theta_var
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        
        # 1. Setup target correlation matrix
        if corr_matrix is None:
            target_corr = torch.eye(d)
        else:
            assert corr_matrix.shape == (d, d), f"Correlation matrix must be {d}x{d}"
            assert torch.allclose(corr_matrix, corr_matrix.T), "Correlation matrix must be symmetric"
            target_corr = corr_matrix
            
        # 2. Adjust for correlation dilution
        if abs(self.rho) < 1.0:
            adj_corr = (target_corr - torch.eye(d)) / (1.0 - self.rho**2) + torch.eye(d)
            
            evals, evecs = torch.linalg.eigh(adj_corr)
            if torch.any(evals < 0):
                evals = torch.clamp(evals, min=1e-8)
                adj_corr = evecs @ torch.diag(evals) @ evecs.T
                
                # Re-normalize to ensure exactly 1.0 on the diagonal
                diag_inv = 1.0 / torch.sqrt(torch.diag(adj_corr))
                adj_corr = diag_inv.unsqueeze(1) * adj_corr * diag_inv.unsqueeze(0)
        else:
            adj_corr = torch.eye(d)

        # 3. Cholesky decomposition
        jitter = torch.eye(d) * 1e-6
        self.L = torch.linalg.cholesky(adj_corr + jitter)

    def _alfonsi_psi(self, t: float) -> float:
        """Helper to compute the psi function"""
        if self.kappa == 0:
            return t
        else:
            return (1 - math.exp(-self.kappa * t)) / self.kappa

    def simulate(self, H: int, dt: float = 1/252, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        psi = self._alfonsi_psi(dt)
        psi_half = self._alfonsi_psi(dt / 2)
        
        term_A = (self.xi**2 / 4) - (self.kappa * self.theta_var)
        term_B = (self.kappa * self.theta_var) - (self.xi**2 / 4)
        
        if self.xi**2 <= 4 * self.kappa * self.theta_var:
            K_2 = 0.0
        else:
            K_2 = math.exp(self.kappa * dt / 2) * (
                term_A * psi_half +
                (math.sqrt(math.exp(self.kappa * dt / 2) * term_A * psi_half) + (self.xi / 2) * math.sqrt(3 * dt))**2
            )
            
        exp_k_dt = math.exp(-self.kappa * dt)
        exp_k_dt_half = math.exp(-self.kappa * dt / 2)

        V = torch.full((self.d,), self.v0, device=device)
        returns = torch.zeros((H, self.d), device=device)
        variances = torch.zeros((H, self.d), device=device)
        
        Z_assets = torch.randn(H, self.d, device=device) @ self.L.T.to(device)

        for t in range(H):
            V_next = torch.zeros_like(V)
            
            # Alfonsi Scheme
            mask = V >= K_2
            
            if mask.any():
                V_m = V[mask]
                U_y = torch.rand(V_m.shape, device=device)
                Y = torch.where(U_y < 1/6, -math.sqrt(3),
                      torch.where(U_y < 5/6, 0.0, math.sqrt(3)))
                w = math.sqrt(dt) * Y
                
                inner = term_B * psi_half + exp_k_dt_half * V_m
                inner = torch.clamp(inner, min=0.0)
                
                phi = exp_k_dt_half * (torch.sqrt(inner) + (self.xi / 2) * w)**2 + term_B * psi_half
                V_next[mask] = phi

            mask_inv = ~mask
            if mask_inv.any():
                V_m = V[mask_inv]
                u_1 = V_m * exp_k_dt + self.kappa * self.theta_var * psi
                u_2 = u_1**2 + self.xi**2 * psi * ((self.kappa * self.theta_var * psi / 2) + V_m * exp_k_dt)
                
                Delta = torch.clamp(1.0 - (u_1**2 / u_2), min=0.0) 
                pi_val = (1.0 - torch.sqrt(Delta)) / 2.0
                
                x_1 = u_1 / (2.0 * pi_val)
                x_2 = u_1 / (2.0 * (1.0 - pi_val))
                
                U_z = torch.rand(V_m.shape, device=device)
                Z = torch.where(U_z < pi_val, x_1, x_2)
                V_next[mask_inv] = Z

            # Euler Scheme
            V_current_safe = torch.clamp(V, min=0.0)

            if self.rho != 0.0:
                # Eliminate division by 0
                vol_increment = V_next - V - self.kappa * (self.theta_var - V) * dt
                martingale_correction = (self.rho / self.xi) * vol_increment
                
                orthogonal_noise = math.sqrt(1.0 - self.rho**2) * torch.sqrt(V_current_safe) * Z_assets[t] * math.sqrt(dt)
                
                returns[t] = (self.mu - 0.5 * V_current_safe) * dt + martingale_correction + orthogonal_noise
            else:
                dW_asset = Z_assets[t] * math.sqrt(dt)
                returns[t] = (self.mu - 0.5 * V_current_safe) * dt + torch.sqrt(V_current_safe) * dW_asset
            
            variances[t] = V
            V = V_next
            
        return returns, variances

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