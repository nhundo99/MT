import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

def normalize_kernel_(w: torch.Tensor, eps: float = 1e-6):
    """Zero-centers and L1-normalizes random kernels[cite: 1138]."""
    w.sub_(w.mean(dim=(-2, -1), keepdim=True))
    w.div_(w.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(eps))

def normalize_proj_(w: torch.Tensor, eps: float = 1e-6):
    """L2-normalizes the random projection matrix[cite: 1134]."""
    w.div_(w.norm(p=2, dim=1, keepdim=True).clamp_min(eps))

class SOCKFeatureMap(nn.Module):
    def __init__(self, d: int, T_total: int, tau: float = 0.1, 
                 K: int = 8, M: int = 256, W: int = 2, L: int = 9):
        super().__init__()
        self.tau = tau
        self.K = K
        self.M = M
        self.W = W
        self.L = L
        
        # Augmentations: int (cumulative sum) and posneg (max(x,0), min(x,0)) [cite: 1117, 1118, 1121]
        self.aug_d = d + d + 2*d 
        
        # Random Projection [cite: 1187]
        self.proj = nn.Linear(self.aug_d, self.M, bias=False)
        
        # Dilations [cite: 1188]
        e_max = math.floor(math.log2((T_total - 1) / (self.L - 1)))
        self.dilations = [2**e for e in range(e_max + 1)]
        
        # Grouped Convolutions [cite: 1191-1199]
        self.groups = self.M // self.W
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.M, 
                      out_channels=self.K * self.groups, 
                      kernel_size=self.L, 
                      padding='same', 
                      dilation=d, 
                      groups=self.groups, 
                      bias=False)
            for d in self.dilations
        ])
        
        # Freeze parameters [cite: 1201]
        for p in self.parameters():
            p.requires_grad = False
            
        self.ft_mean = None
        self.ft_std = None
        self.resample()

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Applies int(X) and posneg(int(X)) [cite: 1117-1121]."""
        x_int = torch.cumsum(x, dim=1)
        x_pos = torch.relu(x_int)
        x_neg = -torch.relu(-x_int)
        return torch.cat([x, x_int, x_pos, x_neg], dim=-1)

    def resample(self):
        """Re-initializes all random parameters [cite: 1203-1208]."""
        self.proj.weight.normal_()
        normalize_proj_(self.proj.weight)
        
        for conv in self.convs:
            conv.weight.normal_()
            normalize_kernel_(conv.weight)

    def fit_ft_scales(self, dataloader: DataLoader, device: str):
        """Fits empirical variance of features for the scaled MSE loss [cite: 1350-1352]."""
        all_feats = []
        for x_minus, x_plus in dataloader:
            x_joined = torch.cat([x_minus.to(device), x_plus.to(device)], dim=1)
            feats = self.forward(x_joined, scale=False)
            all_feats.append(feats)
        
        all_feats = torch.cat(all_feats, dim=0)
        self.ft_mean = all_feats.mean(dim=0, keepdim=True)
        self.ft_std = all_feats.std(dim=0, keepdim=True) + 1e-8
    
    def fit_input_scales(self, x: torch.Tensor):
        """Fits empirical mean and std of the augmented paths[cite: 1129, 1210]."""
        x_aug = self.augment(x)
        # Compute over Batch (dim=0) and Time (dim=1)
        self.input_mean = x_aug.mean(dim=(0, 1), keepdim=True)
        self.input_std = x_aug.std(dim=(0, 1), keepdim=True) + 1e-6

    def forward(self, x: torch.Tensor, scale: bool = False) -> torch.Tensor:
        B, T_steps, _ = x.shape
        x_aug = self.augment(x)
        
        # Apply normalization to augmented features BEFORE projection [cite: 1130, 1217]
        if hasattr(self, 'input_mean') and self.input_mean is not None:
            x_aug = (x_aug - self.input_mean) / self.input_std
            
        # Random projection
        y = self.proj(x_aug) # (B, T, M)
        y = y.permute(0, 2, 1) # (B, M, T)
        
        feats = []
        for conv in self.convs:
            z = conv(y) 
            z = z.view(B, self.groups, self.K, T_steps)
            
            # Soft-deviation pooling (unbiased=False matches the 1/T formula [cite: 1151])
            win_probs = torch.softmax(z / self.tau, dim=2)
            f = torch.std(win_probs, dim=-1, unbiased=False) 
            feats.append(f.view(B, -1))
            
        out = torch.cat(feats, dim=1)
        if scale and self.ft_mean is not None:
            out = (out - self.ft_mean) / self.ft_std
        return out
    
class ConditionalGenerator(nn.Module):
    def __init__(self, d: int, q: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.d = d
        self.q = q
        self.hidden_dim = hidden_dim
        
        # Maps flattened context and initial noise -> Initial hidden state [cite: 1364-1369]
        self.initial_state_gen = nn.Sequential(
            nn.Linear(d * q + d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.proj_in = nn.Linear(d, 2 * hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # Residual Gated Noise Injection [cite: 1373-1376]
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.proj_out = nn.Linear(hidden_dim, d)

    def forward(self, x_minus: torch.Tensor, T: int = 64) -> torch.Tensor:
        B = x_minus.size(0)
        device = x_minus.device
        
        # Initial State Generation [cite: 1383-1385]
        init_noise = torch.randn(B, self.d, device=device)
        context_flat = x_minus.flatten(start_dim=1)
        h0_in = torch.cat([context_flat, init_noise], dim=-1)
        h0 = self.initial_state_gen(h0_in).unsqueeze(0) # (1, B, hidden_dim)
        
        # Decode step-by-step noise [cite: 1387-1390]
        step_noise = torch.randn(B, T, self.d, device=device)
        z_proj = self.proj_in(step_noise)
        z, z_skip = z_proj.chunk(2, dim=-1)
        
        h, _ = self.rnn(F.silu(z), h0)
        
        # Residual injection and output projection [cite: 1391-1392]
        h = h + self.alpha * self.gate(h) * z_skip
        x_hat_plus = self.proj_out(h)
        
        return x_hat_plus