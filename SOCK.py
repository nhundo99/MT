import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# Augmentation Primitives
# -------------------------------------------------------------------------
class CumSumAug(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # It adds the same number of channels it receives
        self.n_add_channels = in_channels 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, torch.cumsum(x, dim=1)], dim=-1)

class DiffAug(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.n_add_channels = in_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = torch.cat([torch.zeros_like(x[:, :1, :]), torch.diff(x, dim=1)], dim=1)
        return torch.cat([x, diff], dim=-1)

class PosNegAug(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # PosNeg adds two new channels (positive and negative parts) for every input channel
        self.n_add_channels = 2 * in_channels 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, torch.relu(x), -torch.relu(-x)], dim=-1)

AUGMENTATIONS = {
    "cumsum": CumSumAug, 
    "diff": DiffAug, 
    "posneg": PosNegAug
}

# -------------------------------------------------------------------------
# Normalization Helpers
# -------------------------------------------------------------------------
def normalize_kernel_(w: torch.Tensor, eps: float = 1e-6) -> None:
    """Zero-centers and L1-normalizes random kernels."""
    w.sub_(w.mean(dim=(-2, -1), keepdim=True)) # dim -2 is kernel width and dim -1 is kernel length
    w.div_(w.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(eps))

def normalize_proj_(w: torch.Tensor, eps: float = 1e-6) -> None:
    """L2-normalizes the random projection matrix."""
    w.div_(w.norm(p=2, dim=1, keepdim=True).clamp_min(eps))


# -------------------------------------------------------------------------
# SOCK Feature Map
# -------------------------------------------------------------------------
class SOCK(nn.Module):
    def __init__(
        self,
        n_steps: int, 
        n_channels: int,
        tau: float = 0.1,
        k: int = 8,
        mix_dim: int = 256,
        kernel_len: int = 9,
        augs: tuple[str, ...] = ("cumsum",),
    ) -> None:
        super().__init__()
        self.tau, self.k = tau, k
        
        # Dynamically build augmentations to compute final channel dimension
        aug_modules = []
        current_channels = n_channels
        for aug in augs:
            module = AUGMENTATIONS[aug](current_channels)
            aug_modules.append(module)
            current_channels += module.n_add_channels
            
        self.augs = nn.Sequential(*aug_modules)
        
        # Exactly matches pseudocode: n_channels + sum(aug.n_add_channels for aug in self.augs)
        self.proj = nn.Linear(n_channels + sum(aug.n_add_channels for aug in self.augs), mix_dim, bias=False)
        
        emax = math.log2((n_steps - 1) / (kernel_len - 1))
        self.dilations = (2 ** torch.arange(int(emax) + 1)).int()
        
        kernel_width = 2
        self.convs = nn.ModuleList()
        for d in self.dilations:
            self.convs.append(
                nn.Conv1d(
                    in_channels=mix_dim,
                    out_channels=k * (mix_dim // kernel_width),
                    kernel_size=kernel_len,
                    padding='same',
                    dilation=d.item(), 
                    groups=mix_dim // kernel_width,
                    bias=False,
                )
            )
            
        for p in self.parameters(): # SOCK's parameters are untrained, only need gradient w.r.t input x
            p.requires_grad = False
            
        # Variables initialized to None for scale fitting (mimics pseudocode logic)
        self.ft_mean = None
        self.ft_scl = None
        self.input_mean = None
        self.input_scl = None
        
        self.resample()

    def resample(self) -> None:
        self.proj.weight.normal_()
        normalize_proj_(self.proj.weight)
        for conv in self.convs:
            conv.weight.normal_()
            normalize_kernel_(conv.weight)

    def fit_input_scales(self, dataloader: DataLoader, device: str) -> None: 
        all_x_aug = []
        with torch.no_grad(): # Saves memory
            for x_minus, x_plus in dataloader:
                x_joined = torch.cat([x_minus.to(device), x_plus.to(device)], dim=1)
                x_aug = self.augs(x_joined)
                all_x_aug.append(x_aug)
                
        all_x_aug = torch.cat(all_x_aug, dim=0)
        self.input_mean = all_x_aug.mean(dim=(0, 1), keepdim=True)
        self.input_scl = all_x_aug.std(dim=(0, 1), keepdim=True) + 1e-6

    def fit_ft_scales(self, dataloader: DataLoader, device: str) -> None: 
        # fits (ft_mean, ft_scl); call after every resample
        # Note: Adapted from pseudocode stub to use your working dataloader loop
        all_feats = []
        for x_minus, x_plus in dataloader:
            x_joined = torch.cat([x_minus.to(device), x_plus.to(device)], dim=1)
            feats = self.forward(x_joined, scale=False)
            all_feats.append(feats)
            
        all_feats = torch.cat(all_feats, dim=0)
        self.ft_mean = all_feats.mean(dim=0, keepdim=True)
        self.ft_scl = all_feats.std(dim=0, keepdim=True) + 1e-8

    def pool(self, z: torch.Tensor) -> torch.Tensor:
        # soft-deviation pooling
        return torch.std(torch.softmax(z / self.tau, dim=2), dim=-1, unbiased=False)

    def forward(self, x: torch.Tensor, scale: bool = True) -> torch.Tensor:
        # x.shape = (B, n_steps, n_channels)
        x = self.augs(x)
        
        if self.input_mean is not None and self.input_scl is not None:
            x = (x - self.input_mean) / self.input_scl
            
        x = self.proj(x).permute(0, 2, 1) # (B, mix_dim, n_steps)
        
        feats = []
        for conv in self.convs: # loop over dilations
            z = conv(x) # (B, n_groups * k, n_steps), where n_groups = mix_dim // kernel_width
            z = z.view(x.size(0), -1, self.k, x.size(-1)) # (B, n_groups, k, n_steps)
            f = self.pool(z) # (B, n_groups, k)
            feats.append(f.view(x.size(0), -1)) # (B, n_groups * k)
            
        out = torch.cat(feats, dim=1)
        
        # Scaling toggle added so your `fit_ft_scales` doesn't throw a NoneType error
        if scale and self.ft_mean is not None and self.ft_scl is not None:
            return (out - self.ft_mean) / self.ft_scl
        return out

# -------------------------------------------------------------------------
# Conditional Generator
# -------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, d: int, hidden_dim: int = 128, q: int = 5) -> None:
        super().__init__()
        self.noise_dim = d
        self.initial_noise_dim = d
        
        # maps (flattened context c, initial noise) -> initial hidden state h0
        self.initial_state_generator = nn.Sequential(
            nn.Linear(d * q + self.initial_noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.proj_in = nn.Linear(self.noise_dim, 2 * hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # residual noise injection: add a gated noise stream after the GRU
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Sigmoid()
        )
        self.proj_out = nn.Linear(hidden_dim, d)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Stabilizing initialization: ensures generated paths start with small variance
        nn.init.orthogonal_(self.proj_out.weight, gain=0.01)
        nn.init.zeros_(self.proj_out.bias)
        
        # Optional: scale down the initial state MLP
        nn.init.orthogonal_(self.initial_state_generator[-1].weight, gain=0.1)

    def forward(self, c: torch.Tensor, n_steps: int = 64) -> torch.Tensor:
        # c.shape = (B, q, d)
        
        # sample initial hidden state h0 conditionally on context c
        initial_noise = torch.randn((c.size(0), self.initial_noise_dim), device=c.device)
        h0_in = torch.cat((c.flatten(start_dim=1), initial_noise), dim=-1)
        h0 = self.initial_state_generator(h0_in)
        
        # decode per-step noise with: linear proj -> SiLU -> GRU
        z = torch.randn((c.size(0), n_steps, self.noise_dim), device=c.device)
        z, z_skip = self.proj_in(z).chunk(2, dim=-1)
        
        h, _ = self.rnn(F.silu(z), h0.unsqueeze(0))
        
        # optional output noise injection
        h = h + self.alpha * self.gate(h) * z_skip
        return self.proj_out(h)