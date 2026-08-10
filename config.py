from dataclasses import dataclass, field
import time
import os
from typing import Optional

# 1. BASE CLASS: Everything both simulators share
@dataclass
class BaseDataConfig:
    simulator: str = "Unknown"
    H: int = 2048
    J: int = 2048
    N: int = 2048
    mu: float = 0.05
    sigma: float = 0.2
    corr_matrix: list = field(default_factory=lambda: [
        [1.0, 0.6, 0.3],
        [0.6, 1.0, -0.5],
        [0.3, -0.5, 1.0]
    ])

# 2. GBM CLASS
@dataclass
class GBMDataConfig(BaseDataConfig):
    simulator: str = "GBM"

# 3. JD CLASS: jump-specific parameters
@dataclass
class JDDataConfig(BaseDataConfig):
    simulator: str = "JumpDiffusion"
    jump_intensity: float = 4.0
    jump_mean: float = 0.01
    jump_std: float = 0.05

@dataclass
class HestonDataConfig(BaseDataConfig):
    simulator: str = "Heston"
    kappa: float = 2.0
    xi: float = 0.3
    rho: float = -0.7
    
    # Will be set Dynamically
    v0: Optional[float] = None
    theta_var: Optional[float] = None

    def __post_init__(self):
        # Dynamically set initial variance based on BaseDataConfig's sigma
        if self.v0 is None:
            self.v0 = self.sigma ** 2
            
        # Dynamically set long-run variance based on BaseDataConfig's sigma
        if self.theta_var is None:
            self.theta_var = self.sigma ** 2

@dataclass
class ModelConfig:
    d: int = 3
    q_len: int = 5
    T_len: int = 64
    hidden_dim: int = 128
    tau: float = 0.1
    K: int = 8
    M: int = 256
    W: int = 2
    L: int = 9
    generator_type: str = "standard"  # if changes to generator are needed. Options: "standard"

@dataclass
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    total_steps: int = 100000
    resample_freq: int = 100
    log_freq: int = 10
    save_freq: int = 10000

    use_volatility: bool = False
    
    # --- Drift Regularization ---
    regularize_drift: bool = False
    drift_control_type: str = "long_monte_carlo"  # Options: "global", "conditional", "monte_carlo", "long_monte_carlo"
    long_mc_horizon: int = 1004
    lambda_reg: float = 5.0
    mc_samples: int = 1000
    target_drift: float = 0.0
    
    experiment_name: str = "baseline_no_vol"
    tb_base_dir: str = "../results/runs"
    model_base_dir: str = "../results/checkpoints"
    
    dataset_path: str = None
    tb_dir: str = None
    save_dir: str = None

@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "Hestonv1" 
    
    # --- Evaluation Override ---
    # Leave empty ("") when training a new model.
    # Paste the exact folder name here when running analysis scripts!
    eval_run_name: str = "20260809_1447_Hestonv1_baseline_no_vol" 
    
    data: BaseDataConfig = field(default_factory=HestonDataConfig) 
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        self.train.dataset_path = f"data/{self.dataset_name}.pt"
        
        if self.eval_run_name != "":
            self.train.experiment_name = self.eval_run_name
        else:
            timestamp = time.strftime("%Y%m%d_%H%M")
            self.train.experiment_name = f"{timestamp}_{self.dataset_name}_{self.train.experiment_name}"
        
        self.train.tb_dir = os.path.join(self.train.tb_base_dir, self.train.experiment_name)
        self.train.save_dir = os.path.join(self.train.model_base_dir, self.train.experiment_name)