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
    mu: float = 0.09
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

    augs: tuple = ("cumsum", "posneg", "diff")

@dataclass
class TrainConfig:
    scheduler: str = "linear" # cosine, linear

    # cosine scheduler
    T_0: int = 10000
    T_mult: int = 2
    eta_min: float = 1e-6

    warm_up_scheduler: float = 0.05
    decay_start_scheduler: float = 0.30

    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    total_steps: int = 50000
    resample_freq: int = 100
    log_freq: int = 10
    save_freq: int = 10000

    use_volatility: bool = False
    
    # --- Drift Regularization ---
    curriculum_learning: bool = True
    regularize_drift: bool = True
    drift_control_type: str = "long_monte_carlo"  # Options: "global", "conditional", "monte_carlo", "long_monte_carlo"
    long_mc_horizon: int = 2048
    lambda_reg: float = 50.0
    mc_samples: int = 256
    sock_warm_up: int = 40000
    short_horizon_steps: int = 10000
    
    experiment_name: str = "baseline_curriculum_short"
    tb_base_dir: str = "../results/runs"
    model_base_dir: str = "../results/checkpoints"
    
    dataset_path: str = None
    tb_dir: str = None
    save_dir: str = None

@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "JD_wrong_drift" 
    
    # --- Evaluation Override ---
    # Leave empty ("") when training a new model.
    # Paste the exact folder name here when running analysis scripts!
    eval_run_name: str = "20260919_1951_JD_wrong_drift_baseline_curriculum_short" 
    
    data: BaseDataConfig = field(default_factory=JDDataConfig) 
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