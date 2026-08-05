from dataclasses import dataclass, field
import time
import os

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
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0

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
    total_steps: int = 10000
    resample_freq: int = 100
    log_freq: int = 10
    save_freq: int = 10000
    
    # --- Drift Regularization ---
    regularize_drift: bool = True
    drift_control_type: str = "monte_carlo"  # Options: "global", "conditional", "monte_carlo"; monte_carlo is what we want
    lambda_reg: float = 1.0
    mc_samples: int = 10000
    target_drift: float = 0.0
    
    experiment_name: str = "monte_carlo_regularized_most_samples"
    tb_base_dir: str = "../results/runs"
    model_base_dir: str = "../results/checkpoints"
    
    dataset_path: str = None
    tb_dir: str = None
    save_dir: str = None

@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "GBM_wrong_drift" 
    
    # --- Evaluation Override ---
    # Leave empty ("") when training a new model.
    # Paste the exact folder name here when running analysis scripts!
    eval_run_name: str = "20260724_1304_GBM_wrong_drift_monte_carlo_regularized" 
    
    data: BaseDataConfig = field(default_factory=GBMDataConfig) 
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