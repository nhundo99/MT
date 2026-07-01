from dataclasses import dataclass, field
import time
import os

# 1. BASE CLASS: Everything both simulators share
@dataclass
class BaseDataConfig:
    simulator: str = "Unknown" # We will overwrite this in subclasses
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

# 2. GBM CLASS: Just inherits the base stuff!
@dataclass
class GBMDataConfig(BaseDataConfig):
    simulator: str = "GBM"
    # Doesn't need any extra parameters

# 3. JD CLASS: Adds the jump-specific parameters
@dataclass
class JDDataConfig(BaseDataConfig):
    simulator: str = "JumpDiffusion"
    jump_intensity: float = 4.0
    jump_mean: float = 0.0
    jump_std: float = 0.1

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

@dataclass
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    total_steps: int = 10000
    resample_freq: int = 100
    log_freq: int = 10          
    save_freq: int = 10000      
    
    experiment_name: str = "baseline" 
    
    tb_base_dir: str = "../results/runs"
    model_base_dir: str = "../results/checkpoints"
    
    # These will be auto-filled, so we default them to None
    dataset_path: str = None
    tb_dir: str = None 
    save_dir: str = None

@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "GBM_v1" 
    
    # --- NEW: Evaluation Override ---
    # Leave empty ("") when training a new model.
    # Paste the exact folder name here when running analysis scripts!
    eval_run_name: str = "20260701_1016_GBM_v1_baseline" 
    
    data: BaseDataConfig = field(default_factory=GBMDataConfig) 
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # 1. Build the dataset path
        self.train.dataset_path = f"data/{self.dataset_name}.pt"
        
        # 2. The Override Logic
        if self.eval_run_name != "":
            # If you provided a name, use it EXACTLY as-is (no new timestamp)
            self.train.experiment_name = self.eval_run_name
        else:
            # Otherwise, generate a fresh timestamp for a new training run
            timestamp = time.strftime("%Y%m%d_%H%M")
            self.train.experiment_name = f"{timestamp}_{self.dataset_name}_{self.train.experiment_name}"
        
        # 3. Automatically build the final save directories
        self.train.tb_dir = os.path.join(self.train.tb_base_dir, self.train.experiment_name)
        self.train.save_dir = os.path.join(self.train.model_base_dir, self.train.experiment_name)