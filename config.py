# config.py
from dataclasses import dataclass, field

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
    total_steps: int = 100000
    resample_freq: int = 100
    log_freq: int = 10          # Save stats to TensorBoard
    save_freq: int = 10000      # Save the heavy model weights
    
    experiment_name: str = "sock_experiment_2"
    tb_base_dir: str = "../results/runs"
    model_base_dir: str = "../results/checkpoints"
    
    tb_dir: str = "" 
    save_dir: str = ""

@dataclass
class Config:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)