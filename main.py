import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from SOCK import *
from data_loader import *
from training import *
from utils import *
from config import Config, GBMDataConfig, JDDataConfig, HestonDataConfig

cfg = Config()
seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

writer = SummaryWriter(log_dir=cfg.train.tb_dir)

print(f"Loading dataset from {cfg.train.dataset_path}...")
data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")

if "dataset_config" in data_dict:
    loaded_data_cfg = data_dict["dataset_config"]
    sim_type = loaded_data_cfg.get("simulator")
    
    if sim_type == "GBM":
        cfg.data = GBMDataConfig(**loaded_data_cfg)
    elif sim_type == "Heston":
        cfg.data = HestonDataConfig(**loaded_data_cfg)
    elif sim_type == "JumpDiffusion":
        cfg.data = JDDataConfig(**loaded_data_cfg)
    else:
        raise ValueError(f"Critical Error: Unknown simulator type '{sim_type}' found in dataset_config.")
        
    print(f"Loaded {cfg.data.simulator} dataset parameters for: {cfg.dataset_name}")

hist_path = data_dict["train_path"]

use_volatility = cfg.train.use_volatility

if "train_vol" in data_dict and use_volatility:
    print("Heston dataset detected and volatility usage is ENABLED. Concatenating returns and volatility paths...")
    hist_vol = data_dict["train_vol"]
    # Stacks (H, d) and (H, d) into (H, 2*d)
    hist_data = torch.cat([hist_path, hist_vol], dim=-1)
    actual_channels = cfg.model.d * 2
else:
    if "train_vol" in data_dict and not use_volatility:
        print("Heston dataset detected, but volatility usage is DISABLED. Using returns only...")
    else:
        print("Using standard returns path...")
        
    hist_data = hist_path
    actual_channels = cfg.model.d

cfg.model.d = actual_channels

dataset = FinancialTimeSeriesDataset(hist_data, q=cfg.model.q_len, T=cfg.model.T_len)
dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

sock = SOCK(
    n_steps=cfg.model.q_len + cfg.model.T_len,
    n_channels=cfg.model.d,
    tau=cfg.model.tau,
    k=cfg.model.K,
    mix_dim=cfg.model.M,
    kernel_len=cfg.model.L,
    kernel_width=cfg.model.W,
    augs=("cumsum", "posneg", "diff")  
) 
gen = build_generator(cfg.model).to(device)

print("Starting generator training via SOCK feature matching...")
loss_hist = train_sock_generator(
    generator=gen, 
    sock_extractor=sock, 
    dataloader=dataloader, 
    device=device, 
    cfg=cfg,             
    writer=writer,
    data_mean=dataset.mean,
    data_std=dataset.std
)

writer.close()