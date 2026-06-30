import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from SOCK import *
from data_loader import *
from training import *
from utils import *
from config import Config

# 1. Instantiate the config
cfg = Config()
seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

cfg.train.tb_dir = os.path.join(cfg.train.tb_base_dir, cfg.train.experiment_name)
cfg.train.save_dir = os.path.join(cfg.train.model_base_dir, cfg.train.experiment_name)

# 2. Setup TensorBoard Writer
writer = SummaryWriter(log_dir=cfg.train.tb_dir)

# 3. Load pre-generated dataset (Using config!)
print(f"Loading dataset from {cfg.train.dataset_path}...")
data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
hist_path = data_dict["train_path"]

data_mean = hist_path.mean(dim=0, keepdim=True)
data_std = hist_path.std(dim=0, keepdim=True) + 1e-6
hist_path = (hist_path - data_mean) / data_std

# 4. Create dataset & loader
dataset = FinancialTimeSeriesDataset(hist_path, q=cfg.model.q_len, T=cfg.model.T_len)
dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

# 5. Instantiate models
sock = SOCK(
    n_steps=cfg.model.q_len + cfg.model.T_len,
    n_channels=cfg.model.d,
    tau=cfg.model.tau,
    k=cfg.model.K,
    mix_dim=cfg.model.M,
    kernel_len=cfg.model.L,
    augs=("cumsum", "posneg")  
) 
gen = Generator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim)

# 6. Train
print("Starting generator training via SOCK feature matching...")
loss_hist = train_sock_generator(
    generator=gen, 
    sock_extractor=sock, 
    dataloader=dataloader, 
    device=device, 
    cfg=cfg,             
    writer=writer,
    # --- NEW: Pass scaling factors to the training loop so they can be saved ---
    data_mean=data_mean,
    data_std=data_std
)

writer.close()