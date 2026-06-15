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
# This will create a folder called 'runs' where all your loss data is stored permanently
writer = SummaryWriter(log_dir=cfg.train.tb_dir)

# 3. Simulate data (using config!)
sim = JumpDiffusionSimulator(d=cfg.model.d)
hist_path = sim.simulate(H=2048)

# 4. Create dataset & loader (using config!)
dataset = FinancialTimeSeriesDataset(hist_path, q=cfg.model.q_len, T=cfg.model.T_len)
dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

# 5. Instantiate models (using config!)
sock = SOCKFeatureMap(
    d=cfg.model.d, 
    T_total=cfg.model.q_len + cfg.model.T_len,
    tau=cfg.model.tau,
    K=cfg.model.K,
    M=cfg.model.M,
    W=cfg.model.W,
    L=cfg.model.L
) 
gen = ConditionalGenerator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim)

# 6. Train
print("Starting generator training via SOCK feature matching...")
# Notice we pass the 'writer' and 'cfg' into the training function now
loss_hist = train_sock_generator(
    generator=gen, 
    sock_extractor=sock, 
    dataloader=dataloader, 
    device=device, 
    cfg=cfg,             # <--- Pass config down
    writer=writer        # <--- Pass TensorBoard writer down
)

# Close the writer when done
writer.close()