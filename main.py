from SOCK import *
from data_loader import *
from training import *
from utils import *

seed_everything(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 1. Simulate data (d=2) [cite: 1247-1250]
sim = JumpDiffusionSimulator(d=3)
hist_path = sim.simulate(H=2048)

# 2. Create dataset & loader
q_len, T_len = 5, 64
dataset = FinancialTimeSeriesDataset(hist_path, q=q_len, T=T_len)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

# 3. Instantiate models
# T_total is q + T
sock = SOCKFeatureMap(d=3, T_total=q_len + T_len) 
gen = ConditionalGenerator(d=3, q=q_len, hidden_dim=128)

SAVE_DIR = "../results/test"

# 4. Train
print("Starting generator training via SOCK feature matching...")
loss_hist = train_sock_generator(
    generator=gen, 
    sock_extractor=sock, 
    dataloader=dataloader, 
    device=device, 
    total_steps=100,     # <--- Changed to match the paper exactly
    resample_freq=100,      # Resample every 100 steps as per the paper
    save_freq=100,
    save_dir=SAVE_DIR
)