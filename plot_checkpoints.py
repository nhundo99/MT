import torch
import os
import matplotlib.pyplot as plt

from SOCK import Generator
from data_loader import FinancialTimeSeriesDataset
from utils import plot_full_autoregressive_rollout, seed_everything
from config import Config

def visualize_checkpoints():
    cfg = Config()
    seed_everything(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Load pre-generated dataset
    print(f"Loading dataset from {cfg.train.dataset_path}...")
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    hist_path = data_dict["train_path"]
    
    dataset = FinancialTimeSeriesDataset(hist_path, q=cfg.model.q_len, T=cfg.model.T_len)

    gen = Generator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim).to(device)
    save_dir = os.path.join(cfg.train.model_base_dir, cfg.train.experiment_name)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    checkpoints_to_plot = [10000, 50000, 100000]

    for step in checkpoints_to_plot:
        ckpt_path = os.path.join(save_dir, f"generator_step_{step}.pt")
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"\n--- Loading intermediate checkpoint: Step {step} ---")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['generator_state_dict'])
        else:
            gen.load_state_dict(checkpoint)
        
        pdf_path = os.path.join(plot_dir, f"rollout_step_{step}.pdf")
        plot_full_autoregressive_rollout(
            generator=gen,
            dataset=dataset,
            device=device,
            path_tensor=hist_path,
            save_path=pdf_path 
        )

    final_path = os.path.join(save_dir, "generator_final.pt")
    if os.path.exists(final_path):
        print(f"\n--- Loading Final Model ---")
        checkpoint = torch.load(final_path, map_location=device)
        
        # --- FIXED: Extract the state_dict from the dictionary safely ---
        if 'generator_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['generator_state_dict'])
        else:
            gen.load_state_dict(checkpoint)
        # ----------------------------------------------------------------
        
        final_pdf_path = os.path.join(plot_dir, "rollout_final.pdf")
        plot_full_autoregressive_rollout(
            generator=gen,
            dataset=dataset,
            device=device,
            path_tensor=hist_path,
            save_path=final_pdf_path 
        )

if __name__ == "__main__":
    visualize_checkpoints()