# sweep_drift.py
import os
import csv
import torch
from dataclasses import asdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config, GBMDataConfig
from data_loader import GeometricBrownianMotionSimulator, FinancialTimeSeriesDataset
from SOCK import SOCK, Generator
from training import train_sock_generator
from utils import seed_everything
from drift_analysis import evaluate_numerical_drift

def run_drift_experiment():
    # The different population drifts you want to test for your thesis
    mu_values = [-0.15, 0.0, 0.15, 0.30]
    
    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    for mu in mu_values:
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT: POPULATION MU = {mu}")
        print(f"{'='*60}")
        
        # 1. Build the configuration for this specific mu
        cfg = Config()
        cfg.data = GBMDataConfig(mu=mu, sigma=0.20) 
        
        # Name the dataset and place it in the drift_sweep subfolder
        mu_str = str(mu).replace(".", "p").replace("-", "neg")
        cfg.dataset_name = f"drift_sweep/GBM_Mu_{mu_str}"
        
        # Ensure the experiment name is consistent so we can resume/skip properly
        cfg.eval_run_name = f"Sweep_Run_GBM_Mu_{mu_str}"
        
        # Re-initialize config paths
        cfg.__post_init__()
        
        # Lower steps for testing if needed (default is 100000 in your config)
        cfg.train.total_steps = 10000 
        
        # 2. DATA GENERATION STEP (If it does not exist)
        os.makedirs(os.path.dirname(cfg.train.dataset_path), exist_ok=True)
        
        if not os.path.exists(cfg.train.dataset_path):
            print(f"[*] Data not found. Generating dataset for Mu = {mu}...")
            seed_everything(cfg.seed)
            rho = torch.tensor(cfg.data.corr_matrix)
            sim = GeometricBrownianMotionSimulator(
                d=cfg.model.d, mu=cfg.data.mu, sigma=cfg.data.sigma, corr_matrix=rho
            )
            train_path = sim.simulate(H=cfg.data.H) 
            test_paths = sim.simulate(H=cfg.data.J * cfg.data.N).view(cfg.data.J, cfg.data.N, cfg.model.d) 
            
            torch.save({
                "train_path": train_path,
                "test_paths": test_paths,
                "dataset_config": asdict(cfg.data) 
            }, cfg.train.dataset_path)
        else:
            print(f"[*] Data already exists at {cfg.train.dataset_path}. Skipping generation.")

        # 3. TRAINING STEP (If final model does not exist)
        final_model_path = os.path.join(cfg.train.save_dir, "generator_final.pt")
        
        if not os.path.exists(final_model_path):
            print(f"[*] Model not found. Training generator for Mu = {mu}...")
            seed_everything(cfg.seed)
            
            # Load Data
            data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
            hist_path = data_dict["train_path"]
            
            # Scale Data
            data_mean = hist_path.mean(dim=0, keepdim=True)
            data_std = hist_path.std(dim=0, keepdim=True) + 1e-6
            scaled_hist_path = (hist_path - data_mean) / data_std
            
            # Setup DataLoader
            dataset = FinancialTimeSeriesDataset(scaled_hist_path, q=cfg.model.q_len, T=cfg.model.T_len)
            dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
            
            # Initialize Models
            sock = SOCK(n_steps=cfg.model.q_len + cfg.model.T_len, n_channels=cfg.model.d, tau=cfg.model.tau, k=cfg.model.K, mix_dim=cfg.model.M, kernel_len=cfg.model.L, augs=("cumsum", "posneg")) 
            gen = Generator(d=cfg.model.d, q=cfg.model.q_len, hidden_dim=cfg.model.hidden_dim)
            
            writer = SummaryWriter(log_dir=cfg.train.tb_dir)
            train_sock_generator(gen, sock, dataloader, device, cfg, writer, data_mean, data_std)
            writer.close()
        else:
            print(f"[*] Trained model already exists at {final_model_path}. Skipping training.")

        # 4. EVALUATION STEP
        print(f"[*] Evaluating drift metrics for Mu = {mu}...")
        results = evaluate_numerical_drift(cfg, ckpt_step="final")
        
        if results:
            all_results.append(results)
            print("\n--- Drift Analysis Results ---")
            for k, v in results.items():
                print(f"  {k}: {v:.6f}")
            print("------------------------------\n")

    # 5. SAVE CSV
    if all_results:
        csv_file = "drift_analysis_thesis_results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[SUCCESS] All experiments complete! Results saved to {csv_file}")

if __name__ == "__main__":
    run_drift_experiment()