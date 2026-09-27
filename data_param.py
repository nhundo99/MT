import torch

# 1. Load the dataset file
file_path = "data/Jump_Only_Sweep_05.pt"
dataset = torch.load(file_path, map_location="cpu")

# 2. Extract the configuration dictionary
if "dataset_config" in dataset:
    config = dataset["dataset_config"]
    
    print("--- Dataset Generation Parameters ---")
    for key, value in config.items():
        print(f"{key}: {value}")
        
    # You can access specific values just like any dictionary:
    # print(f"The simulation ran for {config['H']} steps.")
    # print(f"The drift (mu) was {config['mu']}.")

else:
    print("No configuration found in this file.")
    config = {}

print("\n-------------------------------------")

# 3. Extract train_path and calculate empirical drift
if "train_path" in dataset:
    train_path = dataset["train_path"]
    
    # H is the number of steps. For step-wise log returns, total time is H * dt.
    H = train_path.shape[0]
    dt = config.get("dt", 1 / 252.0) 
    total_years = H * dt  
    
    # Sum the returns to get the total cumulative log-return
    total_drift = train_path.sum(dim=0)
    
    # Annualize it to get empirical geometric drift
    annualized_geom_drift = total_drift / total_years
    
    print(f"Train Path Shape: {train_path.shape}")
    print(f"Annualized Empirical Geometric Drift:\n{annualized_geom_drift.numpy()}")
    
    # 4. Compare it back to the exact 'mu' in your config
    if "mu" in config and "sigma" in config:
        sigma = torch.tensor(config["sigma"])
        
        # Add back the volatility drag (Ito's Lemma)
        estimated_mu = annualized_geom_drift + 0.5 * (sigma**2)
        target_geom_drift = torch.tensor(config['mu']) - 0.5 * (sigma**2)
        
        print(f"\nEstimated Arithmetic mu:\n{estimated_mu.numpy()}")
        print(f"Target Arithmetic mu from config:\n{config['mu']}")
        print(f"Target Geometric drift from config (mu - 0.5*sigma^2):\n{target_geom_drift.numpy()}")

else:
    print("No train_path found in this file.")