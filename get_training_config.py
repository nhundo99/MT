import torch
import os
import glob
from config import Config

def print_saved_train_config(eval_run_name: str):
    """
    Loads a checkpoint for a given run and prints its TrainConfig variables.
    """
    # 1. Initialize config with the eval_run_name to resolve the correct directories
    cfg = Config(eval_run_name=eval_run_name)
    
    # 2. Locate the save directory based on the config logic
    save_dir = cfg.train.save_dir
    
    if not os.path.exists(save_dir):
        print(f"Error: Directory does not exist -> {save_dir}")
        return

    # 3. Find all periodic checkpoints in the directory
    # We avoid generator_final.pt because it doesn't contain the 'config' dictionary
    checkpoint_files = glob.glob(os.path.join(save_dir, "generator_step_*.pt"))
    
    if not checkpoint_files:
        print(f"Error: No periodic checkpoints (generator_step_*.pt) found in {save_dir}")
        return
        
    # 4. Load the first available checkpoint (the config should be identical across steps)
    checkpoint_path = checkpoint_files[0]
    print(f"Loading config from: {checkpoint_path}\n")
    
    # Map to CPU to avoid issues if trained on GPU but evaluated on a CPU-only machine
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 5. Extract and print the TrainConfig
    saved_config = checkpoint.get("config", {})
    train_config = saved_config.get("train", {})
    
    if not train_config:
        print("Error: Could not find 'train' configuration in the saved config dictionary.")
        return
        
    print("-" * 40)
    print(f" TrainConfig for: {eval_run_name}")
    print("-" * 40)
    for key, value in train_config.items():
        print(f"{key}: {value}")
    print("-" * 40)

if __name__ == "__main__":
    # Input your eval_run_name here
    run_name = "20260824_1339_GBM_wrong_drift_curriculum_learning_full"
    print_saved_train_config(run_name)