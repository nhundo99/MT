import torch
import matplotlib.pyplot as plt
import os
import random
import numpy as np

def plot_training_loss(loss_history: list):
    """Visualizes the scaled MSE feature-matching loss over time."""
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, alpha=0.8, color='blue', linewidth=1.5)
    plt.title("SOCK Feature Matching Loss during Training")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Scaled MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_full_autoregressive_rollout(
    generator: torch.nn.Module, 
    dataset, 
    device: str, 
    path_tensor: torch.Tensor,
    save_path: str = None
):
    """
    Plots the full simulated ground truth path alongside a full-length 
    generated path created by autoregressively stitching 64-step chunks together.
    """
    generator.eval()
    
    H, d = path_tensor.shape
    q = dataset.q
    T = dataset.T
    
    # 1. Start with the very first 'q' steps of the scaled real data
    current_context = dataset.scaled_path[:q].unsqueeze(0).to(device) # Shape: (1, q, d)
    
    # We will store our generated chunks here (starting with the initial real context)
    generated_chunks = [current_context.squeeze(0).cpu().numpy()]
    
    steps_generated = q
    
    # 2. Autoregressively generate chunks of length T until we hit H
    with torch.no_grad():
        while steps_generated < H:
            # Generate the next T steps
            next_T = generator(current_context, n_steps=T) # Shape: (1, T, d)
            
            # Store the generated chunk
            generated_chunks.append(next_T.squeeze(0).cpu().numpy())
            steps_generated += T
            
            # Update context for the next loop: use the last 'q' steps of the generated chunk
            current_context = next_T[:, -q:, :]
            
    # 3. Stitch chunks together and trim to exact length H
    full_generated_scaled = np.concatenate(generated_chunks, axis=0)[:H]
    
    # 4. Un-standardize the generated path back to the original returns scale
    mean = dataset.mean.squeeze().cpu().numpy()
    std = dataset.std.squeeze().cpu().numpy()
    full_generated_returns = full_generated_scaled * std + mean
    
    # 5. Convert returns to log prices via cumulative sum
    generated_log_prices = np.cumsum(full_generated_returns, axis=0)
    real_log_prices = np.cumsum(path_tensor.cpu().numpy(), axis=0)
    
    # 6. Plotting
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    for i in range(d):
        plt.plot(real_log_prices[:, i], label=f'Real Asset {i+1}', 
                 linewidth=2.0, alpha=0.8, color=colors[i])
                 
    for i in range(d):
        plt.plot(generated_log_prices[:, i], label=f'Generated Asset {i+1}', 
                 linewidth=1.5, alpha=0.9, linestyle='--', color=colors[i])
        
    plt.title(f"Full Path Comparison: Ground Truth vs. Autoregressive Generation (H={H} steps)")
    plt.xlabel("Time Steps (Trading Days)")
    plt.ylabel("Log Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        # bbox_inches='tight' ensures that no axis labels or legends get cut off in the PDF
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

def seed_everything(seed: int = 42):
    """Locks all random seeds for perfect reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False