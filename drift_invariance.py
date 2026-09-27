import torch
import torch.nn as nn
from SOCK import SOCK # Assuming your SOCK.py is in the same directory

def demonstrate_shift_invariance():
    # 1. Initialize SOCK without augmentations to isolate the convolutions
    print("Initializing SOCK feature extractor...")
    sock = SOCK(
        n_steps=64, 
        n_channels=3, 
        tau=0.1, 
        k=8, 
        mix_dim=256, 
        kernel_len=9, 
        augs=() # Empty tuple to bypass cumsum/diff and hit convolutions directly
    )
    
    # 2. Create a fake generated path (Batch=1, Time=64, Channels=3)
    base_path = torch.randn(1, 64, 3)
    
    # 3. Create a shift vector specifically for the 3 dimensions
    # E.g., Asset 1 gets shifted up by 1000, Asset 2 down by 500, Asset 3 up by 2000
    shift_vector = torch.tensor([[[1000.0, -500.0, 2000.0]]]) 
    shifted_path = base_path + shift_vector
    
    # 4. Extract features
    sock.eval()
    with torch.no_grad():
        features_base = sock(base_path, scale=False)
        features_shifted = sock(shifted_path, scale=False)
        
    # 5. Calculate the difference (MSE Loss)
    mse_loss = torch.nn.functional.mse_loss(features_base, features_shifted)
    
    print("\n--- Results per Asset (Channel) ---")
    # Calculate the mean across the Batch (dim 0) and Time (dim 1), leaving the 3 Channels
    base_means = base_path.mean(dim=(0, 1)).squeeze()
    shifted_means = shifted_path.mean(dim=(0, 1)).squeeze()
    
    for i in range(3):
        print(f"Asset {i+1} -> Base Mean: {base_means[i]:>8.4f} | Shifted Mean: {shifted_means[i]:>9.4f}")
        
    print(f"\nFeature Difference (MSE Loss): {mse_loss.item():.10f}")
    
    if mse_loss.item() < 1e-6:
        print("CONCLUSION: The model is completely blind to the independent vertical shifts across all 3 dimensions.")

if __name__ == "__main__":
    demonstrate_shift_invariance()