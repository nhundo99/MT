import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader

def train_sock_generator(
    generator: nn.Module, 
    sock_extractor: nn.Module, 
    dataloader: DataLoader, 
    device: str, 
    total_steps: int = 100000, 
    resample_freq: int = 100,
    save_freq: int = 10000,                  # <--- New parameter
    save_dir: str = "../results/models"      # <--- Relative path based on your structure
):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)     # <--- Create folder if it doesn't exist
    
    generator.to(device)
    sock_extractor.to(device)
    
    # Optimizer and Scheduler 
    optimizer = torch.optim.AdamW(generator.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Warmup + Decay schedule over total_steps
    warmup_steps = int(0.05 * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    generator.train()
    # 1. Extract a single batch to fit the input scales for the augmented paths
    x_minus_sample, x_plus_sample = next(iter(dataloader))
    real_joined_sample = torch.cat([x_minus_sample, x_plus_sample], dim=1).to(device)
    sock_extractor.fit_input_scales(real_joined_sample)
    
    # 2. Fit feature scales as before
    sock_extractor.fit_ft_scales(dataloader, device)
    
    loss_history = []
    step_count = 0
    data_iter = iter(dataloader)
    
    print(f"Starting training for {total_steps} steps...")
    
    while step_count < total_steps:
        try:
            x_minus, x_plus = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_minus, x_plus = next(data_iter)
            
        x_minus, x_plus = x_minus.to(device), x_plus.to(device)
        
        # The Resampling Trick
        if step_count > 0 and step_count % resample_freq == 0:
            sock_extractor.resample()
            sock_extractor.fit_ft_scales(dataloader, device)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_hat_plus = generator(x_minus, T=x_plus.size(1))
        
        # Joined Segments
        real_joined = torch.cat([x_minus, x_plus], dim=1)
        fake_joined = torch.cat([x_minus, x_hat_plus], dim=1)
        
        # Extract Scaled Features
        real_feats = sock_extractor(real_joined, scale=True)
        fake_feats = sock_extractor(fake_joined, scale=True)
        
        # Mean Squared Error on Batch Means
        real_mean = real_feats.mean(dim=0)
        fake_mean = fake_feats.mean(dim=0)
        loss = torch.nn.functional.mse_loss(fake_mean, real_mean)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        step_count += 1
        
        # --- NEW: Model Checkpointing Logic ---
        if step_count % save_freq == 0:
            save_path = os.path.join(save_dir, f"generator_step_{step_count}.pt")
            
            # Saving a dictionary is best practice so you can resume training later if needed
            torch.save({
                'step': step_count,
                'generator_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
            }, save_path)
            
            print(f"Checkpoint saved to {save_path}")
        # --------------------------------------
        
        if step_count % 1000 == 0:
            print(f"Step {step_count}/{total_steps} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    # Save a final model at the very end just to be safe
    final_save_path = os.path.join(save_dir, "generator_final.pt")
    torch.save(generator.state_dict(), final_save_path)
    print(f"Training complete. Final model saved to {final_save_path}")
            
    return loss_history