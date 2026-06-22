import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from dataclasses import asdict

def train_sock_generator(
    generator: nn.Module, 
    sock_extractor: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: str, 
    cfg,            # <--- Receive config
    writer          # <--- Receive TensorBoard writer
):
    # Use config for training parameters
    os.makedirs(cfg.train.save_dir, exist_ok=True)     
    
    generator.to(device)
    sock_extractor.to(device)
    
    optimizer = torch.optim.AdamW(generator.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    
    warmup_steps = int(0.05 * cfg.train.total_steps)
    decay_start = int(0.30 * cfg.train.total_steps) # Decay starts at the 30% mark to cover the last 70%
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 1. Linear warm up for first 5%
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < decay_start:
            # 2. Stay flat at max learning rate until the decay phase starts
            return 1.0
        else:
            # 3. Linear decay to 0 over the last 70% of steps
            decay_steps = cfg.train.total_steps - decay_start
            steps_passed = current_step - decay_start
            return max(0.0, float(decay_steps - steps_passed) / float(max(1, decay_steps)))
            
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
    
    print(f"Starting training for {cfg.train.total_steps} steps...")
    
    while step_count < cfg.train.total_steps:
        try:
            x_minus, x_plus = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_minus, x_plus = next(data_iter)
            
        x_minus, x_plus = x_minus.to(device), x_plus.to(device)
        
        # The Resampling Trick
        if step_count > 0 and step_count % cfg.train.resample_freq == 0:
            sock_extractor.resample()
            sock_extractor.fit_ft_scales(dataloader, device)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_hat_plus = generator(x_minus, n_steps=x_plus.size(1))
        
        # Joined Segments
        real_joined = torch.cat([x_minus, x_plus], dim=1)
        fake_joined = torch.cat([x_minus, x_hat_plus], dim=1)
        
        # Extract Scaled Features
        real_feats = sock_extractor(real_joined, scale=True)
        fake_feats = sock_extractor(fake_joined, scale=True)
        
        # Mean Squared Error (Summed across the feature dimensions to match L2^2 norm)
        real_mean = real_feats.mean(dim=0)
        fake_mean = fake_feats.mean(dim=0)
        loss = torch.nn.functional.mse_loss(fake_mean, real_mean, reduction='sum')
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        step_count += 1

        if step_count % cfg.train.log_freq == 0:
            writer.add_scalar("Loss/train", loss.item(), step_count)
            writer.add_scalar("LearningRate/train", scheduler.get_last_lr()[0], step_count)
        
        # --- NEW: Model Checkpointing Logic ---
        if step_count % cfg.train.save_freq == 0:
            save_path = os.path.join(cfg.train.save_dir, f"generator_step_{step_count}.pt")
            
            torch.save({
                'step': step_count,
                'generator_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'config': asdict(cfg),
            }, save_path)
            
            print(f"Checkpoint saved to {save_path}")
        # --------------------------------------
            
    # Save a final model at the very end just to be safe
    final_save_path = os.path.join(cfg.train.save_dir, "generator_final.pt")
    torch.save(generator.state_dict(), final_save_path)
    print(f"Training complete. Final model saved to {final_save_path}")
            
    return loss_history