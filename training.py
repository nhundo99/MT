import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import run_lightweight_validation
from dataclasses import asdict

def train_sock_generator(
    generator: nn.Module,
    sock_extractor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    cfg,
    writer,
    data_mean: torch.Tensor,
    data_std: torch.Tensor
):
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    dataset = torch.load(cfg.train.dataset_path, map_location="cpu")
    ds_cfg = dataset.get("dataset_config", {})  
    
    # ==========================================
    # --- 50-Step Validation Protocol Setup ---
    # ==========================================
    sim_type = ds_cfg.get("simulator", "GBM")
    true_mu = ds_cfg.get("mu", 0.0)
    true_sigma = ds_cfg.get("sigma", 0.0)
    
    if sim_type == "GBM":
        theoretical_target = true_mu - (0.5 * (true_sigma ** 2))
    elif sim_type == "JumpDiffusion":
        jump_intensity = ds_cfg.get("jump_intensity", 0.0)
        jump_mean = ds_cfg.get("jump_mean", 0.0)
        theoretical_target = true_mu - (0.5 * (true_sigma ** 2)) + (jump_intensity * jump_mean)
    else:
        theoretical_target = true_mu - (0.5 * (true_sigma ** 2))

    val_n_paths = min(1000, dataset["test_paths"].shape[0])
    val_years = 8.0 # 1-year horizon keeps it fast enough for 50 repeated steps
    val_total_steps = int(val_years * 252)
    val_drift_history = []
    
    raw_val_context = dataset["test_paths"][:val_n_paths, :cfg.model.q_len, :].to(device)
    val_context = (raw_val_context - data_mean.to(device)) / data_std.to(device)
    # ==========================================
    
    generator.to(device)
    sock_extractor.to(device)
    
    optimizer = torch.optim.AdamW(generator.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    if cfg.train.curriculum_learning:
        sock_warmup = cfg.train.sock_warm_up
        short_horizon = sock_warmup + cfg.train.short_horizon_steps
    else:
        # Start drift regularization immediately
        sock_warmup = 0
        short_horizon = 0

    if cfg.train.scheduler == "linear":
        warmup_steps = int(cfg.train.warm_up_scheduler * cfg.train.total_steps)
        decay_start = int(cfg.train.decay_start_scheduler * cfg.train.total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif current_step < decay_start:
                return 1.0
            else:
                decay_steps = cfg.train.total_steps - decay_start
                steps_passed = current_step - decay_start
                return max(0.0, float(decay_steps - steps_passed) / float(max(1, decay_steps)))
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=cfg.train.T_0,  
            T_mult=cfg.train.T_mult,   
            eta_min=cfg.train.eta_min 
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.train.scheduler}")
    
    generator.train()

    x_minus_sample, x_plus_sample = next(iter(dataloader))
    real_joined_sample = torch.cat([x_minus_sample, x_plus_sample], dim=1).to(device)
    sock_extractor.fit_input_scales(dataloader, device)
    
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
        
        x_hat_plus = generator(x_minus, n_steps=x_plus.size(1))

        real_joined = torch.cat([x_minus, x_plus], dim=1)
        fake_joined = torch.cat([x_minus, x_hat_plus], dim=1)

        real_feats = sock_extractor(real_joined, scale=True)
        fake_feats = sock_extractor(fake_joined, scale=True)
        
        real_mean = real_feats.mean(dim=0)
        fake_mean = fake_feats.mean(dim=0)
        loss_sock = torch.nn.functional.mse_loss(fake_mean, real_mean, reduction='mean')
        
        loss = loss_sock
        
        # Drift Regularization
        if cfg.train.regularize_drift and step_count >= sock_warmup:
            if cfg.train.curriculum_learning:
                if step_count < short_horizon:
                    alpha = (step_count - sock_warmup) / max(1, short_horizon - sock_warmup)
                    current_lambda = cfg.train.lambda_reg * alpha
                else:
                    current_lambda = cfg.train.lambda_reg
            else:
                current_lambda = cfg.train.lambda_reg
            
            if cfg.train.drift_control_type == "conditional":
                generated_drift = x_hat_plus.mean(dim=1)
                real_drift = x_plus.mean(dim=1)
                
                loss_drift = torch.nn.functional.mse_loss(generated_drift, real_drift, reduction='mean')
                
            elif cfg.train.drift_control_type == "global":
                expected_model_drift = x_hat_plus.mean(dim=(0, 1)) 
                
                target = torch.tensor(cfg.train.target_drift, device=device)
            
                loss_drift = torch.nn.functional.mse_loss(expected_model_drift, target.expand_as(expected_model_drift))
            
            elif cfg.train.drift_control_type == "monte_carlo":
                num_repeats = max(1, cfg.train.mc_samples // x_minus.size(0))
                mc_contexts = x_minus.repeat(num_repeats, 1, 1)

                mc_fake_scaled = generator(mc_contexts, n_steps=x_plus.size(1))

                mc_fake_returns = mc_fake_scaled * data_std.to(device) + data_mean.to(device)

                mc_annualized_drift = mc_fake_returns.mean(dim=(0, 1)) * 252.0
                
                sim_type = ds_cfg.get("simulator", "GBM")
                true_mu = ds_cfg.get("mu", 0.0)
                true_sigma = ds_cfg.get("sigma", 0.0)
                
                if sim_type == "GBM":
                    adjusted_target = true_mu - (0.5 * (true_sigma ** 2))
                    
                elif sim_type == "JumpDiffusion":
                    jump_intensity = ds_cfg.get("jump_intensity", 0.0)
                    jump_mean = ds_cfg.get("jump_mean", 0.0)
                    
                    adjusted_target = true_mu - (0.5 * (true_sigma ** 2)) + (jump_intensity * jump_mean)
                    
                else:
                    raise ValueError(f"Unknown simulator type for target drift calculation: {sim_type}")

                target_tensor = torch.tensor(adjusted_target, device=device).expand_as(mc_annualized_drift)
                loss_drift = torch.nn.functional.mse_loss(mc_annualized_drift, target_tensor)
            
            elif cfg.train.drift_control_type == "long_monte_carlo":
                num_repeats = max(1, cfg.train.mc_samples // x_minus.size(0))
                current_context_reg = x_minus.repeat(num_repeats, 1, 1)
                
                q = current_context_reg.size(1)
                T = x_plus.size(1)

                # Horizon H logic based on curriculum flag
                if cfg.train.curriculum_learning:
                    # Smoothly increase the horizon H
                    base_H = 252
                    # Note: step_count < sock_warmup is always False here due to the outer if statement
                    if step_count < short_horizon:
                        alpha = (step_count - sock_warmup) / max(1, short_horizon - sock_warmup)
                        target_H = cfg.train.long_mc_horizon
                        H = int(base_H + alpha * (target_H - base_H))
                    else:
                        H = cfg.train.long_mc_horizon
                else:
                    # Normal approach: use the full long-term horizon immediately
                    H = cfg.train.long_mc_horizon
                
                mc_fake_scaled_list = []
                steps_generated = 0
                
                while steps_generated < H:
                    next_T = generator(current_context_reg, n_steps=T)
                    mc_fake_scaled_list.append(next_T)
                    
                    steps_generated += T
                    
                    combined_context = torch.cat([current_context_reg, next_T], dim=1)
                    current_context_reg = combined_context[:, -q:, :]
                
                mc_fake_scaled_full = torch.cat(mc_fake_scaled_list, dim=1)[:, :H, :]
                mc_fake_returns_full = mc_fake_scaled_full * data_std.to(device) + data_mean.to(device)
                mc_annualized_drift = mc_fake_returns_full.mean(dim=(0, 1)) * 252.0
                
                sim_type = ds_cfg.get("simulator", "GBM")
                true_mu = ds_cfg.get("mu", 0.0)
                true_sigma = ds_cfg.get("sigma", 0.0)
                
                if sim_type == "GBM":
                    adjusted_target = true_mu - (0.5 * (true_sigma ** 2))
                elif sim_type == "JumpDiffusion":
                    jump_intensity = ds_cfg.get("jump_intensity", 0.0)
                    jump_mean = ds_cfg.get("jump_mean", 0.0)
                    adjusted_target = true_mu - (0.5 * (true_sigma ** 2)) + (jump_intensity * jump_mean)
                elif sim_type == "Heston":
                    theta_var = ds_cfg.get("theta_var", true_sigma ** 2)
                    adjusted_target = true_mu - (0.5 * theta_var)
                else:
                    raise ValueError(f"Unknown simulator type: {sim_type}")

                target_tensor = torch.tensor(adjusted_target, device=device).expand_as(mc_annualized_drift)
                loss_drift = torch.nn.functional.mse_loss(mc_annualized_drift, target_tensor)
                
            else:
                raise ValueError(f"Unknown drift control type: {cfg.train.drift_control_type}")

            # Apply the annealed lambda
            loss = loss + (current_lambda * loss_drift)
        else:
            loss_drift = torch.tensor(0.0, device=device)
            current_lambda = 0.0
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        step_count += 1
        
        # =================================================================
        # --- LIGHTWEIGHT VALIDATION TRIGGER (50 steps before save) ---
        # =================================================================
        modulo = step_count % cfg.train.save_freq
        if step_count > 0 and (modulo == 0 or modulo > cfg.train.save_freq - 50):
            val_drift = run_lightweight_validation(
                generator=generator,
                init_context=val_context,
                data_mean=data_mean.to(device),
                data_std=data_std.to(device),
                q=cfg.model.q_len,
                T_chunk=cfg.model.T_len,
                total_steps=val_total_steps
            )
            val_drift_history.append(val_drift)
        # =================================================================

        if step_count % cfg.train.log_freq == 0:
            writer.add_scalar("Loss/train_total", loss.item(), step_count)
            writer.add_scalar("Loss/train_sock", loss_sock.item(), step_count)
            
            if cfg.train.regularize_drift:
                writer.add_scalar("Loss/train_drift_penalty", loss_drift.item(), step_count)
                
            writer.add_scalar("LearningRate/train", scheduler.get_last_lr()[0], step_count)
        
        if step_count > 0 and step_count % cfg.train.save_freq == 0:
            
            # Compute 50-step stats
            if len(val_drift_history) > 0:
                mean_drift_50 = float(np.mean(val_drift_history))
                stderr_drift_50 = float(np.std(val_drift_history) / np.sqrt(len(val_drift_history)))
                bias_bps_50 = (mean_drift_50 - theoretical_target) * 10000
                
                # Log actual validation metrics to TensorBoard
                writer.add_scalar("Validation/True_Drift_Mean_50steps", mean_drift_50, step_count)
                writer.add_scalar("Validation/True_Drift_Bias_bps_50steps", bias_bps_50, step_count)
                writer.add_scalar("Validation/True_Drift_StdErr_50steps", stderr_drift_50, step_count)
            else:
                mean_drift_50, stderr_drift_50, bias_bps_50 = 0.0, 0.0, 0.0
                
            save_path = os.path.join(cfg.train.save_dir, f"generator_step_{step_count}.pt")
            
            torch.save({
                'step': step_count,
                'generator_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'config': asdict(cfg),
                'data_mean': data_mean,
                'data_std': data_std,
                # Store the validation metrics directly in the checkpoint file!
                'val_metrics_50_steps': {
                    'mean_drift': mean_drift_50,
                    'stderr_drift': stderr_drift_50,
                    'bias_bps': bias_bps_50,
                    'target': theoretical_target
                }
            }, save_path)
            
            print(f"Checkpoint saved to {save_path} | Val Drift Bias: {bias_bps_50:.2f} bps ± {stderr_drift_50 * 10000:.2f} bps")
            
            # Clear history for the next save cycle
            val_drift_history.clear()
        # --------------------------------------
            
    final_save_path = os.path.join(cfg.train.save_dir, "generator_final.pt")
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'data_mean': data_mean,
        'data_std': data_std
    }, final_save_path)
    print(f"Training complete. Final model saved to {final_save_path}")
            
    return loss_history