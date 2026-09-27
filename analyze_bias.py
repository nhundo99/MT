import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SOCK import build_generator
from config import Config
from utils import seed_everything

def stress_test_bias(run_name: str, horizon: int = 252):
    print(f"\n{'='*55}")
    print(f" LONG-HORIZON AUTOREGRESSIVE STRESS TEST: {run_name}")
    print(f"{'='*55}")

    cfg = Config()
    cfg.eval_run_name = run_name 
    seed_everything(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # 1. Load Data
    data_dict = torch.load(cfg.train.dataset_path, map_location="cpu")
    test_paths = data_dict["test_paths"]
    
    # 2. Load Unregularized Model
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    ckpt_path = os.path.join(save_dir, "generator_final.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    data_mean_tensor = checkpoint['data_mean'].to(device)
    data_std_tensor = checkpoint['data_std'].to(device)
    
    gen = build_generator(cfg.model).to(device)
    gen.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    gen.eval()
    
    # 3. Extract non-overlapping initial contexts
    N_len = test_paths.size(1)
    q = cfg.model.q_len
    
    all_raw_contexts = []
    # Step by q to get completely independent starting conditions
    for t in range(0, N_len - q, q):
        all_raw_contexts.append(test_paths[:, t : t + q, :])
        
    raw_contexts = torch.cat(all_raw_contexts, dim=0).to(device)
    scaled_contexts = (raw_contexts - data_mean_tensor) / data_std_tensor
    
    print(f"Executing stress test over {horizon} steps using {len(scaled_contexts)} independent paths...")
    
    # 4. Chunked Autoregressive Generation Loop
    batch_size = 2048
    generated_scaled_list = []
    
    T = cfg.model.T_len  # Your trained output horizon (64)
    q = cfg.model.q_len  # Your context window size
    
    # 4 chunks of 64 days = 256 days (approx. 1 trading year)
    num_chunks = 4 
    horizon = num_chunks * T 
    
    print(f"Executing stress test over {horizon} steps using {num_chunks} chunks of {T} days...")
    
    with torch.no_grad():
        for i in range(0, len(scaled_contexts), batch_size):
            current_context = scaled_contexts[i : i + batch_size]
            batch_gen_chunks = []
            
            # Roll forward in chunks of T=64
            for chunk in range(num_chunks):
                # Generate the full 64-day block
                next_chunk = gen(current_context, n_steps=T)
                batch_gen_chunks.append(next_chunk)
                
                # Update the context window for the next jump
                # Concatenate the old context with the new chunk, then slice the last 'q' steps
                combined_sequence = torch.cat([current_context, next_chunk], dim=1)
                current_context = combined_sequence[:, -q:, :]
                
            # Stitch the 4 chunks together into a single 256-day path
            batch_generated = torch.cat(batch_gen_chunks, dim=1)
            generated_scaled_list.append(batch_generated)
            
    generated_scaled = torch.cat(generated_scaled_list, dim=0)
    
    # 5. Calculate Expected Cumulative Trajectories
    gen_cum_returns = torch.cumsum(generated_scaled, dim=1).cpu().numpy()
    
    # Average across all paths for Asset 0 to cancel out pure volatility
    expected_gen_path = gen_cum_returns[:, :, 0].mean(axis=0)
    
    # 6. Quantify the True Network Bias strictly against 0.0
    final_gen_drift = expected_gen_path[-1]
    true_bias_at_T = final_gen_drift - 0.0
    
    # Annualize the bias (Assuming 252 trading days per year)
    annualization_factor = 252 / horizon
    annualized_bias_std = true_bias_at_T * annualization_factor
    
    print(f"\nMetric: Expected Cumulative Return at T={horizon} (Standardized Space)")
    print(f"Target (Theoretical Ideal): 0.0000")
    print(f"Model Mean (Actual):        {final_gen_drift:.4f}")
    print(f"\nRaw Compounded Bias over {horizon} steps: {true_bias_at_T:.4f} standard deviations")
    print(f"Annualized Phantom Drift: {annualized_bias_std:.4f} standard deviations / year")
    
    print("\n--- VERDICT ---")
    if abs(annualized_bias_std) > 0.5:
        print("SEVERE BIAS: The unregularized model collapses/drifts heavily over long horizons.")
        print("Conclusion: Global drift regularization is STRICTLY REQUIRED.")
    elif abs(annualized_bias_std) > 0.1:
        print("MODERATE BIAS: The model drifts slightly as autoregressive errors compound.")
        print("Conclusion: Global drift regularization is HIGHLY RECOMMENDED for long-term stability.")
    else:
        print("NEGLIGIBLE BIAS: The GRU architecture is remarkably stable and holds a zero-mean drift.")
        print("Conclusion: Drift regularization is UNNECESSARY. The unregularized model handles it well.")
    print("=" * 55)
    
    # 7. Plotting Model Isolation
    plt.figure(figsize=(9, 5))
    time_steps = np.arange(1, horizon + 1)
    
    plt.axhline(0.0, color='black', linestyle='--', linewidth=2, label='Theoretical Ideal (0.0)')
    plt.plot(time_steps, expected_gen_path, color='red', linewidth=2.5, label='Generated Model Mean')
    
    # Add a confidence-style cone to visualize the acceptable < 0.1 bounds
    acceptable_upper = np.linspace(0, 0.1 / annualization_factor, horizon)
    acceptable_lower = np.linspace(0, -0.1 / annualization_factor, horizon)
    plt.fill_between(time_steps, acceptable_lower, acceptable_upper, color='green', alpha=0.1, label='Stable Threshold (<0.1 std/yr)')
    
    plt.title(f"252-Step Autoregressive Stress Test: {run_name}", fontsize=13)
    plt.xlabel(f"Time Step (up to T={horizon})", fontsize=11)
    plt.ylabel("Expected Cumulative Sum (Standardized)", fontsize=11)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "long_horizon_stress_test.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Saved diagnostic plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Insert your baseline unregularized run name here
    stress_test_bias("20260720_1524_GBM_v1_global_regularized")