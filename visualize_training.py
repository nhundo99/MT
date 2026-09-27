import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from config import Config

def plot_training_losses(window_size=100, start_step=0):
    """
    Reads TensorBoard logs to extract loss histories, computes rolling mean 
    and standard deviation, and plots them side-by-side using a logarithmic scale.
    Allows filtering to only show data from `start_step` onwards.
    """
    cfg = Config()
    
    # Locate the directory where the logs are saved
    run_name = getattr(cfg, 'eval_run_name', cfg.train.experiment_name)
    save_dir = os.path.join(cfg.train.model_base_dir, run_name)
    tb_dir = os.path.join(cfg.train.tb_base_dir, run_name)
    
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load TensorBoard event data
    print(f"Loading TensorBoard logs from: {tb_dir}")
    # size_guidance sets how many records to load per tag (0 means all)
    event_acc = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    event_acc.Reload()
    
    # Find available tags in the logs
    tags = event_acc.Tags().get('scalars', [])
    
    # Map friendly titles to the exact TensorBoard tags you used
    loss_tags = {
        "Total Loss": "Loss/train_total",
        "SOCK Loss": "Loss/train_sock",
        "Drift Penalty": "Loss/train_drift_penalty"
    }
    
    # Filter to only the tags that actually exist in your specific log file
    available_loss_tags = {title: tag for title, tag in loss_tags.items() if tag in tags}
    
    if not available_loss_tags:
        print("No loss tags found in the TensorBoard logs. Make sure the path is correct.")
        return
        
    num_plots = len(available_loss_tags)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    fig.suptitle(f"Training Loss Evolution (Step {start_step}+) with Rolling Variance", fontsize=16, y=1.05)
    
    # Ensure axes is iterable even if there's only 1 plot
    if num_plots == 1:
        axes = [axes]
        
    for ax, (title, tag) in zip(axes, available_loss_tags.items()):
        # Extract steps and values from TensorBoard
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        # Load into a Pandas DataFrame
        df = pd.DataFrame({"step": steps, "value": values}).set_index("step")
        
        # Calculate rolling mean and standard deviation BEFORE truncating
        # so the window is fully populated at the start_step
        df["rolling_mean"] = df["value"].rolling(window=window_size, min_periods=1).mean()
        df["rolling_std"] = df["value"].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        # Filter the dataframe to only include steps >= start_step
        if start_step > 0:
            df = df[df.index >= start_step]
            
        if df.empty:
            print(f"Warning: No data available for '{title}' after step {start_step}.")
            continue
        
        # Clip lower bound for the shaded region to avoid math domain errors on log scale
        lower_bound = (df["rolling_mean"] - df["rolling_std"]).clip(lower=1e-8)
        upper_bound = df["rolling_mean"] + df["rolling_std"]
        
        # 1. Plot raw values faintly in the background
        ax.plot(df.index, df["value"], color="gray", alpha=0.3, label="Raw Loss", linewidth=1)
        
        # 2. Fill the area for the rolling standard deviation
        ax.fill_between(
            df.index, 
            lower_bound, 
            upper_bound, 
            color="blue", alpha=0.2, label=rf"$\pm 1$ Std Dev (w={window_size})"
        )
        
        # 3. Plot the rolling mean on top
        ax.plot(df.index, df["rolling_mean"], color="blue", label="Rolling Mean", linewidth=2)
        
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Enable grid for both major and minor ticks to improve log scale readability
        ax.grid(True, alpha=0.4, which="both")
        
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.legend(loc="upper right", fontsize=10)
        
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"training_losses_evolution_from_{start_step}.pdf")
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Successfully saved training losses plot to: {plot_path}")

# Example usage:
plot_training_losses(window_size=1000, start_step=12000)