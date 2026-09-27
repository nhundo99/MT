import torch
import matplotlib.pyplot as plt
from data_loader import HestonSimulator

# --- Assume HestonSimulator class is defined here ---

def run_variance_study():
    # 1. Study Parameters
    num_paths = 2000
    years = 5.0          # Simulate over 5 years to allow mean reversion
    dt = 1/252
    H = int(years * 252)
    
    # Model parameters
    kappa = 2.0          # Speed of mean reversion
    theta_var = 0.04     # Long-term mean (Target)
    v0 = 0.01            # Start deliberately far from theta to see the curve
    
    print(f"Starting simulation of {num_paths} paths...")
    
    # We use d=1 to simulate one asset per path to avoid massive matrices
    simulator = HestonSimulator(
        d=1, 
        mu=0.05, 
        kappa=kappa, 
        theta_var=theta_var, 
        xi=0.2, 
        rho=-0.7, 
        v0=v0
    )

    # 2. Run Monte Carlo Simulation
    # Store the variance paths: shape (num_paths, H)
    all_variances = torch.zeros((num_paths, H))
    
    for i in range(num_paths):
        _, variances = simulator.simulate(H=H, dt=dt)
        all_variances[i, :] = variances[:, 0]

    # 3. Calculate Means
    # Cross-sectional mean across all paths at each time step
    simulated_mean_path = all_variances.mean(dim=0)
    
    # Time array for the theoretical calculation
    time = torch.linspace(dt, years, H)
    
    # Exact theoretical expectation: E[V_t] = V_0 * exp(-k*t) + theta * (1 - exp(-k*t))
    theoretical_mean_path = v0 * torch.exp(-kappa * time) + theta_var * (1.0 - torch.exp(-kappa * time))

    # Terminal statistics (at t = T)
    simulated_terminal_mean = simulated_mean_path[-1].item()
    theoretical_terminal_mean = theoretical_mean_path[-1].item()
    
    # 4. Print Results
    print("\n--- Variance Simulation Study Results ---")
    print(f"Target Long-term Mean (theta):   {theta_var:.6f}")
    print(f"Theoretical Mean at T={years}y:     {theoretical_terminal_mean:.6f}")
    print(f"Simulated Mean at T={years}y:       {simulated_terminal_mean:.6f}")
    print(f"Absolute Error:                  {abs(simulated_terminal_mean - theoretical_terminal_mean):.6f}")

    # 5. Visualize Convergence
    plt.figure(figsize=(10, 6))
    
    # Plot a few individual paths lightly in the background to show the noise
    for i in range(min(50, num_paths)):
        plt.plot(time.numpy(), all_variances[i, :].numpy(), color='gray', alpha=0.1, linewidth=0.5)
        
    # Plot the aggregated simulation mean
    plt.plot(time.numpy(), simulated_mean_path.numpy(), color='blue', linewidth=2, 
             label=f'Simulated Mean ({num_paths} paths)')
    
    # Plot the exact mathematical expectation
    plt.plot(time.numpy(), theoretical_mean_path.numpy(), color='red', linestyle='--', linewidth=2, 
             label='Theoretical Expected Value')
    
    # Plot the long-term asymptote
    plt.axhline(theta_var, color='black', linestyle=':', linewidth=2, 
                label='Long-Term Mean ($\\theta$)')

    plt.title(f'Heston Variance Mean Reversion ($V_0={v0}$, $\\theta={theta_var}$, $\\kappa={kappa}$)')
    plt.xlabel('Time (Years)')
    plt.ylabel('Variance ($V_t$)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_heston_paths():
    # 1. Define Simulation Parameters
    d = 3                # Number of assets
    years = 2            # Simulation horizon in years
    dt = 1/252           # Daily time steps
    H = int(years * 252) # Total number of steps
    
    # Target correlation matrix for the assets
    target_corr = torch.tensor([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])

    # 2. Instantiate Simulator
    # Using strong mean reversion (kappa=3.0) and high vol-of-vol (xi=0.3) 
    # to make the variance paths visually interesting.
    simulator = HestonSimulator(
        d=d, 
        mu=0.08,           # 8% expected return
        kappa=3.0,         # Mean reversion speed
        theta_var=0.04,    # Long-term variance (approx 20% vol)
        xi=0.3,            # Volatility of volatility
        rho=-0.7,          # Strong leverage effect (price drops when vol spikes)
        v0=0.04,           # Initial variance
        corr_matrix=target_corr
    )

    # 3. Run Simulation
    # Ensure you are on CPU for matplotlib, or move tensors to CPU after
    returns, variances = simulator.simulate(H=H, dt=dt)

    # 4. Process Data for Plotting
    # Convert log-returns to cumulative prices starting at 100
    prices = 100.0 * torch.exp(torch.cumsum(returns, dim=0))
    
    # Prepend the initial state (t=0) to the paths
    S0 = torch.full((1, d), 100.0)
    V0 = torch.full((1, d), simulator.v0)
    
    prices = torch.cat([S0, prices], dim=0).numpy()
    variances = torch.cat([V0, variances], dim=0).numpy()
    
    # Create time axis in years
    time = torch.linspace(0, years, H + 1).numpy()

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Prices
    for i in range(d):
        ax1.plot(time, prices[:, i], label=f'Asset {i+1}')
        
    ax1.set_title('Simulated Asset Prices ($S_t$)')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Price')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot Variances
    for i in range(d):
        ax2.plot(time, variances[:, i], alpha=0.8, label=f'Variance {i+1}')
        
    # Add a line for the long-term mean variance (theta)
    ax2.axhline(simulator.theta_var, color='black', linestyle='--', 
                label='Long-term Mean ($\\theta$)')
    
    ax2.set_title('Simulated Variance Paths ($V_t$)')
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Variance')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Run the function
if __name__ == "__main__":
    run_variance_study()
    plot_heston_paths()