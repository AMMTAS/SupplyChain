"""Example usage of the supply chain environment."""

import numpy as np
from supply_chain_env import SupplyChainEnv
import matplotlib.pyplot as plt


def run_simple_policy():
    """Run a simple base-stock policy."""
    # Create environment
    env = SupplyChainEnv(
        num_echelons=3,
        max_steps=100,
        demand_mean=100,
        demand_std=20,
        seed=42
    )
    
    # Initialize metrics tracking
    inventory_history = []
    backlog_history = []
    service_level_history = []
    reward_history = []
    
    # Run one episode
    obs, _ = env.reset()
    
    for step in range(env.max_steps):
        # Simple base-stock policy: order up to target inventory level
        target_levels = np.array([150, 200, 250])  # Higher targets for upstream
        inventory_levels = obs[:env.num_echelons]
        orders = np.maximum(0, target_levels - inventory_levels)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(orders)
        
        # Record metrics
        inventory_history.append(info['inventory_levels'].copy())
        backlog_history.append(info['backlog_levels'].copy())
        service_level_history.append(info['service_level'])
        reward_history.append(reward)
        
        if terminated or truncated:
            break
    
    return {
        'inventory': np.array(inventory_history),
        'backlog': np.array(backlog_history),
        'service_level': np.array(service_level_history),
        'reward': np.array(reward_history)
    }


def plot_results(results):
    """Plot the results of the simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot inventory levels
    axes[0, 0].plot(results['inventory'])
    axes[0, 0].set_title('Inventory Levels')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Inventory')
    axes[0, 0].legend(['Retailer', 'Warehouse', 'Distributor'])
    
    # Plot backlog levels
    axes[0, 1].plot(results['backlog'])
    axes[0, 1].set_title('Backlog Levels')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Backlog')
    axes[0, 1].legend(['Retailer', 'Warehouse', 'Distributor'])
    
    # Plot service level
    axes[1, 0].plot(results['service_level'])
    axes[1, 0].set_title('Service Level')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Service Level')
    axes[1, 0].set_ylim([0, 1])
    
    # Plot rewards
    axes[1, 1].plot(results['reward'])
    axes[1, 1].set_title('Rewards')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Reward')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run simulation
    results = run_simple_policy()
    
    # Plot results
    plot_results(results)
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"Average Service Level: {results['service_level'].mean():.2%}")
    print(f"Average Reward: {results['reward'].mean():.2f}")
    print(f"Final Inventory Levels: {results['inventory'][-1]}")
    print(f"Final Backlog Levels: {results['backlog'][-1]}")
