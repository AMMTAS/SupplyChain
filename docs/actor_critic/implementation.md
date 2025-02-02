# Actor-Critic Network Implementation

## Overview

The Actor-Critic network is a key component of our supply chain optimization system, integrating demand predictions, expert knowledge, and evolutionary optimization to learn optimal ordering policies.

## Architecture

### 1. Input Components

#### Transformer Integration
- Takes 24-dimensional demand predictions
- Captures temporal patterns and uncertainties
- Encoded through dedicated neural network layer

#### Fuzzy Controller Integration
- Takes 2-dimensional recommendations:
  1. Order quantity adjustment
  2. Risk assessment level
- Incorporates expert knowledge into policy

#### MOEA Integration
- Takes 8-dimensional parameter vector:
  1. Network architecture parameters (4)
  2. Cost function weights (3)
  3. Risk sensitivity parameter (1)
- Enables dynamic policy adaptation

### 2. Network Structure

#### Feature Encoders
```python
# Demand encoder
self.demand_encoder = nn.Sequential(
    nn.Linear(demand_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU()
)

# Fuzzy encoder
self.fuzzy_encoder = nn.Sequential(
    nn.Linear(fuzzy_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU()
)

# MOEA encoder
self.moea_encoder = nn.Sequential(
    nn.Linear(moea_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU()
)
```

#### Feature Combination
```python
# Combine features
combined = torch.cat([
    demand_features,
    fuzzy_features,
    moea_features
], dim=-1)
features = self.feature_combiner(combined)
```

#### Policy Network
- Gaussian policy with state-dependent mean and std
- Uses SoftPlus activation for std to ensure positivity
- Includes entropy regularization for exploration

#### Value Network
- Estimates state-value function V(s)
- Uses same feature extractor as policy
- Helps reduce variance in policy updates

## Training

### 1. PPO Algorithm

The network is trained using Proximal Policy Optimization (PPO) with:
- Clipped objective function
- Generalized Advantage Estimation (GAE)
- Early stopping based on KL divergence
- Value function clipping

### 2. Integration Flow

1. **Forward Pass**
   ```python
   # Get inputs
   demand_pred = transformer.predict(history)
   fuzzy_rec = fuzzy_controller.evaluate(state)
   moea_params = moea.get_parameters()
   
   # Get policy distribution and value
   action_dist, value = actor_critic(
       demand_pred,
       fuzzy_rec,
       moea_params
   )
   ```

2. **Action Selection**
   ```python
   # Sample action from distribution
   action = action_dist.sample()
   log_prob = action_dist.log_prob(action)
   ```

3. **Policy Update**
   ```python
   # Update policy using PPO
   metrics = actor_critic.update(
       demand_preds=batch_demand_preds,
       fuzzy_recs=batch_fuzzy_recs,
       moea_params=batch_moea_params,
       actions=batch_actions,
       old_log_probs=batch_log_probs,
       advantages=batch_advantages,
       returns=batch_returns
   )
   ```

## MOEA Parameter Interface

### 1. Parameter Structure

The 8-dimensional MOEA parameter vector controls:

1. **Network Architecture** (4 parameters)
   - Hidden layer sizes
   - Learning rates
   - Entropy coefficient
   - GAE lambda

2. **Cost Function Weights** (3 parameters)
   - Inventory holding cost weight
   - Backlog cost weight
   - Ordering cost weight

3. **Risk Sensitivity** (1 parameter)
   - Controls trade-off between expected return and variance

### 2. Parameter Updates

MOEA optimizes these parameters based on three objectives:
1. Minimize total costs
2. Maximize service level
3. Minimize bullwhip effect

The optimization process:
1. Evaluates current parameters
2. Generates new parameter sets
3. Updates based on Pareto dominance
4. Provides new parameters to Actor-Critic

## Integration Tests

The implementation includes comprehensive integration tests:

1. **Component Integration**
   - Tests interaction between all components
   - Verifies input/output shapes
   - Checks data flow correctness

2. **Training Loop**
   - Tests full training iteration
   - Verifies gradient flow
   - Checks metric calculations

3. **MOEA Interface**
   - Tests parameter encoding
   - Verifies parameter effects
   - Checks update mechanism

## Performance Metrics

The system tracks several key metrics:

1. **Training Metrics**
   - Policy loss
   - Value loss
   - Policy entropy
   - KL divergence

2. **Supply Chain Metrics**
   - Order fill rate
   - Inventory costs
   - Bullwhip effect ratio

3. **Integration Metrics**
   - Component update frequency
   - Parameter adaptation rate
   - Policy convergence speed

## References

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

[2] Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 712-731.

[3] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

[4] Zadeh, L. A. (1996). Fuzzy logic = computing with words. IEEE Transactions on Fuzzy Systems, 4(2), 103-111.

[5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
