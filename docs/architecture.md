# System Architecture

## Design Philosophy and Justification

Our system architecture is based on several key research findings and principles:

1. **Hybrid AI Approach**
   - Combines data-driven learning with expert knowledge
   - Based on findings from Dulman & van Hoof (2019) [1] showing improved robustness in supply chain optimization

2. **Component Isolation**
   - Modular design for independent testing and validation
   - Follows microservices principles adapted for AI systems (Lewis & Fowler, 2014) [2]

3. **Multi-objective Optimization**
   - Balances competing objectives (cost, service level, bullwhip effect)
   - Based on proven approaches in supply chain optimization (Deb et al., 2019) [3]

## References

[1] Dulman, S., & van Hoof, J. (2019). Combining deep learning and optimization for strategic supply chain decisions. European Journal of Operational Research, 281(3), 550-573.

[2] Lewis, J., & Fowler, M. (2014). Microservices: Designing fine-grained systems. O'Reilly Media.

[3] Deb, K., Bandaru, S., Greiner, D., Gaspar-Cunha, A., & Tutum, C. C. (2019). An integrated approach to automated innovization for discovering useful design principles: Case studies from engineering. Applied Soft Computing, 78, 310-328.

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.

[5] Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. Advances in Neural Information Processing Systems, 13.

[6] Zadeh, L. A. (1996). Fuzzy logic = computing with words. IEEE Transactions on Fuzzy Systems, 4(2), 103-111.

[7] Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 712-731.

## Component Overview

### 1. Supply Chain Environment (Completed)
- Multi-echelon simulation environment
- Implements Gymnasium interface
- Handles inventory, backlog, and cost calculations
- Provides state observations and rewards

### 2. Transformer-based Demand Predictor
- Input: Historical demand data
- Output: Multi-horizon demand predictions with uncertainty estimates
- Purpose: Provide accurate demand forecasts to reduce bullwhip effect
- Interface: Feeds predictions to Actor-Critic and Fuzzy Logic components

### 3. Actor-Critic Network
- Input: 
  - Current state from environment
  - Demand predictions from transformer
  - Fuzzy rules evaluation
- Output: 
  - Actor: Order quantity policies
  - Critic: Value function estimation
- Purpose: Learn optimal ordering policies while balancing multiple objectives

### 4. Fuzzy Logic Controller
- Input:
  - Current inventory levels
  - Demand predictions
  - Service levels
  - Cost metrics
- Output: Rule-based recommendations
- Purpose: Incorporate expert knowledge and handle uncertainty
- Rules cover:
  - Safety stock adjustments
  - Order quantity modifications
  - Risk assessment

### 5. Hybrid Information Sharing Network
- Purpose: Enable efficient information flow between echelons
- Features:
  - Distributed state representation
  - Local and global information integration
  - Delay and distortion modeling
- Interface: Provides enhanced state information to Actor-Critic

### 6. Multi-Objective Evolutionary Algorithm (MOEA)
- Objectives:
  1. Minimize total costs
  2. Maximize service level
  3. Minimize bullwhip effect
- Purpose: Optimize:
  - Network architecture parameters
  - Fuzzy rule parameters
  - Information sharing protocols
  - Cost function weights

## Component Interactions

```
[Environment] <─────────────────────────────────────┐
     │                                             │
     │ State                                    Actions
     │                                             │
     ▼                                             │
[Information Sharing Network] ───────┐             │
     │                              │             │
     │                              ▼             │
     │                    [Fuzzy Logic Controller] │
     │                              │             │
     ▼                              │             │
[Transformer]                        │             │
     │                              │             │
     │                              │             │
     └─────────────> [Actor-Critic Network] ──────┘
                            ▲
                            │
                     [MOEA Optimizer]
```

## Development Sequence

1. ✓ Supply Chain Environment
2. Transformer-based Demand Predictor
   - Model architecture
   - Training pipeline
   - Integration with environment
3. Actor-Critic Network
   - Policy network design
   - Value network design
   - Training algorithm
4. Fuzzy Logic Controller
   - Rule base design
   - Inference engine
   - Integration with Actor-Critic
5. Information Sharing Network
   - Network topology
   - Communication protocols
   - State aggregation
6. MOEA Implementation
   - Objective functions
   - Parameter encoding
   - Evolution strategy

## Next Steps

Based on our component-by-component approach, we should proceed with implementing the Transformer-based demand predictor:

1. Design the transformer architecture specifically for supply chain demand prediction
2. Create the training pipeline using historical demand data
3. Implement the interface with our existing environment
4. Evaluate prediction accuracy and uncertainty estimation

Each component will be developed and tested independently, with clear interfaces to the rest of the system. This ensures we can validate each part separately before integration.

## Supply Chain Optimization System Architecture

### System Overview

This document outlines the architecture of our supply chain optimization system, which combines reinforcement learning with advanced information sharing mechanisms to optimize multi-echelon supply chain operations.

### Core Components

#### 1. Information Sharing Network (ISN)

The Information Sharing Network is a specialized neural network architecture that facilitates efficient information flow between supply chain echelons.

##### Key Features
- **Node Architecture**: Each node represents a supply chain echelon with:
  - Local processing network for state handling
  - Order smoothing network for variance reduction
  - Configurable delay buffers
  
- **Information Aggregation**:
  - Attention-based mechanism for dynamic weighting
  - Gated information flow control
  - Support for various topology types

- **State Handling**:
  - Three-dimensional state representation (inventory, backlog, demand)
  - Noise-resistant processing
  - Batch processing support

For detailed implementation and theory, see [Information Sharing Theory](./information_sharing/theory.md).

#### 2. Environment

The supply chain environment implements a realistic simulation of multi-echelon dynamics:

- **State Space**: 
  - Per-echelon state vectors
  - Global state representation
  - Historical information tracking

- **Action Space**:
  - Order quantity decisions
  - Information sharing controls
  - Policy parameters

- **Dynamics**:
  - Material flow simulation
  - Information flow with delays
  - Cost calculations

#### 3. Policy Network

The policy network makes ordering decisions based on local and shared information:

- **Architecture**:
  - Actor-Critic implementation
  - Integration with ISN outputs
  - Multi-head attention for temporal dependencies

- **Training**:
  - PPO algorithm adaptation
  - Custom reward shaping
  - Experience replay with prioritization

### System Integration

#### Data Flow
```
[Environment] → State → [ISN] → Enhanced State → [Policy Network] → Actions → [Environment]
                ↑                                                              ↓
                └──────────────────────── Feedback ───────────────────────────┘
```

#### Training Pipeline
1. Environment initialization
2. ISN state processing
3. Policy network decision
4. Environment step
5. Reward calculation
6. Policy update
7. ISN update (if needed)

### Implementation Details

#### Code Structure
```
src/
├── models/
│   ├── information_sharing/
│   │   ├── network.py
│   │   └── config.py
│   └── policy/
│       ├── network.py
│       └── config.py
├── environment/
│   ├── supply_chain_env.py
│   └── config.py
└── training/
    ├── trainer.py
    └── config.py
```

#### Key Classes

##### InformationSharingNetwork
- Implements the ISN architecture
- Handles information flow and processing
- Manages delay buffers and noise

##### SupplyChainEnv
- Implements the OpenAI Gym interface
- Manages supply chain simulation
- Calculates costs and rewards

##### PolicyNetwork
- Implements the decision-making network
- Integrates with ISN outputs
- Handles action generation

### Testing Strategy

#### Unit Tests
- Component-level testing
- Input/output validation
- Edge case handling

#### Integration Tests
- System-wide behavior
- Performance metrics
- Stability checks

#### Performance Tests
- Scalability verification
- Resource utilization
- Response time measurement

### Configuration

#### Environment Variables
- Training parameters
- Network architectures
- System constants

#### Runtime Configuration
- Supply chain topology
- Cost parameters
- Information sharing settings

### Performance Characteristics

#### Scalability
- Linear with number of echelons
- Batch processing capability
- Distributed training support

#### Efficiency
- Optimized tensor operations
- Memory-efficient implementations
- GPU acceleration support

#### Reliability
- Error handling
- State validation
- Automatic recovery

### Future Enhancements

1. Advanced topology support
2. Real-time adaptation
3. Multi-objective optimization
4. Distributed training
5. Advanced visualization tools

### References

See [Information Sharing Theory](./information_sharing/theory.md) for detailed references.
