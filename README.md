# Intelligent Supply Chain Management System

A comprehensive supply chain management system that combines modern AI techniques to optimize inventory management, demand forecasting, and decision-making across multiple echelons.

## Components

### 1. Environment (Supply Chain Simulator)
- Multi-echelon supply chain environment
- Realistic demand patterns and inventory dynamics
- Service level and cost metrics
- Built on OpenAI Gym interface

### 2. Information Sharing Network (ISN)
- Neural network-based information sharing across echelons
- Reduces information distortion and bullwhip effect
- Enhances state representation for better decision-making
- References:
  - Lee, H. L., Padmanabhan, V., & Whang, S. (1997). "Information distortion in a supply chain: The bullwhip effect." Management Science, 43(4), 546-558.
  - Chen, F., Drezner, Z., Ryan, J. K., & Simchi-Levi, D. (2000). "Quantifying the bullwhip effect in a simple supply chain: The impact of forecasting, lead times, and information." Management Science, 46(3), 436-443.

### 3. Transformer-based Demand Predictor
- State-of-the-art sequence modeling for demand forecasting
- Attention mechanism captures long-term dependencies
- Probabilistic output for uncertainty estimation
- References:
  - Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS.
  - Salinas, D., et al. (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." International Journal of Forecasting, 36(3), 1181-1191.

### 4. Fuzzy Logic Controller
- Rule-based decision support system
- Expert knowledge integration
- Handles uncertainty and imprecision
- References:
  - Petrovic, D., Roy, R., & Petrovic, R. (1999). "Supply chain modelling using fuzzy sets." International Journal of Production Economics, 59(1-3), 443-453.
  - Zadeh, L. A. (1996). "Fuzzy logic = computing with words." IEEE Transactions on Fuzzy Systems, 4(2), 103-111.

### 5. Multi-Objective Evolutionary Algorithm (MOEA)
- Optimizes multiple competing objectives
- Pareto-optimal solutions for trade-off analysis
- Dynamic parameter adaptation
- References:
  - Zhang, Q., & Li, H. (2007). "MOEA/D: A multiobjective evolutionary algorithm based on decomposition." IEEE Transactions on Evolutionary Computation, 11(6), 712-731.
  - Deb, K., & Jain, H. (2014). "An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach." IEEE Transactions on Evolutionary Computation, 18(4), 577-601.

### 6. Actor-Critic Network
- Deep reinforcement learning for final decision-making
- Combines value and policy learning
- Integrates information from all components
- References:
  - Konda, V. R., & Tsitsiklis, J. N. (2000). "Actor-critic algorithms." NeurIPS.
  - Lillicrap, T. P., et al. (2016). "Continuous control with deep reinforcement learning." ICLR.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/new_supply_chain.git
cd new_supply_chain
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage:
```python
from main import SupplyChainSystem, SystemConfig

# Initialize system with default configuration
config = SystemConfig()
system = SupplyChainSystem(config)

# Train the system
system.train(n_episodes=1000)
```

2. Custom configuration:
```python
config = SystemConfig(
    n_echelons=4,
    demand_history_len=24,
    episode_length=100,
    hidden_size=64,
    n_heads=4,
    n_layers=3,
    dropout=0.1,
    learning_rate=1e-4
)
system = SupplyChainSystem(config)
```

3. Running tests:
```bash
pytest -v
```

## Project Structure

```
new_supply_chain/
├── docs/                    # Documentation
│   ├── architecture.md      # System architecture
│   ├── actor_critic/        # Actor-critic documentation
│   ├── demand_prediction/   # Demand prediction documentation
│   ├── fuzzy_controller/    # Fuzzy controller documentation
│   ├── information_sharing/ # ISN documentation
│   ├── moea/               # MOEA documentation
│   └── research_documentation/ # Research papers and references
├── src/                     # Source code
│   ├── environment/         # Supply chain environment
│   └── models/             # AI/ML models
├── tests/                   # Test cases
├── main.py                 # Main script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{supply_chain_ai_2025,
  title={Intelligent Supply Chain Management System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/new_supply_chain}
}
```

## References

See individual component documentation in the `docs/` directory for detailed references and theoretical foundations.
