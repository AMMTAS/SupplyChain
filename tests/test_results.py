"""Test results logging module."""

import logging
import json
from datetime import datetime
from pathlib import Path

class TestResults:
    def __init__(self):
        self.results = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('test_results')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Create file handler
        fh = logging.FileHandler(log_dir / 'test_results.log')
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(fh)
        return logger
    
    def log_transformer_results(self, mse, coverage_1sigma, coverage_2sigma, n_samples):
        """Log transformer test results."""
        results = {
            'mse': float(mse),
            'coverage_1sigma': float(coverage_1sigma),
            'coverage_2sigma': float(coverage_2sigma),
            'n_samples': int(n_samples),
            'timestamp': datetime.now().isoformat()
        }
        self.results['transformer'] = results
        self.logger.info(f"Transformer Results: {json.dumps(results)}")
        
    def log_fuzzy_results(self, activation_rate, inference_time, correlation):
        """Log fuzzy controller test results."""
        results = {
            'rule_activation_rate': float(activation_rate),
            'avg_inference_time_ms': float(inference_time),
            'input_output_correlation': float(correlation),
            'timestamp': datetime.now().isoformat()
        }
        self.results['fuzzy'] = results
        self.logger.info(f"Fuzzy Controller Results: {json.dumps(results)}")
        
    def log_moea_results(self, convergence_time, pareto_size, hypervolume):
        """Log MOEA test results."""
        results = {
            'convergence_time_s': float(convergence_time),
            'pareto_front_size': int(pareto_size),
            'hypervolume_indicator': float(hypervolume),
            'timestamp': datetime.now().isoformat()
        }
        self.results['moea'] = results
        self.logger.info(f"MOEA Results: {json.dumps(results)}")
        
    def log_training_results(self, episode_reward, service_level, bullwhip, policy_loss, value_loss):
        """Log end-to-end training results."""
        results = {
            'avg_episode_reward': float(episode_reward),
            'service_level': float(service_level),
            'bullwhip_ratio': float(bullwhip),
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'timestamp': datetime.now().isoformat()
        }
        self.results['training'] = results
        self.logger.info(f"Training Results: {json.dumps(results)}")
        
    def log_objective_performance(self, avg_cost, avg_service, avg_bullwhip, eval_time):
        """Log objective function performance metrics."""
        results = {
            'avg_cost': float(avg_cost),
            'avg_service_level': float(avg_service),
            'avg_bullwhip': float(avg_bullwhip),
            'avg_eval_time_s': float(eval_time),
            'timestamp': datetime.now().isoformat()
        }
        self.results['objectives'] = results
        self.logger.info(f"Objective Function Results: {json.dumps(results)}")
        
    def log_transformer_network(self, avg_loss, loss_std, avg_attention, training_time):
        """Log transformer network performance metrics."""
        results = {
            'avg_loss': float(avg_loss),
            'loss_std': float(loss_std),
            'avg_attention': float(avg_attention),
            'training_time_s': float(training_time),
            'timestamp': datetime.now().isoformat()
        }
        self.results['transformer_network'] = results
        self.logger.info(f"Transformer Network Results: {json.dumps(results)}")
        
    def log_value_network(self, avg_loss, loss_std, value_std, training_time):
        """Log value network performance metrics."""
        results = {
            'avg_loss': float(avg_loss),
            'loss_std': float(loss_std),
            'value_std': float(value_std),
            'training_time_s': float(training_time),
            'timestamp': datetime.now().isoformat()
        }
        self.results['value_network'] = results
        self.logger.info(f"Value Network Results: {json.dumps(results)}")
        
    def log_policy_network(self, avg_loss, loss_std, avg_entropy, avg_kl, training_time):
        """Log policy network performance metrics."""
        results = {
            'avg_loss': float(avg_loss),
            'loss_std': float(loss_std),
            'avg_entropy': float(avg_entropy),
            'avg_kl': float(avg_kl),
            'training_time_s': float(training_time),
            'timestamp': datetime.now().isoformat()
        }
        self.results['policy_network'] = results
        self.logger.info(f"Policy Network Results: {json.dumps(results)}")
        
    def log_isn_integration(self, avg_loss, loss_std, avg_attention, avg_message_norm, training_time):
        """Log ISN integration performance metrics."""
        results = {
            'avg_loss': float(avg_loss),
            'loss_std': float(loss_std),
            'avg_attention': float(avg_attention),
            'avg_message_norm': float(avg_message_norm),
            'training_time_s': float(training_time),
            'timestamp': datetime.now().isoformat()
        }
        self.results['isn_integration'] = results
        self.logger.info(f"ISN Integration Results: {json.dumps(results)}")
        
    def log_isn_network(self, avg_loss, loss_std, avg_attention, avg_hidden_norm, training_time):
        """Log ISN network performance metrics."""
        results = {
            'avg_loss': float(avg_loss),
            'loss_std': float(loss_std),
            'avg_attention': float(avg_attention),
            'avg_hidden_norm': float(avg_hidden_norm),
            'training_time_s': float(training_time),
            'timestamp': datetime.now().isoformat()
        }
        self.results['isn_network'] = results
        self.logger.info(f"ISN Network Results: {json.dumps(results)}")
        
    def log_demand_prediction(self, mse, coverage_1sigma, coverage_2sigma):
        """Log demand prediction performance metrics."""
        results = {
            'mse': float(mse),
            'coverage_1sigma': float(coverage_1sigma),
            'coverage_2sigma': float(coverage_2sigma),
            'timestamp': datetime.now().isoformat()
        }
        self.results['demand_prediction'] = results
        self.logger.info(f"Demand Prediction Results: {json.dumps(results)}")
        
    def save_results(self):
        """Save all results to a JSON file."""
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'test_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
