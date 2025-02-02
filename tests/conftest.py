"""Test fixtures for supply chain tests."""

import pytest
import logging
import json
from datetime import datetime
from pathlib import Path


class ResultsLogger:
    """Helper class for logging test results."""
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
    
    def log_result(self, component, **kwargs):
        """Generic method to log results for any component."""
        results = {**kwargs, 'timestamp': datetime.now().isoformat()}
        self.results[component] = results
        self.logger.info(f"{component.title()} Results: {json.dumps(results)}")
        
    def save_results(self):
        """Save all results to file."""
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)


@pytest.fixture
def results_logger():
    """Fixture providing a results logger instance."""
    return ResultsLogger()
