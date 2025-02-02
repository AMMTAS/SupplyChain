"""Training module for the Transformer-based Demand Predictor.

This module handles the training loop, loss calculation, and evaluation
metrics for the demand predictor model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime

from config.transformer_config import TransformerConfig
from .network import DemandPredictor


class NegativeLogLikelihood(nn.Module):
    """Negative log-likelihood loss for probabilistic predictions."""
    
    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate NLL loss.
        
        Args:
            mean: Predicted mean [batch_size, seq_len, output_dim]
            std: Predicted standard deviation [batch_size, seq_len, output_dim]
            target: Target values [batch_size, seq_len, output_dim]
            
        Returns:
            NLL loss value
        """
        # Calculate gaussian NLL
        nll = 0.5 * (
            torch.log(2 * np.pi * std.pow(2)) +
            (target - mean).pow(2) / std.pow(2)
        )
        return nll.mean()


class DemandPredictorTrainer:
    """Trainer class for the Demand Predictor model."""
    
    def __init__(
        self,
        model: DemandPredictor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        log_dir: Optional[str] = None
    ):
        """Initialize trainer.
        
        Args:
            model: DemandPredictor model instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate for optimizer
            log_dir: Directory for saving logs (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = NegativeLogLikelihood()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.metrics = {"train": [], "val": []}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("transformer_trainer")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(self.log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(fh)
        return logger
        
    def _log_metrics(self, metrics: Dict[str, float], phase: str, epoch: int):
        """Log metrics to file."""
        metrics["epoch"] = epoch
        self.metrics[phase].append(metrics)
        
        # Save metrics to JSON
        metrics_file = self.log_dir / f"{phase}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics[phase], f, indent=4)
            
        # Log to logger
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != "epoch"])
        self.logger.info(f"{phase.capitalize()} Epoch {epoch}: {metrics_str}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src = src.to(self.model.device)
            tgt = tgt.to(self.model.device)
            
            # Forward pass
            mean, std = self.model(src)
            loss = self.criterion(mean, std, tgt)
            loss_value = loss.item()  # Store loss value before backward pass
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)  # Don't retain graph to free memory
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Add gradient clipping
            self.optimizer.step()
            
            # Free memory
            del mean, std, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Update metrics
            total_loss += loss_value * len(src)
            total_samples += len(src)
            
        return {
            "train_loss": total_loss / total_samples
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model.
        
        Returns:
            Dictionary of metrics
        """
        if not self.val_loader:
            return {}
            
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src = src.to(self.model.device)
                tgt = tgt.to(self.model.device)
                
                # Forward pass
                mean, std = self.model(src)
                loss = self.criterion(mean, std, tgt)
                
                # Update metrics
                total_loss += loss.item() * len(src)
                total_samples += len(src)
                
        return {
            "val_loss": total_loss / total_samples
        }
    
    def train(self) -> Dict[str, float]:
        """Train model.
        
        Returns:
            Dictionary of final metrics
        """
        for epoch in range(self.model.config.max_epochs):
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self._log_metrics(train_metrics, "train", epoch)
            if val_metrics:
                self._log_metrics(val_metrics, "val", epoch)
            
            # Early stopping
            if val_metrics and val_metrics['val_loss'] < getattr(self.model, 'best_val_loss', float('inf')):
                self.model.best_val_loss = val_metrics['val_loss']
                self.model.patience_counter = 0
                self.model.save_checkpoint(str(self.log_dir / 'best.pt'))
            else:
                self.model.patience_counter = getattr(self.model, 'patience_counter', 0) + 1
            
            if getattr(self.model, 'patience_counter', 0) >= self.model.config.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Return final metrics
        return {**train_metrics, **val_metrics}
