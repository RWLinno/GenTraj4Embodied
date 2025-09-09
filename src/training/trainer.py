"""
Training utilities for trajectory generation models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import time


class ModelTrainer:
    """
    Generic trainer for trajectory generation models
    """
    
    def __init__(self, model: nn.Module, dataset, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            dataset: Training dataset
            config: Training configuration
            logger: Logger instance
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger
        
        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.num_epochs = config.get('num_epochs', 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        optimizer_type = config.get('optimizer', 'adam')
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Setup scheduler
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Early stopping
        self.early_stopping = config.get('early_stopping', {})
        self.patience = self.early_stopping.get('patience', 10)
        self.min_delta = self.early_stopping.get('min_delta', 1e-6)
        self.patience_counter = 0
        
        # Gradient clipping
        self.gradient_clip = config.get('gradient_clip', 1.0)
        
        # Validation frequency
        self.validation_freq = config.get('validation_freq', 5)
        self.checkpoint_freq = config.get('checkpoint_freq', 10)
    
    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            data_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                start_pose = batch['start_pose']
                end_pose = batch['end_pose']
                target_trajectory = batch['trajectory']
                
                # Generate trajectory
                predicted_trajectory = self.model.forward(start_pose, end_pose)
                
                # Compute loss
                loss = self.model.compute_loss(predicted_trajectory, target_trajectory)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                # Update parameters
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if batch_idx % 100 == 0:
                    self.logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.6f}')
                    
            except Exception as e:
                self.logger.error(f'Error in training batch {batch_idx}: {str(e)}')
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self, data_loader: DataLoader) -> float:
        """
        Validate for one epoch
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                try:
                    start_pose = batch['start_pose']
                    end_pose = batch['end_pose']
                    target_trajectory = batch['trajectory']
                    
                    # Generate trajectory
                    predicted_trajectory = self.model.forward(start_pose, end_pose)
                    
                    # Compute loss
                    loss = self.model.compute_loss(predicted_trajectory, target_trajectory)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f'Error in validation batch {batch_idx}: {str(e)}')
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def train(self):
        """
        Main training loop
        """
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # Create data loader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Logging
            self.logger.info(f'Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss:.6f}')
            
            # Validation (if validation data is available)
            if hasattr(self.dataset, 'val_data') and epoch % self.validation_freq == 0:
                val_loss = self.validate_epoch(train_loader)  # Using train_loader for now
                self.val_losses.append(val_loss)
                self.logger.info(f'Epoch {epoch}/{self.num_epochs}, Val Loss: {val_loss:.6f}')
                
                # Early stopping check
                if val_loss < self.best_loss - self.min_delta:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    self.logger.info(f'New best validation loss: {val_loss:.6f}')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.logger.info(f'Early stopping triggered after {epoch} epochs')
                        break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Checkpoint saving
            if epoch % self.checkpoint_freq == 0:
                self.logger.info(f'Saving checkpoint at epoch {epoch}')
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time:.2f} seconds')
    
    def save_checkpoint(self, filepath: Path):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f'Checkpoint saved to {filepath}')
    
    def load_checkpoint(self, filepath: Path):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to load checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f'Checkpoint loaded from {filepath}')