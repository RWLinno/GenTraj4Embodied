"""
Dataset utilities for trajectory generation models
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory generation training and evaluation
    """
    
    def __init__(self, train_path: Path, val_path: Path, test_path: Path,
                 config: Dict[str, Any], mode: str = 'train'):
        """
        Initialize trajectory dataset
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            config: Dataset configuration
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.config = config
        self.mode = mode
        
        # Load appropriate data based on mode
        if mode == 'train':
            self.data = self._load_data(train_path)
        elif mode == 'val':
            self.data = self._load_data(val_path)
        elif mode == 'test':
            self.data = self._load_data(test_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Data preprocessing
        self.normalize = config.get('normalize', False)
        if self.normalize:
            self._compute_normalization_stats()
    
    def _load_data(self, filepath: Path) -> Dict[str, np.ndarray]:
        """
        Load data from HDF5 file
        
        Args:
            filepath: Path to data file
            
        Returns:
            Data dictionary
        """
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        
        return data
    
    def _compute_normalization_stats(self):
        """
        Compute normalization statistics for trajectories
        """
        trajectories = self.data['trajectories']
        
        # Compute mean and std for each dimension
        self.trajectory_mean = np.mean(trajectories, axis=(0, 1))
        self.trajectory_std = np.std(trajectories, axis=(0, 1))
        
        # Avoid division by zero
        self.trajectory_std = np.maximum(self.trajectory_std, 1e-8)
    
    def _normalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Normalize trajectory using computed statistics
        """
        if self.normalize:
            return (trajectory - self.trajectory_mean) / self.trajectory_std
        return trajectory
    
    def _denormalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Denormalize trajectory using computed statistics
        """
        if self.normalize:
            return trajectory * self.trajectory_std + self.trajectory_mean
        return trajectory
    
    def __len__(self) -> int:
        """
        Get dataset length
        """
        return len(self.data['trajectories'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing trajectory data
        """
        trajectory = self.data['trajectories'][idx]
        start_pose = self.data['start_poses'][idx]
        end_pose = self.data['end_poses'][idx]
        
        # Normalize if enabled
        trajectory = self._normalize_trajectory(trajectory)
        start_pose = self._normalize_trajectory(start_pose.reshape(1, -1)).flatten()
        end_pose = self._normalize_trajectory(end_pose.reshape(1, -1)).flatten()
        
        # Convert to tensors
        return {
            'trajectory': torch.from_numpy(trajectory).float(),
            'start_pose': torch.from_numpy(start_pose).float(),
            'end_pose': torch.from_numpy(end_pose).float(),
            'idx': torch.tensor(idx, dtype=torch.long)
        }


def create_data_loaders(train_dataset: TrajectoryDataset,
                       val_dataset: TrajectoryDataset,
                       test_dataset: TrajectoryDataset,
                       config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader