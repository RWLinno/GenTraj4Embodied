"""
Data generation utilities for trajectory generation models
"""

import numpy as np
import h5py
from typing import Dict, Any, Tuple, List
from pathlib import Path
import torch


class TrajectoryDataGenerator:
    """
    Generate synthetic trajectory data for training and evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_trajectories = config.get('num_trajectories', 10000)
        self.trajectory_length = config.get('trajectory_length', 50)
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.workspace_bounds = config.get('workspace_bounds', [[-1, 1], [-1, 1], [0, 2]])
        self.add_noise = config.get('add_noise', True)
        self.noise_std = config.get('noise_std', 0.01)
        
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray) -> np.ndarray:
        """
        Generate a single trajectory from start to end pose
        
        Args:
            start_pose: Starting pose [input_dim]
            end_pose: Ending pose [input_dim]
            
        Returns:
            Generated trajectory [trajectory_length, output_dim]
        """
        # Linear interpolation as base trajectory
        t = np.linspace(0, 1, self.trajectory_length)
        trajectory = np.outer(1 - t, start_pose) + np.outer(t, end_pose)
        
        # Add smooth variations to make trajectory more realistic
        if self.trajectory_length > 2:
            # Add sinusoidal variations
            for i in range(self.output_dim):
                frequency = np.random.uniform(0.5, 2.0)
                amplitude = np.random.uniform(0.05, 0.15)
                phase = np.random.uniform(0, 2 * np.pi)
                
                variation = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                trajectory[:, i] += variation
        
        # Add noise if enabled
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std, trajectory.shape)
            trajectory += noise
        
        # Ensure start and end poses are preserved
        trajectory[0] = start_pose
        trajectory[-1] = end_pose
        
        return trajectory
    
    def generate_random_pose(self) -> np.ndarray:
        """
        Generate a random pose within workspace bounds
        
        Returns:
            Random pose [output_dim]
        """
        pose = np.zeros(self.output_dim)
        
        # Position (first 3 dimensions)
        for i in range(min(3, self.output_dim)):
            if i < len(self.workspace_bounds):
                bounds = self.workspace_bounds[i]
                pose[i] = np.random.uniform(bounds[0], bounds[1])
            else:
                pose[i] = np.random.uniform(-1, 1)
        
        # Orientation (quaternion - last 4 dimensions if available)
        if self.output_dim >= 7:
            # Generate random quaternion
            u1, u2, u3 = np.random.uniform(0, 1, 3)
            q = np.array([
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
                np.sqrt(u1) * np.cos(2 * np.pi * u3)
            ])
            pose[3:7] = q
        elif self.output_dim > 3:
            # Fill remaining dimensions with small random values
            pose[3:] = np.random.uniform(-0.1, 0.1, self.output_dim - 3)
        
        return pose
    
    def generate_dataset(self, num_trajectories: int) -> Dict[str, np.ndarray]:
        """
        Generate a dataset of trajectories
        
        Args:
            num_trajectories: Number of trajectories to generate
            
        Returns:
            Dataset dictionary containing trajectories, start poses, and end poses
        """
        trajectories = []
        start_poses = []
        end_poses = []
        
        for i in range(num_trajectories):
            # Generate random start and end poses
            start_pose = self.generate_random_pose()
            end_pose = self.generate_random_pose()
            
            # Generate trajectory
            trajectory = self.generate_trajectory(start_pose, end_pose)
            
            trajectories.append(trajectory)
            start_poses.append(start_pose)
            end_poses.append(end_pose)
        
        return {
            'trajectories': np.array(trajectories),
            'start_poses': np.array(start_poses),
            'end_poses': np.array(end_poses)
        }
    
    def generate_all_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[Dict, Dict, Dict]:
        """
        Generate train, validation, and test splits
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Generate full dataset
        full_data = self.generate_dataset(self.num_trajectories)
        
        # Calculate split sizes
        num_train = int(self.num_trajectories * train_ratio)
        num_val = int(self.num_trajectories * val_ratio)
        num_test = self.num_trajectories - num_train - num_val
        
        # Create splits
        train_data = {
            'trajectories': full_data['trajectories'][:num_train],
            'start_poses': full_data['start_poses'][:num_train],
            'end_poses': full_data['end_poses'][:num_train]
        }
        
        val_data = {
            'trajectories': full_data['trajectories'][num_train:num_train + num_val],
            'start_poses': full_data['start_poses'][num_train:num_train + num_val],
            'end_poses': full_data['end_poses'][num_train:num_train + num_val]
        }
        
        test_data = {
            'trajectories': full_data['trajectories'][num_train + num_val:],
            'start_poses': full_data['start_poses'][num_train + num_val:],
            'end_poses': full_data['end_poses'][num_train + num_val:]
        }
        
        return train_data, val_data, test_data
    
    def save_data(self, data: Dict[str, np.ndarray], filepath: Path) -> None:
        """
        Save data to HDF5 file
        
        Args:
            data: Data dictionary
            filepath: Path to save file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value, compression='gzip')
    
    def load_data(self, filepath: Path) -> Dict[str, np.ndarray]:
        """
        Load data from HDF5 file
        
        Args:
            filepath: Path to load file
            
        Returns:
            Data dictionary
        """
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        
        return data