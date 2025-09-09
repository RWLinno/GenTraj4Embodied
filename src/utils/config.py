"""
Configuration utilities for trajectory generation models
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Create default config if not exists
        default_config = get_default_config()
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'experiment': {
            'name': 'trajectory_generation',
            'output_dir': 'experiments',
            'seed': 42
        },
        'data': {
            'num_trajectories': 10000,
            'trajectory_length': 50,
            'input_dim': 7,
            'output_dim': 7,
            'workspace_bounds': [[-1, 1], [-1, 1], [0, 2]],
            'add_noise': True,
            'noise_std': 0.01
        },
        'models': {
            'diffusion_policy': {
                'enabled': True,
                'architecture': {
                    'horizon': 16,
                    'num_steps': 100,
                    'unet_dim': 256,
                    'num_layers': 4,
                    'time_embed_dim': 128,
                    'beta_schedule': 'cosine',
                    'prediction_type': 'epsilon',
                    'dropout': 0.1
                }
            },
            'transformer': {
                'enabled': True,
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'max_seq_length': 50
            },
            'vae': {
                'enabled': True,
                'latent_dim': 64,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout': 0.1,
                'beta': 1.0
            },
            'mlp': {
                'enabled': True,
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.1,
                'activation': 'relu',
                'use_batch_norm': False,
                'use_residual': False
            },
            'gflownets': {
                'enabled': True,
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.1,
                'max_trajectory_length': 50,
                'temperature': 1.0
            }
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'gradient_clip': 1.0,
            'early_stopping': {
                'patience': 10,
                'min_delta': 1e-6
            },
            'validation_freq': 5,
            'checkpoint_freq': 10
        },
        'evaluation': {
            'metrics': ['mse', 'smoothness', 'end_error'],
            'num_samples': 100,
            'visualization': True
        }
    }


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values
    
    Args:
        config: Original configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    updated_config = config.copy()
    deep_update(updated_config, updates)
    return updated_config