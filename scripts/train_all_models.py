#!/usr/bin/env python3
"""
Train all trajectory generation models
"""

import sys
import argparse
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.data_generator import TrajectoryDataGenerator
from src.data.dataset import TrajectoryDataset
from src.training.trainer import ModelTrainer

# Import all models - Fixed import paths
from baselines.diffusion_policy_model import DiffusionPolicyModel
from baselines.transformer_model import TransformerModel
from baselines.vae_model import VAEModel
from baselines.mlp_model import MLPModel
from baselines.gflownets_model import GFlowNetModel


def main():
    parser = argparse.ArgumentParser(description="Train all trajectory generation models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['experiment']['output_dir'] = args.output_dir
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train_all_models", output_dir / "logs")
    
    # Generate data (if not exists)
    data_dir = output_dir / "data"
    if not (data_dir / "train.h5").exists():
        logger.info("Generating training data...")
        data_generator = TrajectoryDataGenerator(config['data'])
        train_data, val_data, test_data = data_generator.generate_all_splits()
        
        data_generator.save_data(train_data, data_dir / "train.h5")
        data_generator.save_data(val_data, data_dir / "val.h5")
        data_generator.save_data(test_data, data_dir / "test.h5")
    
    # Model class mapping - Fixed model names
    model_classes = {
        'diffusion_policy': DiffusionPolicyModel,
        'transformer': TransformerModel,
        'vae': VAEModel,
        'mlp': MLPModel,
        'gflownets': GFlowNetModel
    }
    
    # Train all enabled models
    for model_name, model_class in model_classes.items():
        if config['models'][model_name]['enabled']:
            logger.info(f"Starting training for model: {model_name}")
            
            try:
                # Create dataset
                dataset = TrajectoryDataset(
                    train_path=data_dir / "train.h5",
                    val_path=data_dir / "val.h5",
                    test_path=data_dir / "test.h5",
                    config=config['data'],
                    mode='train'
                )
                
                # Create model
                model = model_class(config['models'][model_name])
                
                # Create trainer
                trainer = ModelTrainer(
                    model=model,
                    dataset=dataset,
                    config=config['training'],
                    logger=logger
                )
                
                # Train model
                trainer.train()
                
                # Save model
                checkpoint_dir = output_dir / "checkpoints" / model_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(checkpoint_dir / "best_model.pth")
                
                logger.info(f"Model {model_name} training completed")
                
            except Exception as e:
                logger.error(f"Error training model {model_name}: {str(e)}")
                continue
    
    logger.info("All model training completed!")


if __name__ == "__main__":
    main()