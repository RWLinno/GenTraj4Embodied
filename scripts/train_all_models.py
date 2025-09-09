#!/usr/bin/env python3
"""
Train all trajectory generation models
训练所有轨迹生成模型
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.data_generator import TrajectoryDataGenerator
from src.data.dataset import TrajectoryDataset
from src.training.trainer import ModelTrainer

# 导入所有模型
from baselines.diffusion_policy.model import DiffusionPolicyModel
from baselines.transformer.model import TransformerModel
from baselines.vae.model import VAEModel
from baselines.mlp.model import MLPModel
from baselines.gflownets.model import GFlowNetModel


def main():
    parser = argparse.ArgumentParser(description="训练所有轨迹生成模型")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="experiments", help="输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config['experiment']['output_dir'] = args.output_dir
    
    # 设置日志
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train_all_models", output_dir / "logs")
    
    # 生成数据（如果不存在）
    data_dir = output_dir / "data"
    if not (data_dir / "train.h5").exists():
        logger.info("生成训练数据...")
        data_generator = TrajectoryDataGenerator(config['data'])
        train_data, val_data, test_data = data_generator.generate_all_splits()
        
        data_generator.save_data(train_data, data_dir / "train.h5")
        data_generator.save_data(val_data, data_dir / "val.h5")
        data_generator.save_data(test_data, data_dir / "test.h5")
    
    # 模型类映射
    model_classes = {
        'diffusion_policy': DiffusionPolicyModel,
        'transformer': TransformerModel,
        'vae': VAEModel,
        'mlp': MLPModel,
        'gflownets': GFlowNetModel
    }
    
    # 训练所有启用的模型
    for model_name, model_class in model_classes.items():
        if config['models'][model_name]['enabled']:
            logger.info(f"开始训练模型: {model_name}")
            
            try:
                # 创建数据集
                dataset = TrajectoryDataset(
                    train_path=data_dir / "train.h5",
                    val_path=data_dir / "val.h5",
                    test_path=data_dir / "test.h5",
                    config=config['data'],
                    mode='train'
                )
                
                # 创建模型
                model = model_class(config['models'][model_name])
                
                # 创建训练器
                trainer = ModelTrainer(
                    model=model,
                    dataset=dataset,
                    config=config['training'],
                    logger=logger
                )
                
                # 训练模型
                trainer.train()
                
                # 保存模型
                checkpoint_dir = output_dir / "checkpoints" / model_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(checkpoint_dir / "best_model.pth")
                
                logger.info(f"模型 {model_name} 训练完成")
                
            except Exception as e:
                logger.error(f"训练模型 {model_name} 时出错: {str(e)}")
                continue
    
    logger.info("所有模型训练完成!")


if __name__ == "__main__":
    main()