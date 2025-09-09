#!/usr/bin/env python3
"""
3D End-Effector Trajectory Generation - Main Runner
主运行脚本，支持训练、评估和生成轨迹
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import random
from typing import Dict, Any

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_generator import TrajectoryDataGenerator
from src.data.dataset import TrajectoryDataset
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from src.utils.config import load_config, validate_config
from src.utils.visualization import TrajectoryVisualizer

# 导入所有模型 - 新的6类分类框架

# 1. Classical & Optimization-based Methods
from baselines.classical_optimization.interpolation.linear_interpolation import LinearInterpolationModel
from baselines.classical_optimization.interpolation.spline_interpolation import SplineInterpolationModel
from baselines.classical_optimization.movement_primitives.dmp_model import DMPTrajectoryModel
from baselines.classical_optimization.movement_primitives.promp_model import ProMPTrajectoryModel
from baselines.classical_optimization.optimal_control.mpc_model import MPCTrajectoryModel
from baselines.classical_optimization.sampling_planning.rrt_model import RRTTrajectoryModel
from baselines.classical_optimization.sampling_planning.prm_model import PRMTrajectoryModel
from baselines.classical_optimization.statistical_modeling.gmm_model import GMMTrajectoryModel

# 2. Generative Probabilistic Models
from baselines.generative_probabilistic.variational.vae_model import VAETrajectoryModel
from baselines.generative_probabilistic.diffusion.ddpm_model import DDPMTrajectoryModel
from baselines.generative_probabilistic.diffusion.ddim_model import DDIMTrajectoryModel
from baselines.generative_probabilistic.diffusion.difftraj_model import DiffTrajModel
from baselines.generative_probabilistic.diffusion.score_based_model import ScoreBasedTrajectoryModel
from baselines.generative_probabilistic.diffusion.conditional_diffusion_model import ConditionalDiffusionTrajectoryModel
from baselines.generative_probabilistic.flows.normalizing_flows_model import NormalizingFlowsModel
from baselines.generative_probabilistic.adversarial.gflownets_model import GFlowNetTrajectoryModel

# 3. Sequential Modeling Methods
from baselines.sequential_modeling.recurrent.lstm_model import LSTMTrajectoryModel
from baselines.sequential_modeling.recurrent.gru_model import GRUTrajectoryModel
from baselines.sequential_modeling.recurrent.rnn_model import RNNTrajectoryModel
from baselines.sequential_modeling.transformers.bert_model import BERTTrajectoryModel
from baselines.sequential_modeling.transformers.gpt_model import GPTTrajectoryModel
from baselines.sequential_modeling.decision_transformers.decision_transformer_model import DecisionTransformerModel
from baselines.sequential_modeling.mixture_density.mdn_model import MDNTrajectoryModel

# 4. Reinforcement Learning Methods
from baselines.reinforcement_learning.model_free.ppo_model import PPOTrajectoryModel
from baselines.reinforcement_learning.model_free.actor_critic_model import ActorCriticTrajectoryModel
from baselines.reinforcement_learning.model_free.policy_gradient_model import PolicyGradientTrajectoryModel
from baselines.reinforcement_learning.imitation.imitation_learning_model import ImitationLearningTrajectoryModel
from baselines.reinforcement_learning.offline.inverse_rl_model import InverseRLTrajectoryModel

# 5. Hybrid / Hierarchical Policies
from baselines.hybrid_hierarchical.il_rl.il_rl_model import ILRLTrajectoryModel
from baselines.hybrid_hierarchical.mpc_learning.mpc_learning_model import MPCLearningModel
from baselines.hybrid_hierarchical.hierarchical_control.hierarchical_control_model import HierarchicalControlModel

# 6. Fundamental Architectures
from baselines.fundamental_architectures.neural_networks.mlp_model import MLPTrajectoryModel
from baselines.fundamental_architectures.convolutional.cnn_model import CNNTrajectoryModel
from baselines.fundamental_architectures.graph_networks.gnn_model import GNNTrajectoryModel

# Legacy models for backward compatibility
from baselines.diffusion_policy.model import DiffusionPolicyModel
from baselines.transformer.model import TransformerTrajectoryModel


def set_seed(seed: int):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_class(model_name: str):
    """根据模型名称获取对应的模型类 - 新的6类分类框架"""
    model_classes = {
        # 1. Classical & Optimization-based Methods
        'linear_interpolation': LinearInterpolationModel,
        'spline_interpolation': SplineInterpolationModel,
        'dmp': DMPTrajectoryModel,
        'promp': ProMPTrajectoryModel,
        'mpc': MPCTrajectoryModel,
        'rrt': RRTTrajectoryModel,
        'prm': PRMTrajectoryModel,
        'gmm': GMMTrajectoryModel,
        
        # 2. Generative Probabilistic Models
        'vae': VAETrajectoryModel,
        'ddpm': DDPMTrajectoryModel,
        'ddim': DDIMTrajectoryModel,
        'difftraj': DiffTrajModel,
        'score_based': ScoreBasedTrajectoryModel,
        'conditional_diffusion': ConditionalDiffusionTrajectoryModel,
        'normalizing_flows': NormalizingFlowsModel,
        'gflownets': GFlowNetTrajectoryModel,
        
        # 3. Sequential Modeling Methods
        'lstm': LSTMTrajectoryModel,
        'gru': GRUTrajectoryModel,
        'rnn': RNNTrajectoryModel,
        'bert': BERTTrajectoryModel,
        'gpt': GPTTrajectoryModel,
        'decision_transformer': DecisionTransformerModel,
        'mdn': MDNTrajectoryModel,
        
        # 4. Reinforcement Learning Methods
        'ppo': PPOTrajectoryModel,
        'actor_critic': ActorCriticTrajectoryModel,
        'policy_gradient': PolicyGradientTrajectoryModel,
        'imitation_learning': ImitationLearningTrajectoryModel,
        'inverse_rl': InverseRLTrajectoryModel,
        
        # 5. Hybrid / Hierarchical Policies
        'il_rl': ILRLTrajectoryModel,
        'mpc_learning': MPCLearningModel,
        'hierarchical_control': HierarchicalControlModel,
        
        # 6. Fundamental Architectures
        'mlp': MLPTrajectoryModel,
        'cnn': CNNTrajectoryModel,
        'gnn': GNNTrajectoryModel,
        
        # Legacy models for backward compatibility
        'diffusion_policy': DiffusionPolicyModel,
        'transformer': TransformerTrajectoryModel
    }
    
    if model_name not in model_classes:
        available_models = list(model_classes.keys())
        print(f"\n可用的模型类别:")
        print(f"1. Classical & Optimization-based: {[m for m in available_models if m in ['linear_interpolation', 'spline_interpolation', 'dmp', 'promp', 'mpc', 'rrt', 'prm', 'gmm']]}")
        print(f"2. Generative Probabilistic: {[m for m in available_models if m in ['vae', 'ddpm', 'ddim', 'difftraj', 'score_based', 'conditional_diffusion', 'normalizing_flows', 'gflownets']]}")
        print(f"3. Sequential Modeling: {[m for m in available_models if m in ['lstm', 'gru', 'rnn', 'bert', 'gpt', 'decision_transformer', 'mdn']]}")
        print(f"4. Reinforcement Learning: {[m for m in available_models if m in ['ppo', 'actor_critic', 'policy_gradient', 'imitation_learning', 'inverse_rl']]}")
        print(f"5. Hybrid / Hierarchical: {[m for m in available_models if m in ['il_rl', 'mpc_learning', 'hierarchical_control']]}")
        print(f"6. Fundamental Architectures: {[m for m in available_models if m in ['mlp', 'cnn', 'gnn']]}")
        raise ValueError(f"Unknown model: {model_name}. Total available models: {len(available_models)}")
    
    return model_classes[model_name]


def generate_data(config: Dict[str, Any], logger: logging.Logger):
    """生成合成轨迹数据"""
    logger.info("开始生成合成轨迹数据...")
    
    data_generator = TrajectoryDataGenerator(config['data'])
    
    # 生成训练、验证和测试数据
    train_data, val_data, test_data = data_generator.generate_all_splits()
    
    # 保存数据
    data_dir = Path(config['experiment']['output_dir']) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_generator.save_data(train_data, data_dir / "train.h5")
    data_generator.save_data(val_data, data_dir / "val.h5")
    data_generator.save_data(test_data, data_dir / "test.h5")
    
    logger.info(f"数据生成完成! 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
    
    return train_data, val_data, test_data


def train_model(config: Dict[str, Any], model_name: str, logger: logging.Logger):
    """训练指定模型"""
    logger.info(f"开始训练模型: {model_name}")
    
    # 检查模型是否启用
    if not config['models'][model_name]['enabled']:
        logger.warning(f"模型 {model_name} 未启用，跳过训练")
        return None
    
    # 加载数据
    data_dir = Path(config['experiment']['output_dir']) / "data"
    dataset = TrajectoryDataset(
        train_path=data_dir / "train.h5",
        val_path=data_dir / "val.h5",
        test_path=data_dir / "test.h5",
        config=config['data']
    )
    
    # 创建模型
    model_class = get_model_class(model_name)
    model = model_class(config['models'][model_name])
    
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        dataset=dataset,
        config=config['training'],
        logger=logger
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    checkpoint_dir = Path(config['experiment']['output_dir']) / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_dir / "best_model.pth")
    
    logger.info(f"模型 {model_name} 训练完成")
    return model


def evaluate_model(config: Dict[str, Any], model_name: str, logger: logging.Logger):
    """评估指定模型"""
    logger.info(f"开始评估模型: {model_name}")
    
    # 加载数据
    data_dir = Path(config['experiment']['output_dir']) / "data"
    dataset = TrajectoryDataset(
        train_path=data_dir / "train.h5",
        val_path=data_dir / "val.h5", 
        test_path=data_dir / "test.h5",
        config=config['data']
    )
    
    # 加载模型
    model_class = get_model_class(model_name)
    model = model_class(config['models'][model_name])
    
    checkpoint_path = Path(config['experiment']['output_dir']) / "checkpoints" / model_name / "best_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"加载模型检查点: {checkpoint_path}")
    else:
        logger.warning(f"未找到模型检查点: {checkpoint_path}")
        return None
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model=model,
        dataset=dataset,
        config=config['evaluation'],
        logger=logger
    )
    
    # 进行评估
    results = evaluator.evaluate()
    
    # 保存结果
    results_dir = Path(config['experiment']['output_dir']) / "results" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "evaluation_results.yaml", 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"模型 {model_name} 评估完成")
    return results


def visualize_results(config: Dict[str, Any], logger: logging.Logger):
    """可视化所有模型的结果"""
    logger.info("开始可视化结果...")
    
    visualizer = TrajectoryVisualizer(config['visualization'])
    
    # 加载所有模型的评估结果
    results_dir = Path(config['experiment']['output_dir']) / "results"
    all_results = {}
    
    for model_name in config['models'].keys():
        if config['models'][model_name]['enabled']:
            result_file = results_dir / model_name / "evaluation_results.yaml"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    all_results[model_name] = yaml.safe_load(f)
    
    if not all_results:
        logger.warning("未找到任何评估结果，跳过可视化")
        return
    
    # 生成比较图表
    output_dir = Path(config['experiment']['output_dir']) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer.plot_model_comparison(all_results, output_dir)
    visualizer.plot_trajectory_samples(all_results, output_dir)
    
    logger.info("结果可视化完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="3D End-Effector Trajectory Generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["generate", "train", "evaluate", "visualize", "all"], 
                       default="all", help="运行模式")
    parser.add_argument("--model", type=str, help="指定模型名称 (仅在train/evaluate模式下有效)")
    parser.add_argument("--output-dir", type=str, help="输出目录 (覆盖配置文件设置)")
    parser.add_argument("--seed", type=int, help="随机种子 (覆盖配置文件设置)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], help="计算设备")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    validate_config(config)
    
    # 覆盖配置参数
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    if args.seed:
        config['experiment']['seed'] = args.seed
    if args.device:
        config['training']['device'] = args.device
    
    # 设置输出目录
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("trajectory_generation", output_dir / "logs", level=log_level)
    
    # 设置随机种子
    set_seed(config['experiment']['seed'])
    
    # 设置设备
    if config['training']['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['training']['device']
    
    logger.info(f"使用设备: {device}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"随机种子: {config['experiment']['seed']}")
    
    try:
        # 根据模式执行相应操作
        if args.mode in ["generate", "all"]:
            generate_data(config, logger)
        
        if args.mode in ["train", "all"]:
            if args.model:
                # 训练指定模型
                train_model(config, args.model, logger)
            else:
                # 训练所有启用的模型
                for model_name in config['models'].keys():
                    if config['models'][model_name]['enabled']:
                        train_model(config, model_name, logger)
        
        if args.mode in ["evaluate", "all"]:
            if args.model:
                # 评估指定模型
                evaluate_model(config, args.model, logger)
            else:
                # 评估所有启用的模型
                for model_name in config['models'].keys():
                    if config['models'][model_name]['enabled']:
                        evaluate_model(config, model_name, logger)
        
        if args.mode in ["visualize", "all"]:
            visualize_results(config, logger)
        
        logger.info("所有任务完成!")
        
    except Exception as e:
        logger.error(f"执行过程中出现错误: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()