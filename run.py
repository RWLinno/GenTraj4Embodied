#!/usr/bin/env python3
"""
3D End-Effector Trajectory Generation - Updated Main Runner
主运行脚本，支持新的5大类分类和扁平化baselines结构
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

# 导入所有模型 - 新的5大类分类框架 + 扁平化结构
try:
    # 使用统一的baselines导入
    from baselines import get_model_class, list_available_models, MODEL_REGISTRY
    
    # 如果统一导入失败，则使用单独导入
    UNIFIED_IMPORT = True
except ImportError:
    UNIFIED_IMPORT = False
    print("警告: 统一导入失败，使用单独导入模式")
    
    # 单独导入所有模型
    try:
        # Classical Methods
        from baselines.linear_interpolation_model import LinearInterpolationModel
        from baselines.spline_interpolation_model import SplineInterpolationModel
        from baselines.dmp_model import DMPModel
        from baselines.promp_model import ProMPModel
        from baselines.gmm_model import GMMModel
        
        # Fundamental Architectures
        from baselines.mlp_model import MLPModel
        from baselines.cnn_model import CNNModel
        from baselines.gnn_model import GNNModel
        from baselines.vae_model import VAEModel
        from baselines.conditional_vae_model import ConditionalVAEModel
        
        # Probabilistic Generative Models
        from baselines.diffusion_policy_model import DiffusionPolicyModel
        from baselines.ddpm_model import DDPMModel
        from baselines.ddim_model import DDIMModel
        from baselines.difftraj_model import DiffTrajModel
        from baselines.score_based_model import ScoreBasedModel
        from baselines.conditional_diffusion_model import ConditionalDiffusionModel
        from baselines.latent_diffusion_model import LatentDiffusionModel
        from baselines.normalizing_flows_model import NormalizingFlowsModel
        from baselines.gflownets_model import GFlowNetsModel
        from baselines.conditional_gan_model import ConditionalGANModel
        from baselines.conditional_flow_model import ConditionalFlowModel
        from baselines.kinematic_diffusion_model import KinematicDiffusionModel
        
        # Sequential Modeling
        from baselines.lstm_model import LSTMModel
        from baselines.gru_model import GRUModel
        from baselines.rnn_model import RNNModel
        from baselines.transformer_model import TransformerModel
        from baselines.gpt_model import GPTModel
        from baselines.bert_model import BERTModel
        from baselines.decision_transformer_model import DecisionTransformerModel
        from baselines.mdn_model import MDNModel
        from baselines.seq2seq_model import Seq2SeqModel
        from baselines.conditional_transformer_model import ConditionalTransformerModel
        
        # Hybrid_Hierarchical
        from baselines.policy_gradient_model import PolicyGradientModel
        from baselines.actor_critic_model import ActorCriticModel
        from baselines.ppo_model import PPOModel
        from baselines.imitation_learning_model import ImitationLearningModel
        from baselines.inverse_rl_model import InverseRLModel
        from baselines.il_rl_hybrid_model import ILRLHybridModel
        from baselines.mpc_learning_model import MPCLearningModel
        
        # Additional models
        from baselines.mpc_model import MPCModel
        from baselines.rrt_model import RRTModel
        from baselines.prm_model import PRMModel
        
    except ImportError as e:
        print(f"警告: 部分模型导入失败: {e}")


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


def get_model_class_fallback(model_name: str):
    """备用的模型类获取函数 - 新的5大类分类框架"""
    model_classes = {
        # 1. Classical Methods
        'linear_interpolation': LinearInterpolationModel,
        'spline_interpolation': SplineInterpolationModel,
        'dmp': DMPModel,
        'promp': ProMPModel,
        'gmm': GMMModel,
        
        # 2. Fundamental Architectures
        'mlp': MLPModel,
        'cnn': CNNModel,
        'gnn': GNNModel,
        'vae': VAEModel,
        'conditional_vae': ConditionalVAEModel,
        
        # 3. Probabilistic Generative Models
        'diffusion_policy': DiffusionPolicyModel,
        'ddpm': DDPMModel,
        'ddim': DDIMModel,
        'difftraj': DiffTrajModel,
        'score_based': ScoreBasedModel,
        'conditional_diffusion': ConditionalDiffusionModel,
        'latent_diffusion': LatentDiffusionModel,
        'normalizing_flows': NormalizingFlowsModel,
        'gflownets': GFlowNetsModel,
        'conditional_gan': ConditionalGANModel,
        'conditional_flow': ConditionalFlowModel,
        'kinematic_diffusion': KinematicDiffusionModel,
        
        # 4. Sequential Modeling
        'lstm': LSTMModel,
        'gru': GRUModel,
        'rnn': RNNModel,
        'transformer': TransformerModel,
        'gpt': GPTModel,
        'bert': BERTModel,
        'decision_transformer': DecisionTransformerModel,
        'mdn': MDNModel,
        'seq2seq': Seq2SeqModel,
        'conditional_transformer': ConditionalTransformerModel,
        
        # 5. Hybrid_Hierarchical
        'policy_gradient': PolicyGradientModel,
        'actor_critic': ActorCriticModel,
        'ppo': PPOModel,
        'imitation_learning': ImitationLearningModel,
        'inverse_rl': InverseRLModel,
        'il_rl_hybrid': ILRLHybridModel,
        'mpc_learning': MPCLearningModel,
        'mpc': MPCModel,
        'rrt': RRTModel,
        'prm': PRMModel,
    }
    
    if model_name not in model_classes:
        available_models = list(model_classes.keys())
        print(f"\n可用的模型类别 (5大类分类):")
        print(f"1. Classical Methods: {[m for m in available_models if m in ['linear_interpolation', 'spline_interpolation', 'dmp', 'promp', 'gmm']]}")
        print(f"2. Fundamental Architectures: {[m for m in available_models if m in ['mlp', 'cnn', 'gnn', 'vae', 'conditional_vae']]}")
        print(f"3. Probabilistic Generative Models: {[m for m in available_models if m in ['diffusion_policy', 'ddpm', 'ddim', 'difftraj', 'score_based', 'conditional_diffusion', 'latent_diffusion', 'normalizing_flows', 'gflownets', 'conditional_gan', 'conditional_flow', 'kinematic_diffusion']]}")
        print(f"4. Sequential Modeling: {[m for m in available_models if m in ['lstm', 'gru', 'rnn', 'transformer', 'gpt', 'bert', 'decision_transformer', 'mdn', 'seq2seq', 'conditional_transformer']]}")
        print(f"5. Hybrid_Hierarchical: {[m for m in available_models if m in ['policy_gradient', 'actor_critic', 'ppo', 'imitation_learning', 'inverse_rl', 'il_rl_hybrid', 'mpc_learning', 'mpc', 'rrt', 'prm']]}")
        raise ValueError(f"Unknown model: {model_name}. Total available models: {len(available_models)}")
    
    return model_classes[model_name]


def get_model_class_unified(model_name: str):
    """使用统一导入的模型类获取函数"""
    if UNIFIED_IMPORT:
        return get_model_class(model_name)
    else:
        return get_model_class_fallback(model_name)


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
    if model_name not in config['models']:
        logger.error(f"模型 {model_name} 不在配置文件中")
        return None
        
    if not config['models'][model_name]['enabled']:
        logger.warning(f"模型 {model_name} 未启用，跳过训练")
        return None
    
    try:
        # 加载数据
        data_dir = Path(config['experiment']['output_dir']) / "data"
        dataset = TrajectoryDataset(
            train_path=data_dir / "train.h5",
            val_path=data_dir / "val.h5",
            test_path=data_dir / "test.h5",
            config=config['data']
        )
        
        # 创建模型
        model_class = get_model_class_unified(model_name)
        model_config = config['models'][model_name].copy()
        model_config['trajectory_length'] = config['data']['generation']['trajectory_length']
        model = model_class(model_config)
        
        logger.info(f"模型 {model_name} 创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
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
        
    except Exception as e:
        logger.error(f"训练模型 {model_name} 时出错: {str(e)}")
        return None


def evaluate_model(config: Dict[str, Any], model_name: str, logger: logging.Logger):
    """评估指定模型"""
    logger.info(f"开始评估模型: {model_name}")
    
    try:
        # 加载数据
        data_dir = Path(config['experiment']['output_dir']) / "data"
        dataset = TrajectoryDataset(
            train_path=data_dir / "train.h5",
            val_path=data_dir / "val.h5", 
            test_path=data_dir / "test.h5",
            config=config['data']
        )
        
        # 加载模型
        model_class = get_model_class_unified(model_name)
        model_config = config['models'][model_name].copy()
        model_config['trajectory_length'] = config['data']['generation']['trajectory_length']
        model = model_class(model_config)
        
        checkpoint_path = Path(config['experiment']['output_dir']) / "checkpoints" / model_name / "best_model.pth"
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
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
        
    except Exception as e:
        logger.error(f"评估模型 {model_name} 时出错: {str(e)}")
        return None


def test_model_import(config: Dict[str, Any], logger: logging.Logger):
    """测试所有模型的导入和初始化"""
    logger.info("开始测试模型导入和初始化...")
    
    success_count = 0
    fail_count = 0
    failed_models = []
    
    for model_name in config['models'].keys():
        if not config['models'][model_name]['enabled']:
            continue
            
        try:
            # 尝试获取模型类
            model_class = get_model_class_unified(model_name)
            
            # 尝试创建模型实例
            model_config = config['models'][model_name].copy()
            model_config['trajectory_length'] = config['data']['generation']['trajectory_length']
            model = model_class(model_config)
            
            # 获取模型信息
            model_info = model.get_model_info()
            
            logger.info(f"✓ {model_name}: {model_info.get('model_type', 'Unknown')} - {model_info.get('total_parameters', 0)} 参数")
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ {model_name}: {str(e)}")
            failed_models.append(model_name)
            fail_count += 1
    
    logger.info(f"模型测试完成: 成功 {success_count}, 失败 {fail_count}")
    
    if failed_models:
        logger.warning(f"失败的模型: {failed_models}")
    
    return success_count, fail_count, failed_models


def generate_sample_trajectories(config: Dict[str, Any], logger: logging.Logger):
    """为所有模型生成样本轨迹"""
    logger.info("开始生成样本轨迹...")
    
    # 定义测试用的起点和终点
    start_pose = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])  # x, y, z, qx, qy, qz, qw
    end_pose = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    
    output_dir = Path(config['experiment']['output_dir']) / "sample_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for model_name in config['models'].keys():
        if not config['models'][model_name]['enabled']:
            continue
            
        try:
            # 创建模型
            model_class = get_model_class_unified(model_name)
            model_config = config['models'][model_name].copy()
            model_config['trajectory_length'] = config['data']['generation']['trajectory_length']
            model = model_class(model_config)
            
            # 生成轨迹
            trajectories = model.generate_trajectory(start_pose, end_pose, num_samples=3)
            
            # 保存轨迹
            np.save(output_dir / f"{model_name}_trajectories.npy", trajectories)
            
            logger.info(f"✓ {model_name}: 生成轨迹形状 {trajectories.shape}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ {model_name}: 轨迹生成失败 - {str(e)}")
    
    logger.info(f"轨迹生成完成: 成功 {success_count} 个模型")


def visualize_results(config: Dict[str, Any], logger: logging.Logger):
    """可视化所有模型的结果"""
    logger.info("开始可视化结果...")
    
    try:
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
        
    except Exception as e:
        logger.error(f"可视化过程中出错: {str(e)}")


def print_model_categories(config: Dict[str, Any]):
    """打印模型分类信息"""
    print("\n=== 5大类轨迹生成模型分类 ===")
    
    if 'model_categories' in config:
        for category_name, category_info in config['model_categories'].items():
            print(f"\n{category_name}:")
            print(f"  描述: {category_info['description']}")
            print(f"  模型: {category_info['models']}")
            if 'characteristics' in category_info:
                print(f"  特点: {category_info['characteristics']}")
    else:
        print("配置文件中未找到model_categories信息")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="3D End-Effector Trajectory Generation (5-Category Framework)")
    parser.add_argument("--config", type=str, default="config_5categories.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["generate", "train", "evaluate", "visualize", "test", "sample", "all"], 
                       default="test", help="运行模式")
    parser.add_argument("--model", type=str, help="指定模型名称 (仅在train/evaluate模式下有效)")
    parser.add_argument("--output-dir", type=str, help="输出目录 (覆盖配置文件设置)")
    parser.add_argument("--seed", type=int, help="随机种子 (覆盖配置文件设置)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], help="计算设备")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--list-models", action="store_true", help="列出所有可用模型")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
        validate_config(config)
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        # 使用默认配置
        config = {
            'experiment': {'output_dir': 'experiments_5categories', 'seed': 42},
            'training': {'device': 'cpu'},
            'data': {'generation': {'trajectory_length': 50}},
            'models': {},
            'evaluation': {},
            'visualization': {}
        }
    
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
    logger = setup_logger("trajectory_generation_5categories", output_dir / "logs", level=log_level)
    
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
    logger.info(f"配置文件: {args.config}")
    
    # 打印模型分类信息
    if args.list_models or args.mode == "test":
        print_model_categories(config)
        
        if UNIFIED_IMPORT:
            available_models = list_available_models()
            print(f"\n可用模型 ({len(available_models)}个): {available_models}")
        else:
            print("\n使用备用导入模式")
    
    try:
        # 根据模式执行相应操作
        if args.mode in ["generate", "all"]:
            generate_data(config, logger)
        
        if args.mode in ["test"]:
            test_model_import(config, logger)
        
        if args.mode in ["sample"]:
            generate_sample_trajectories(config, logger)
        
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