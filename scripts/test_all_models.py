#!/usr/bin/env python3
"""
Test Script for All 25+ Trajectory Generation Models
测试脚本，验证所有25+轨迹生成模型的实现正确性
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import traceback

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logger import setup_logger


def test_model_import_and_initialization():
    """测试所有模型的导入和初始化"""
    print("=" * 60)
    print("测试模型导入和初始化")
    print("=" * 60)
    
    # 定义所有模型的导入信息
    model_imports = {
        # 原有模型
        'diffusion_policy': ('baselines.diffusion_policy.model', 'DiffusionPolicyModel'),
        'transformer': ('baselines.transformer.model', 'TransformerTrajectoryModel'),
        'vae': ('baselines.vae.model', 'VAETrajectoryModel'),
        'mlp': ('baselines.mlp.model', 'MLPTrajectoryModel'),
        'gflownets': ('baselines.gflownets.model', 'GFlowNetTrajectoryModel'),
        
        # Classical Methods
        'linear_interpolation': ('baselines.classical_methods.linear_interpolation', 'LinearInterpolationModel'),
        'spline_interpolation': ('baselines.classical_methods.spline_interpolation', 'SplineInterpolationModel'),
        'dmp': ('baselines.classical_methods.dmp_model', 'DMPTrajectoryModel'),
        'promp': ('baselines.classical_methods.promp_model', 'ProMPTrajectoryModel'),
        'gmm': ('baselines.classical_methods.gmm_model', 'GMMTrajectoryModel'),
        
        # Linear Architecture
        'lstm': ('baselines.linear_architecture.lstm_model', 'LSTMTrajectoryModel'),
        'gru': ('baselines.linear_architecture.gru_model', 'GRUTrajectoryModel'),
        'rnn': ('baselines.linear_architecture.rnn_model', 'RNNTrajectoryModel'),
        'mdn': ('baselines.linear_architecture.mdn_model', 'MDNTrajectoryModel'),
        'seq2seq': ('baselines.linear_architecture.seq2seq_model', 'Seq2SeqTrajectoryModel'),
        
        # Transformer Variants
        'gpt_style': ('baselines.transformer_variants.gpt_style_model', 'GPTStyleTrajectoryModel'),
        'bert_style': ('baselines.transformer_variants.bert_style_model', 'BERTStyleTrajectoryModel'),
        'positional_transformer': ('baselines.transformer_variants.positional_transformer', 'PositionalTransformerModel'),
        'multihead_attention': ('baselines.transformer_variants.multihead_attention_model', 'MultiHeadAttentionTrajectoryModel'),
        'hierarchical_transformer': ('baselines.transformer_variants.hierarchical_transformer', 'HierarchicalTransformerModel'),
        
        # Diffusion Variants
        'ddpm': ('baselines.diffusion_variants.ddpm_model', 'DDPMTrajectoryModel'),
        'frmd': ('baselines.diffusion_variants.frmd_model', 'FRMDTrajectoryModel'),
        'score_based': ('baselines.diffusion_variants.score_based_model', 'ScoreBasedTrajectoryModel'),
        'conditional_diffusion': ('baselines.diffusion_variants.conditional_diffusion_model', 'ConditionalDiffusionTrajectoryModel'),
        'latent_diffusion': ('baselines.diffusion_variants.latent_diffusion_model', 'LatentDiffusionTrajectoryModel'),
        
        # RL-based Methods
        'policy_gradient': ('baselines.rl_methods.policy_gradient_model', 'PolicyGradientTrajectoryModel'),
        'actor_critic': ('baselines.rl_methods.actor_critic_model', 'ActorCriticTrajectoryModel'),
        'imitation_learning': ('baselines.rl_methods.imitation_learning_model', 'ImitationLearningTrajectoryModel'),
        'inverse_rl': ('baselines.rl_methods.inverse_rl_model', 'InverseRLTrajectoryModel'),
        'ppo': ('baselines.rl_methods.ppo_model', 'PPOTrajectoryModel')
    }
    
    successful_imports = []
    failed_imports = []
    
    for model_name, (module_path, class_name) in model_imports.items():
        try:
            # 尝试导入模块
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # 测试基本配置初始化
            basic_config = {
                'input_dim': 7,
                'output_dim': 7,
                'max_seq_length': 50,
                'device': 'cpu'
            }
            
            # 尝试创建模型实例
            model = model_class(basic_config)
            
            print(f"✅ {model_name}: 导入和初始化成功")
            successful_imports.append(model_name)
            
        except Exception as e:
            print(f"❌ {model_name}: 失败 - {str(e)}")
            failed_imports.append((model_name, str(e)))
    
    print(f"\n总结: {len(successful_imports)}/{len(model_imports)} 模型导入成功")
    
    if failed_imports:
        print("\n失败的模型:")
        for model_name, error in failed_imports:
            print(f"  - {model_name}: {error}")
    
    return successful_imports, failed_imports


def test_model_forward_pass():
    """测试所有模型的前向传播"""
    print("\n" + "=" * 60)
    print("测试模型前向传播")
    print("=" * 60)
    
    successful_imports, _ = test_model_import_and_initialization()
    
    # 准备测试数据
    batch_size = 2
    input_dim = 7
    output_dim = 7
    seq_length = 50
    
    start_pose = torch.randn(batch_size, input_dim)
    end_pose = torch.randn(batch_size, input_dim)
    
    successful_forward = []
    failed_forward = []
    
    for model_name in successful_imports:
        try:
            # 获取模型类
            from run import get_model_class
            model_class = get_model_class(model_name)
            
            # 创建模型
            config = {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'max_seq_length': seq_length,
                'device': 'cpu',
                # 模型特定配置
                'hidden_dim': 128,
                'd_model': 256,
                'nhead': 4,
                'num_layers': 2,
                'latent_dim': 32,
                'num_timesteps': 100,
                'gamma': 0.99
            }
            
            model = model_class(config)
            model.eval()
            
            # 测试前向传播
            with torch.no_grad():
                output = model.forward(start_pose, end_pose)
            
            # 验证输出形状
            expected_shape = (batch_size, seq_length, output_dim)
            if output.shape == expected_shape:
                print(f"✅ {model_name}: 前向传播成功 - 输出形状: {output.shape}")
                successful_forward.append(model_name)
            else:
                print(f"⚠️  {model_name}: 输出形状不匹配 - 期望: {expected_shape}, 实际: {output.shape}")
                failed_forward.append((model_name, f"形状不匹配: {output.shape}"))
                
        except Exception as e:
            print(f"❌ {model_name}: 前向传播失败 - {str(e)}")
            failed_forward.append((model_name, str(e)))
    
    print(f"\n总结: {len(successful_forward)}/{len(successful_imports)} 模型前向传播成功")
    
    return successful_forward, failed_forward


def test_model_trajectory_generation():
    """测试所有模型的轨迹生成功能"""
    print("\n" + "=" * 60)
    print("测试模型轨迹生成")
    print("=" * 60)
    
    successful_forward, _ = test_model_forward_pass()
    
    # 准备测试数据
    start_pose = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    end_pose = np.array([0.5, 0.3, 0.8, 1.0, 0.0, 0.0, 0.0])
    num_points = 30
    
    successful_generation = []
    failed_generation = []
    
    for model_name in successful_forward:
        try:
            # 获取模型类
            from run import get_model_class
            model_class = get_model_class(model_name)
            
            # 创建模型
            config = {
                'input_dim': 7,
                'output_dim': 7,
                'max_seq_length': 50,
                'device': 'cpu',
                'hidden_dim': 128,
                'd_model': 256,
                'nhead': 4,
                'num_layers': 2,
                'latent_dim': 32,
                'num_timesteps': 100,
                'gamma': 0.99
            }
            
            model = model_class(config)
            model.eval()
            
            # 测试轨迹生成
            trajectory = model.generate_trajectory(start_pose, end_pose, num_points)
            
            # 验证输出
            expected_shape = (num_points, 7)
            if trajectory.shape == expected_shape:
                # 验证边界条件
                start_error = np.linalg.norm(trajectory[0] - start_pose)
                end_error = np.linalg.norm(trajectory[-1] - end_pose)
                
                if start_error < 1e-3 and end_error < 1e-3:
                    print(f"✅ {model_name}: 轨迹生成成功 - 形状: {trajectory.shape}, 边界误差: {start_error:.6f}, {end_error:.6f}")
                    successful_generation.append(model_name)
                else:
                    print(f"⚠️  {model_name}: 边界条件不满足 - 起点误差: {start_error:.6f}, 终点误差: {end_error:.6f}")
                    failed_generation.append((model_name, f"边界条件不满足"))
            else:
                print(f"❌ {model_name}: 输出形状错误 - 期望: {expected_shape}, 实际: {trajectory.shape}")
                failed_generation.append((model_name, f"形状错误: {trajectory.shape}"))
                
        except Exception as e:
            print(f"❌ {model_name}: 轨迹生成失败 - {str(e)}")
            failed_generation.append((model_name, str(e)))
    
    print(f"\n总结: {len(successful_generation)}/{len(successful_forward)} 模型轨迹生成成功")
    
    return successful_generation, failed_generation


def test_model_training_step():
    """测试所有模型的训练步骤"""
    print("\n" + "=" * 60)
    print("测试模型训练步骤")
    print("=" * 60)
    
    successful_generation, _ = test_model_trajectory_generation()
    
    # 准备训练数据
    batch_size = 4
    seq_length = 30
    input_dim = 7
    output_dim = 7
    
    batch = {
        'start_pose': torch.randn(batch_size, input_dim),
        'end_pose': torch.randn(batch_size, input_dim),
        'trajectory': torch.randn(batch_size, seq_length, output_dim)
    }
    
    successful_training = []
    failed_training = []
    
    for model_name in successful_generation:
        try:
            # 获取模型类
            from run import get_model_class
            model_class = get_model_class(model_name)
            
            # 创建模型
            config = {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'max_seq_length': seq_length,
                'device': 'cpu',
                'hidden_dim': 128,
                'd_model': 256,
                'nhead': 4,
                'num_layers': 2,
                'latent_dim': 32,
                'num_timesteps': 100,
                'gamma': 0.99,
                'learning_rate': 1e-3
            }
            
            model = model_class(config)
            model.train()
            
            # 测试训练步骤
            loss_dict = model.train_step(batch)
            
            # 验证损失
            if isinstance(loss_dict, dict) and 'loss' in loss_dict:
                loss_value = loss_dict['loss']
                if isinstance(loss_value, (int, float)) and not np.isnan(loss_value):
                    print(f"✅ {model_name}: 训练步骤成功 - 损失: {loss_value:.6f}")
                    successful_training.append(model_name)
                else:
                    print(f"❌ {model_name}: 损失值无效 - {loss_value}")
                    failed_training.append((model_name, f"损失值无效: {loss_value}"))
            else:
                print(f"❌ {model_name}: 训练步骤返回格式错误 - {loss_dict}")
                failed_training.append((model_name, "训练步骤返回格式错误"))
                
        except Exception as e:
            print(f"❌ {model_name}: 训练步骤失败 - {str(e)}")
            failed_training.append((model_name, str(e)))
    
    print(f"\n总结: {len(successful_training)}/{len(successful_generation)} 模型训练步骤成功")
    
    return successful_training, failed_training


def test_model_categories():
    """测试模型分类"""
    print("\n" + "=" * 60)
    print("测试模型分类")
    print("=" * 60)
    
    model_categories = {
        "Classical Methods": ["linear_interpolation", "spline_interpolation", "dmp", "promp", "gmm", "vae", "mlp", "gflownets"],
        "Linear Architecture": ["lstm", "gru", "rnn", "mdn", "seq2seq"],
        "Transformer Architecture": ["transformer", "gpt_style", "bert_style", "positional_transformer", "multihead_attention", "hierarchical_transformer"],
        "Diffusion-based Methods": ["diffusion_policy", "ddpm", "frmd", "score_based", "conditional_diffusion", "latent_diffusion"],
        "RL-based Methods": ["policy_gradient", "actor_critic", "imitation_learning", "inverse_rl", "ppo"]
    }
    
    total_models = 0
    for category, models in model_categories.items():
        print(f"\n{category}: {len(models)} 个模型")
        for model in models:
            print(f"  - {model}")
        total_models += len(models)
    
    print(f"\n总模型数量: {total_models}")
    
    return model_categories


def create_test_report(successful_training: List[str], failed_training: List[Tuple[str, str]]):
    """创建测试报告"""
    print("\n" + "=" * 60)
    print("创建测试报告")
    print("=" * 60)
    
    # 创建测试报告
    report = {
        'test_summary': {
            'total_models': len(successful_training) + len(failed_training),
            'successful_models': len(successful_training),
            'failed_models': len(failed_training),
            'success_rate': len(successful_training) / (len(successful_training) + len(failed_training)) * 100
        },
        'successful_models': successful_training,
        'failed_models': [{'model': name, 'error': error} for name, error in failed_training],
        'model_categories': test_model_categories()
    }
    
    # 保存报告
    output_dir = Path("trajectory_generation_project/code/test_results")
    output_dir.mkdir(exist_ok=True)
    
    import json
    with open(output_dir / "model_test_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # 创建Markdown报告
    markdown_content = f"""# 轨迹生成模型测试报告

## 测试摘要

- **总模型数量**: {report['test_summary']['total_models']}
- **成功模型数量**: {report['test_summary']['successful_models']}
- **失败模型数量**: {report['test_summary']['failed_models']}
- **成功率**: {report['test_summary']['success_rate']:.1f}%

## 成功的模型

"""
    
    for model in successful_training:
        markdown_content += f"- ✅ {model}\n"
    
    if failed_training:
        markdown_content += "\n## 失败的模型\n\n"
        for model_name, error in failed_training:
            markdown_content += f"- ❌ {model_name}: {error}\n"
    
    markdown_content += "\n## 模型分类\n\n"
    
    for category, models in report['model_categories'].items():
        markdown_content += f"### {category} ({len(models)} 个模型)\n\n"
        for model in models:
            status = "✅" if model in successful_training else "❌"
            markdown_content += f"- {status} {model}\n"
        markdown_content += "\n"
    
    with open(output_dir / "model_test_report.md", 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"测试报告已保存到: {output_dir}")
    
    return report


def main():
    """主函数"""
    print("开始测试所有25+轨迹生成模型...")
    
    try:
        # 1. 测试模型导入和初始化
        successful_imports, failed_imports = test_model_import_and_initialization()
        
        # 2. 测试前向传播
        successful_forward, failed_forward = test_model_forward_pass()
        
        # 3. 测试训练步骤
        successful_training, failed_training = test_model_training_step()
        
        # 4. 创建测试报告
        report = create_test_report(successful_training, failed_training)
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        print(f"成功率: {report['test_summary']['success_rate']:.1f}%")
        print(f"成功模型: {report['test_summary']['successful_models']}/{report['test_summary']['total_models']}")
        
        if report['test_summary']['success_rate'] >= 80:
            print("🎉 测试结果良好! 大部分模型工作正常")
        elif report['test_summary']['success_rate'] >= 60:
            print("⚠️  测试结果一般，需要修复部分模型")
        else:
            print("🚨 测试结果较差，需要大量修复工作")
        
        return report
        
    except Exception as e:
        print(f"测试过程中出现严重错误: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()