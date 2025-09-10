#!/usr/bin/env python3
"""
Trajectory Generation Visualization Script
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional

sys.path.append(str(Path(__file__).parent / "src"))

from src.data.dataset import TrajectoryDataset
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.visualization import TrajectoryVisualizer
from src.evaluation.evaluator import ModelEvaluator
from baselines import get_model_class, list_available_models


class TrajectoryVisualizer3D:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_trajectory_comparison(self, 
                                 generated_traj: np.ndarray, 
                                 ground_truth_traj: np.ndarray,
                                 model_name: str = "Model",
                                 save_path: Optional[Path] = None) -> plt.Figure:
        fig = plt.figure(figsize=(15, 10))
        
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(ground_truth_traj[:, 0], ground_truth_traj[:, 1], ground_truth_traj[:, 2], 
                'b-', linewidth=2, label=' Ground Truth', alpha=0.8)
        ax1.scatter(ground_truth_traj[0, 0], ground_truth_traj[0, 1], ground_truth_traj[0, 2], 
                   c='blue', s=100, marker='o', label=' GT Start')
        ax1.scatter(ground_truth_traj[-1, 0], ground_truth_traj[-1, 1], ground_truth_traj[-1, 2], 
                   c='blue', s=100, marker='s', label=' GT End')
        
        ax1.plot(generated_traj[:, 0], generated_traj[:, 1], generated_traj[:, 2], 
                'r--', linewidth=2, label=model_name+' Generated', alpha=0.8)
        ax1.scatter(generated_traj[0, 0], generated_traj[0, 1], generated_traj[0, 2], 
                   c='red', s=100, marker='o', label=model_name+' Start')
        ax1.scatter(generated_traj[-1, 0], generated_traj[-1, 1], generated_traj[-1, 2], 
                   c='red', s=100, marker='s', label=model_name+' End')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Position')
        ax1.legend()
        ax1.grid(True)
        
        ax2 = fig.add_subplot(222)
        ax2.plot(ground_truth_traj[:, 0], ground_truth_traj[:, 1], 'b-', linewidth=2, label='Ground Truth')
        ax2.plot(generated_traj[:, 0], generated_traj[:, 1], 'r--', linewidth=2, label=model_name+'Generated')
        ax2.scatter(ground_truth_traj[0, 0], ground_truth_traj[0, 1], c='blue', s=100, marker='o')
        ax2.scatter(generated_traj[0, 0], generated_traj[0, 1], c='red', s=100, marker='o')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('2D Plane Projection')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        ax3 = fig.add_subplot(223)
        t = np.arange(len(ground_truth_traj))
        ax3.plot(t, ground_truth_traj[:, 0], 'b-', label='GT X', alpha=0.8)
        ax3.plot(t, ground_truth_traj[:, 1], 'g-', label='GT Y', alpha=0.8)
        ax3.plot(t, ground_truth_traj[:, 2], 'r-', label='GT Z', alpha=0.8)
        ax3.plot(t, generated_traj[:, 0], 'b--', label=model_name+' Gen X', alpha=0.8)
        ax3.plot(t, generated_traj[:, 1], 'g--', label=model_name+' Gen Y', alpha=0.8)
        ax3.plot(t, generated_traj[:, 2], 'r--', label=model_name+' Gen Z', alpha=0.8)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Position (m)')
        ax3.set_title('Position vs Time')
        ax3.legend()
        ax3.grid(True)
        
        ax4 = fig.add_subplot(224)
        position_error = np.linalg.norm(generated_traj[:, :3] - ground_truth_traj[:, :3], axis=1)
        ax4.plot(t, position_error, 'r-', linewidth=2)
        ax4.fill_between(t, position_error, alpha=0.3, color='red')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Position Error (m)')
        ax4.set_title(f'Position Error (Mean: {np.mean(position_error):.4f}m)')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class TrajectoryMetrics:
    @staticmethod
    def compute_position_error(generated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """计算位置误差指标"""
        pos_error = np.linalg.norm(generated[:, :3] - ground_truth[:, :3], axis=1)        
        return {
            'mean_position_error': float(np.mean(pos_error)),
            'max_position_error': float(np.max(pos_error)),
            'rmse_position': float(np.sqrt(np.mean(pos_error**2))),
            'final_position_error': float(pos_error[-1])
        }
    
    @staticmethod
    def compute_smoothness(trajectory: np.ndarray) -> Dict[str, float]:
        vel = np.diff(trajectory[:, :3], axis=0)
        acc = np.diff(vel, axis=0)
        
        vel_norm = np.linalg.norm(vel, axis=1)
        acc_norm = np.linalg.norm(acc, axis=1)
        
        return {
            'mean_velocity': float(np.mean(vel_norm)),
            'max_velocity': float(np.max(vel_norm)),
            'mean_acceleration': float(np.mean(acc_norm)),
            'max_acceleration': float(np.max(acc_norm)),
            'jerk': float(np.mean(np.linalg.norm(np.diff(acc, axis=0), axis=1)))
        }
    
    @staticmethod
    def compute_path_efficiency(generated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """计算路径效率"""
        gen_length = np.sum(np.linalg.norm(np.diff(generated[:, :3], axis=0), axis=1))
        gt_length = np.sum(np.linalg.norm(np.diff(ground_truth[:, :3], axis=0), axis=1))
        
        start_pos = ground_truth[0, :3]
        end_pos = ground_truth[-1, :3]
        straight_distance = np.linalg.norm(end_pos - start_pos)
        
        return {
            'generated_path_length': float(gen_length),
            'ground_truth_path_length': float(gt_length),
            'straight_line_distance': float(straight_distance),
            'path_efficiency_gen': float(straight_distance / gen_length) if gen_length > 0 else 0.0,
            'path_efficiency_gt': float(straight_distance / gt_length) if gt_length > 0 else 0.0,
            'length_ratio': float(gen_length / gt_length) if gt_length > 0 else 0.0
        }


def load_model_from_checkpoint(model_name: str, config: Dict[str, Any], checkpoint_path: Path) -> torch.nn.Module:
    model_class = get_model_class(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = config['models'][model_name].copy()
    model_config['trajectory_length'] = config['data']['generation']['trajectory_length']
    model = model_class(model_config)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model state from checkpoint: {checkpoint_path}")
            print(f"Checkpoint info - Epoch: {checkpoint.get('epoch', 'N/A')}, Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model


def generate_sample_trajectories(model: torch.nn.Module, dataset: TrajectoryDataset, 
                               num_samples: int = 10) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """生成样本轨迹"""
    generated_trajectories = []
    ground_truth_trajectories = []
    
    print(f"开始生成 {num_samples} 个轨迹样本...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                start_pose = sample['start_pose'].unsqueeze(0)  # [1, 7]
                end_pose = sample['end_pose'].unsqueeze(0)      # [1, 7]
                gt_traj = sample['trajectory'].numpy()          # [seq_len, 7]
                
                print(f"样本 {i}: 输入维度 {start_pose.shape}, 真实轨迹 {gt_traj.shape}")
                
                generated_tensor = model(start_pose, end_pose)  # 应该输出 [1, 50, 7]
                generated = generated_tensor.numpy()[0]         # [50, 7]
                
                print(f"样本 {i}: 生成轨迹 {generated.shape}")
                
                gt_len = gt_traj.shape[0]
                gen_len = generated.shape[0]
                
                if gen_len != gt_len:
                    print(f"样本 {i}: 调整轨迹长度从 {gen_len} 到 {gt_len}")
                    import torch.nn.functional as F
                    
                    gen_torch = torch.from_numpy(generated).float().unsqueeze(0).transpose(1, 2)  # [1, 7, gen_len]
                    gen_resized = F.interpolate(gen_torch, size=gt_len, mode='linear', align_corners=True)
                    generated = gen_resized.squeeze(0).transpose(0, 1).numpy()  # [gt_len, 7]
                    
                    print(f"样本 {i}: 调整后轨迹 {generated.shape}")
                
                pos_range = np.ptp(generated[:, :3], axis=0)  # 位置变化范围
                quat_norms = np.linalg.norm(generated[:, 3:], axis=1)  # 四元数模长
                
                if np.all(pos_range > 1e-6) and np.all(quat_norms > 0.9):
                    generated_trajectories.append(generated)
                    ground_truth_trajectories.append(gt_traj)
                    print(f"✓ 样本 {i}: 轨迹生成成功")
                else:
                    print(f"✗ 样本 {i}: 轨迹不合理，跳过")
                    
            except Exception as e:
                print(f"✗ 样本 {i}: 生成失败 - {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"轨迹生成完成: 成功 {len(generated_trajectories)}/{num_samples}")
    return generated_trajectories, ground_truth_trajectories


def evaluate_model_performance(generated_trajectories: List[np.ndarray], 
                             ground_truth_trajectories: List[np.ndarray]) -> Dict[str, Any]:
    """评估模型性能"""
    if len(generated_trajectories) == 0:
        return {
            'individual_metrics': [],
            'average_metrics': {},
            'num_samples': 0
        }
    
    metrics = TrajectoryMetrics()
    all_metrics = []
    
    for gen_traj, gt_traj in zip(generated_trajectories, ground_truth_trajectories):
        traj_metrics = {}
        
        traj_metrics.update(metrics.compute_position_error(gen_traj, gt_traj))
        
        gen_smoothness = metrics.compute_smoothness(gen_traj)
        gt_smoothness = metrics.compute_smoothness(gt_traj)
        traj_metrics.update({f'gen_{k}': v for k, v in gen_smoothness.items()})
        traj_metrics.update({f'gt_{k}': v for k, v in gt_smoothness.items()})
        
        traj_metrics.update(metrics.compute_path_efficiency(gen_traj, gt_traj))
        
        all_metrics.append(traj_metrics)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f'std_{key}'] = np.std([m[key] for m in all_metrics])
    
    return {
        'individual_metrics': all_metrics,
        'average_metrics': avg_metrics,
        'num_samples': len(all_metrics)
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Trajectory Generation Visualization")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--checkpoint", type=str, help="checkpoint文件路径")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="输出目录")
    parser.add_argument("--num-samples", type=int, default=5, help="生成样本数量")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--seed", type=int, help="随机种子 (用于找到对应的checkpoint)")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        return
    
    output_dir = Path(args.output_dir+'/'+args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("trajectory_visualization", output_dir / "logs")
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"使用设备: {device}")
    logger.info(f"输出目录: {output_dir}")
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # 确定种子
        seed = args.seed if args.seed is not None else config['experiment']['seed']
        
        # 查找checkpoint文件 - 使用新的命名规范
        checkpoint_dir = Path(config['experiment']['output_dir']) / "checkpoints" / f"{args.model}_{seed}"
        checkpoint_files = list(checkpoint_dir.glob(f"{args.model}_ckpt_epoch_*.pth"))
        
        if not checkpoint_files:
            # 如果没找到新格式的，尝试查找最终模型
            final_checkpoint = checkpoint_dir / f"{args.model}_ckpt_final.pth"
            if final_checkpoint.exists():
                checkpoint_files = [final_checkpoint]
        
        if checkpoint_files:
            # 选择最新的epoch checkpoint
            if len(checkpoint_files) == 1:
                checkpoint_path = checkpoint_files[0]
            else:
                checkpoint_path = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]) if 'epoch' in x.stem else 999)
        else:
            logger.error(f"未找到checkpoint文件，查找路径: {checkpoint_dir}")
            logger.error("请确保模型已训练并保存了checkpoint")
            return
    logger.info(f"使用checkpoint: {checkpoint_path}")
    
    try:
        model = load_model_from_checkpoint(args.model, config, checkpoint_path)
        logger.info(f"模型 {args.model} 加载成功")
        
        data_dir = Path(config['experiment']['output_dir']) / "data"
        dataset = TrajectoryDataset(
            train_path=data_dir / "train.h5",
            val_path=data_dir / "val.h5",
            test_path=data_dir / "test.h5",
            config=config['data'],
            mode='test'
        )
        logger.info("数据集加载成功")
        
        logger.info(f"开始生成 {args.num_samples} 个样本轨迹...")
        start_time = time.time()
        
        generated_trajectories, ground_truth_trajectories = generate_sample_trajectories(
            model, dataset, args.num_samples
        )
        
        generation_time = time.time() - start_time
        logger.info(f"轨迹生成完成，耗时: {generation_time:.2f}s")
        logger.info(f"成功生成 {len(generated_trajectories)} 个轨迹")
        
        visualizer = TrajectoryVisualizer3D(config.get('visualization', {}))
        
        logger.info("开始生成可视化...")
        
        for i, (gen_traj, gt_traj) in enumerate(zip(generated_trajectories, ground_truth_trajectories)):
            if i < args.num_samples:
                fig = visualizer.plot_trajectory_comparison(
                    gen_traj, gt_traj,
                    model_name=args.model,
                    save_path=output_dir / f"{args.model}_sample_{i+1}.png"
                )
                plt.close(fig)
        
        logger.info("开始评估模型性能...")
        performance_metrics = evaluate_model_performance(generated_trajectories, ground_truth_trajectories)
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        clean_metrics = convert_numpy_types(performance_metrics)
        
        results_file = output_dir / f"evaluation_results_{args.model}.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(clean_metrics, f, default_flow_style=False)
        
        avg_metrics = performance_metrics['average_metrics']
        print("\n" + "="*60)
        print(f"模型评估结果 - {args.model.upper()}")
        print("="*60)
        print(f"样本数量: {performance_metrics['num_samples']}")
        print(f"生成时间: {generation_time:.2f}s")
        if len(generated_trajectories) > 0:
            print(f"平均生成时间: {generation_time/len(generated_trajectories):.4f}s/trajectory")
        else:
            print("平均生成时间: N/A (没有成功生成轨迹)")
        
        if len(generated_trajectories) > 0:
            print("\n位置误差:")
            print(f"  平均位置误差: {avg_metrics['avg_mean_position_error']:.4f} ± {avg_metrics['std_mean_position_error']:.4f} m")
            print(f"  最大位置误差: {avg_metrics['avg_max_position_error']:.4f} ± {avg_metrics['std_max_position_error']:.4f} m")
            print(f"  RMSE位置误差: {avg_metrics['avg_rmse_position']:.4f} ± {avg_metrics['std_rmse_position']:.4f} m")
            print(f"  终点位置误差: {avg_metrics['avg_final_position_error']:.4f} ± {avg_metrics['std_final_position_error']:.4f} m")
            
            print("\n路径效率:")
            print(f"  路径长度比: {avg_metrics['avg_length_ratio']:.4f} ± {avg_metrics['std_length_ratio']:.4f}")
            print(f"  生成轨迹效率: {avg_metrics['avg_path_efficiency_gen']:.4f} ± {avg_metrics['std_path_efficiency_gen']:.4f}")
            print(f"  真实轨迹效率: {avg_metrics['avg_path_efficiency_gt']:.4f} ± {avg_metrics['std_path_efficiency_gt']:.4f}")
            
            print("\n平滑度对比:")
            print(f"  生成轨迹平均速度: {avg_metrics['avg_gen_mean_velocity']:.4f} ± {avg_metrics['std_gen_mean_velocity']:.4f} m/s")
            print(f"  真实轨迹平均速度: {avg_metrics['avg_gt_mean_velocity']:.4f} ± {avg_metrics['std_gt_mean_velocity']:.4f} m/s")
            print(f"  生成轨迹平均加速度: {avg_metrics['avg_gen_mean_acceleration']:.4f} ± {avg_metrics['std_gen_mean_acceleration']:.4f} m/s²")
            print(f"  真实轨迹平均加速度: {avg_metrics['avg_gt_mean_acceleration']:.4f} ± {avg_metrics['std_gt_mean_acceleration']:.4f} m/s²")
        else:
            print("\n无法计算详细指标 - 没有成功生成轨迹")
            print("请检查模型配置和数据兼容性")
        
        print("\n" + "="*60)
        print(f"结果已保存到: {output_dir}")
        print(f"详细评估结果: {results_file}")
        
        logger.info("Visualization and evaluation completed!")
        
    except Exception as e:
        logger.error(f"执行过程中出现错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
