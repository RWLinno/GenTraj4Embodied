"""
Evaluation metrics for trajectory generation
轨迹生成评估指标
"""

import numpy as np
from typing import List, Dict, Any
from ..utils.math_utils import (
    compute_trajectory_smoothness, 
    compute_trajectory_length,
    compute_pose_distance,
    compute_trajectory_diversity
)


class TrajectoryMetrics:
    """轨迹评估指标类"""
    
    def __init__(self, metric_configs: List[Dict[str, Any]]):
        """
        初始化评估指标
        
        Args:
            metric_configs: 指标配置列表
        """
        self.metric_configs = metric_configs
        self.workspace_bounds = {
            'x': [-1.0, 1.0],
            'y': [-1.0, 1.0], 
            'z': [0.2, 1.2]
        }
    
    def compute_smoothness(self, trajectories: np.ndarray) -> np.ndarray:
        """
        计算轨迹平滑度
        
        Args:
            trajectories: 轨迹数组 [N, seq_len, 7]
            
        Returns:
            平滑度分数数组 [N]
        """
        smoothness_scores = []
        
        for traj in trajectories:
            # 计算平滑度（基于加速度变化）
            smoothness = compute_trajectory_smoothness(traj)
            # 转换为0-1分数（越小越好，所以取倒数）
            score = 1.0 / (1.0 + smoothness)
            smoothness_scores.append(score)
        
        return np.array(smoothness_scores)
    
    def compute_task_completion(self, trajectories: np.ndarray, 
                              conditions: np.ndarray) -> np.ndarray:
        """
        计算任务完成度
        
        Args:
            trajectories: 轨迹数组 [N, seq_len, 7]
            conditions: 条件数组 [N, 14] (start_pose + end_pose)
            
        Returns:
            任务完成度分数数组 [N]
        """
        completion_scores = []
        
        for traj, condition in zip(trajectories, conditions):
            start_pose = condition[:7]
            end_pose = condition[7:]
            
            # 检查起点匹配度
            start_distance = compute_pose_distance(traj[0], start_pose)
            start_score = np.exp(-start_distance)  # 距离越小分数越高
            
            # 检查终点匹配度
            end_distance = compute_pose_distance(traj[-1], end_pose)
            end_score = np.exp(-end_distance)
            
            # 综合分数
            completion_score = 0.3 * start_score + 0.7 * end_score
            completion_scores.append(completion_score)
        
        return np.array(completion_scores)
    
    def compute_diversity(self, trajectories: np.ndarray) -> float:
        """
        计算轨迹集合的多样性
        
        Args:
            trajectories: 轨迹数组 [N, seq_len, 7]
            
        Returns:
            多样性分数
        """
        if len(trajectories) < 2:
            return 0.0
        
        # 使用工具函数计算多样性
        trajectory_list = [traj for traj in trajectories]
        diversity = compute_trajectory_diversity(trajectory_list)
        
        # 归一化到0-1范围
        normalized_diversity = min(1.0, diversity / 2.0)
        
        return normalized_diversity
    
    def compute_feasibility(self, trajectories: np.ndarray) -> np.ndarray:
        """
        计算轨迹的物理可行性
        
        Args:
            trajectories: 轨迹数组 [N, seq_len, 7]
            
        Returns:
            可行性分数数组 [N]
        """
        feasibility_scores = []
        
        for traj in trajectories:
            score = 1.0
            
            # 检查工作空间约束
            positions = traj[:, :3]
            for i, axis in enumerate(['x', 'y', 'z']):
                min_bound, max_bound = self.workspace_bounds[axis]
                out_of_bounds = np.sum((positions[:, i] < min_bound) | (positions[:, i] > max_bound))
                score *= (1.0 - out_of_bounds / len(positions))
            
            # 检查速度约束
            velocities = np.diff(positions, axis=0)
            max_velocity = np.max(np.linalg.norm(velocities, axis=1))
            if max_velocity > 2.0:  # 最大速度限制
                score *= 0.5
            
            # 检查加速度约束
            accelerations = np.diff(velocities, axis=0)
            max_acceleration = np.max(np.linalg.norm(accelerations, axis=1))
            if max_acceleration > 5.0:  # 最大加速度限制
                score *= 0.5
            
            # 检查四元数归一化
            quaternions = traj[:, 3:]
            norms = np.linalg.norm(quaternions, axis=1)
            norm_error = np.mean(np.abs(norms - 1.0))
            score *= np.exp(-norm_error * 10)  # 惩罚未归一化的四元数
            
            feasibility_scores.append(score)
        
        return np.array(feasibility_scores)
    
    def compute_accuracy(self, generated_trajectories: np.ndarray,
                        ground_truth_trajectories: np.ndarray) -> np.ndarray:
        """
        计算轨迹准确度（与真实轨迹的相似度）
        
        Args:
            generated_trajectories: 生成的轨迹 [N, seq_len, 7]
            ground_truth_trajectories: 真实轨迹 [N, seq_len, 7]
            
        Returns:
            准确度分数数组 [N]
        """
        accuracy_scores = []
        
        for gen_traj, gt_traj in zip(generated_trajectories, ground_truth_trajectories):
            # 调整长度匹配
            min_len = min(len(gen_traj), len(gt_traj))
            gen_traj = gen_traj[:min_len]
            gt_traj = gt_traj[:min_len]
            
            # 计算逐点距离
            distances = []
            for i in range(min_len):
                dist = compute_pose_distance(gen_traj[i], gt_traj[i])
                distances.append(dist)
            
            # 计算平均距离并转换为分数
            avg_distance = np.mean(distances)
            accuracy = np.exp(-avg_distance)  # 距离越小准确度越高
            accuracy_scores.append(accuracy)
        
        return np.array(accuracy_scores)
    
    def compute_efficiency(self, trajectories: np.ndarray) -> np.ndarray:
        """
        计算轨迹效率（路径长度相对于直线距离）
        
        Args:
            trajectories: 轨迹数组 [N, seq_len, 7]
            
        Returns:
            效率分数数组 [N]
        """
        efficiency_scores = []
        
        for traj in trajectories:
            # 计算轨迹总长度
            trajectory_length = compute_trajectory_length(traj)
            
            # 计算直线距离
            start_pos = traj[0, :3]
            end_pos = traj[-1, :3]
            straight_distance = np.linalg.norm(end_pos - start_pos)
            
            # 计算效率（直线距离/轨迹长度）
            if trajectory_length > 0:
                efficiency = straight_distance / trajectory_length
            else:
                efficiency = 0.0
            
            efficiency_scores.append(min(1.0, efficiency))  # 限制在0-1范围
        
        return np.array(efficiency_scores)
    
    def compute_consistency(self, trajectories: np.ndarray) -> float:
        """
        计算轨迹集合的一致性
        
        Args:
            trajectories: 轨迹数组 [N, seq_len, 7]
            
        Returns:
            一致性分数
        """
        if len(trajectories) < 2:
            return 1.0
        
        # 计算所有轨迹的平均轨迹
        mean_trajectory = np.mean(trajectories, axis=0)
        
        # 计算每个轨迹与平均轨迹的距离
        distances = []
        for traj in trajectories:
            traj_distances = []
            min_len = min(len(traj), len(mean_trajectory))
            for i in range(min_len):
                dist = compute_pose_distance(traj[i], mean_trajectory[i])
                traj_distances.append(dist)
            distances.append(np.mean(traj_distances))
        
        # 一致性为距离的倒数
        avg_distance = np.mean(distances)
        consistency = 1.0 / (1.0 + avg_distance)
        
        return consistency
    
    def compute_robustness(self, trajectories: np.ndarray, 
                          noise_level: float = 0.01) -> np.ndarray:
        """
        计算轨迹对噪声的鲁棒性
        
        Args:
            trajectories: 轨迹数组 [N, seq_len, 7]
            noise_level: 噪声水平
            
        Returns:
            鲁棒性分数数组 [N]
        """
        robustness_scores = []
        
        for traj in trajectories:
            # 添加噪声
            noise = np.random.normal(0, noise_level, traj.shape)
            noisy_traj = traj + noise
            
            # 计算原轨迹和噪声轨迹的相似度
            distances = []
            for i in range(len(traj)):
                dist = compute_pose_distance(traj[i], noisy_traj[i])
                distances.append(dist)
            
            # 鲁棒性为相似度的度量
            avg_distance = np.mean(distances)
            robustness = np.exp(-avg_distance / noise_level)
            robustness_scores.append(robustness)
        
        return np.array(robustness_scores)
    
    def compute_all_metrics(self, generated_trajectories: np.ndarray,
                           ground_truth_trajectories: np.ndarray = None,
                           conditions: np.ndarray = None) -> Dict[str, Any]:
        """
        计算所有指标
        
        Args:
            generated_trajectories: 生成的轨迹
            ground_truth_trajectories: 真实轨迹（可选）
            conditions: 条件（可选）
            
        Returns:
            所有指标的结果字典
        """
        results = {}
        
        # 基本指标
        results['smoothness'] = np.mean(self.compute_smoothness(generated_trajectories))
        results['diversity'] = self.compute_diversity(generated_trajectories)
        results['feasibility'] = np.mean(self.compute_feasibility(generated_trajectories))
        results['efficiency'] = np.mean(self.compute_efficiency(generated_trajectories))
        results['consistency'] = self.compute_consistency(generated_trajectories)
        results['robustness'] = np.mean(self.compute_robustness(generated_trajectories))
        
        # 需要条件的指标
        if conditions is not None:
            results['task_completion'] = np.mean(self.compute_task_completion(generated_trajectories, conditions))
        
        # 需要真实轨迹的指标
        if ground_truth_trajectories is not None:
            results['accuracy'] = np.mean(self.compute_accuracy(generated_trajectories, ground_truth_trajectories))
        
        return results
    
    def get_metric_summary(self, results: Dict[str, Any]) -> str:
        """
        获取指标摘要
        
        Args:
            results: 指标结果字典
            
        Returns:
            指标摘要字符串
        """
        summary_lines = []
        summary_lines.append("=== 轨迹生成评估指标摘要 ===")
        
        for metric_name, value in results.items():
            if isinstance(value, (int, float)):
                summary_lines.append(f"{metric_name}: {value:.4f}")
            else:
                summary_lines.append(f"{metric_name}: {value}")
        
        return "\n".join(summary_lines)