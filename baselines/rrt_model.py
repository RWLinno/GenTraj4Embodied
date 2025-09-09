"""
Rapidly-exploring Random Tree (RRT) for Trajectory Generation
基于快速探索随机树的轨迹生成方法

RRT通过随机采样和树扩展来探索配置空间，
生成从起点到终点的无碰撞轨迹。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import random
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

from ...base_model import BaseTrajectoryModel


class RRTNode:
    """RRT树节点"""
    
    def __init__(self, position: np.ndarray, parent=None):
        self.position = position
        self.parent = parent
        self.children = []
        
    def add_child(self, child):
        """添加子节点"""
        self.children.append(child)
        child.parent = self


class RRTTrajectoryModel(BaseTrajectoryModel):
    """
    RRT轨迹生成模型
    
    使用快速探索随机树算法生成从起点到终点的轨迹
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # RRT参数
        self.max_iterations = config.get('max_iterations', 1000)
        self.step_size = config.get('step_size', 0.1)
        self.goal_bias = config.get('goal_bias', 0.1)  # 向目标偏置的概率
        self.goal_tolerance = config.get('goal_tolerance', 0.05)
        
        # 工作空间边界
        self.workspace_bounds = config.get('workspace_bounds', [[-2, 2]] * self.output_dim)
        
        # 障碍物检测 (简化版本)
        self.obstacles = config.get('obstacles', [])
        self.obstacle_radius = config.get('obstacle_radius', 0.1)
        
        # 路径平滑参数
        self.smooth_iterations = config.get('smooth_iterations', 10)
        self.smooth_step_size = config.get('smooth_step_size', 0.01)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        RRT前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, output_dim]
            end_pose: 终止位姿 [batch_size, output_dim]
            context: 上下文信息 (未使用)
            
        Returns:
            生成的轨迹 [batch_size, max_seq_length, output_dim]
        """
        batch_size = start_pose.size(0)
        trajectories = []
        
        for i in range(batch_size):
            start = start_pose[i].detach().cpu().numpy()
            end = end_pose[i].detach().cpu().numpy()
            
            # 生成单条轨迹
            trajectory = self._generate_single_trajectory(start, end)
            trajectories.append(torch.from_numpy(trajectory).float())
        
        # 填充到统一长度
        max_len = max(len(traj) for traj in trajectories)
        padded_trajectories = []
        
        for traj in trajectories:
            if len(traj) < max_len:
                # 用最后一个点填充
                padding = traj[-1:].repeat(max_len - len(traj), 1)
                traj = torch.cat([traj, padding], dim=0)
            padded_trajectories.append(traj)
        
        return torch.stack(padded_trajectories).to(start_pose.device)
    
    def _generate_single_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray) -> np.ndarray:
        """
        生成单条轨迹
        
        Args:
            start_pose: 起始位姿
            end_pose: 终止位姿
            
        Returns:
            生成的轨迹 [seq_length, output_dim]
        """
        # 初始化RRT树
        root = RRTNode(start_pose)
        nodes = [root]
        
        # RRT主循环
        for iteration in range(self.max_iterations):
            # 采样随机点
            if random.random() < self.goal_bias:
                # 向目标偏置
                random_point = end_pose + np.random.normal(0, 0.01, self.output_dim)
            else:
                # 随机采样
                random_point = self._sample_random_point()
            
            # 找到最近的节点
            nearest_node = self._find_nearest_node(nodes, random_point)
            
            # 扩展树
            new_point = self._steer(nearest_node.position, random_point)
            
            # 碰撞检测
            if not self._is_collision_free(nearest_node.position, new_point):
                continue
            
            # 添加新节点
            new_node = RRTNode(new_point, nearest_node)
            nearest_node.add_child(new_node)
            nodes.append(new_node)
            
            # 检查是否到达目标
            if euclidean(new_point, end_pose) < self.goal_tolerance:
                # 构建路径
                path = self._extract_path(new_node)
                
                # 路径平滑
                smoothed_path = self._smooth_path(path)
                
                # 重新采样到指定长度
                return self._resample_path(smoothed_path)
        
        # 如果未找到路径，返回直线插值
        return self._linear_interpolation(start_pose, end_pose)
    
    def _sample_random_point(self) -> np.ndarray:
        """采样随机点"""
        point = np.zeros(self.output_dim)
        for i, (min_val, max_val) in enumerate(self.workspace_bounds):
            point[i] = random.uniform(min_val, max_val)
        return point
    
    def _find_nearest_node(self, nodes: List[RRTNode], point: np.ndarray) -> RRTNode:
        """找到最近的节点"""
        min_distance = float('inf')
        nearest_node = nodes[0]
        
        for node in nodes:
            distance = euclidean(node.position, point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """从一个点向另一个点扩展固定步长"""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_point
        else:
            unit_direction = direction / distance
            return from_point + unit_direction * self.step_size
    
    def _is_collision_free(self, from_point: np.ndarray, to_point: np.ndarray) -> bool:
        """检查路径是否无碰撞"""
        # 简化的碰撞检测：检查线段与球形障碍物的碰撞
        num_checks = int(euclidean(from_point, to_point) / 0.01) + 1
        
        for i in range(num_checks + 1):
            alpha = i / num_checks if num_checks > 0 else 0
            check_point = (1 - alpha) * from_point + alpha * to_point
            
            # 检查与所有障碍物的碰撞
            for obstacle in self.obstacles:
                if euclidean(check_point, obstacle) < self.obstacle_radius:
                    return False
        
        return True
    
    def _extract_path(self, goal_node: RRTNode) -> np.ndarray:
        """从目标节点提取路径"""
        path = []
        current_node = goal_node
        
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        
        # 反转路径 (从起点到终点)
        path.reverse()
        return np.array(path)
    
    def _smooth_path(self, path: np.ndarray) -> np.ndarray:
        """路径平滑"""
        if len(path) <= 2:
            return path
        
        smoothed_path = path.copy()
        
        for _ in range(self.smooth_iterations):
            for i in range(1, len(smoothed_path) - 1):
                # 尝试连接前一个点和后一个点
                if self._is_collision_free(smoothed_path[i-1], smoothed_path[i+1]):
                    # 移动当前点向连线靠近
                    direction = smoothed_path[i+1] - smoothed_path[i-1]
                    midpoint = smoothed_path[i-1] + 0.5 * direction
                    
                    # 朝中点移动一小步
                    move_direction = midpoint - smoothed_path[i]
                    smoothed_path[i] += self.smooth_step_size * move_direction
        
        return smoothed_path
    
    def _resample_path(self, path: np.ndarray) -> np.ndarray:
        """重新采样路径到指定长度"""
        if len(path) <= 1:
            return np.tile(path[0], (self.max_seq_length, 1))
        
        # 计算累积距离
        distances = [0]
        for i in range(1, len(path)):
            dist = euclidean(path[i], path[i-1])
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        
        if total_distance == 0:
            return np.tile(path[0], (self.max_seq_length, 1))
        
        # 均匀重新采样
        new_distances = np.linspace(0, total_distance, self.max_seq_length)
        resampled_path = np.zeros((self.max_seq_length, self.output_dim))
        
        for dim in range(self.output_dim):
            f = interp1d(distances, path[:, dim], kind='linear', 
                        fill_value='extrapolate')
            resampled_path[:, dim] = f(new_distances)
        
        return resampled_path
    
    def _linear_interpolation(self, start_pose: np.ndarray, end_pose: np.ndarray) -> np.ndarray:
        """线性插值作为备选方案"""
        trajectory = np.zeros((self.max_seq_length, self.output_dim))
        for i in range(self.max_seq_length):
            alpha = i / (self.max_seq_length - 1)
            trajectory[i] = (1 - alpha) * start_pose + alpha * end_pose
        return trajectory
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [output_dim]
            end_pose: 终止位姿 [output_dim]
            num_points: 轨迹点数量
            
        Returns:
            生成的轨迹 [num_points, output_dim]
        """
        # 临时调整序列长度
        original_max_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        try:
            trajectory = self._generate_single_trajectory(start_pose, end_pose)
            return trajectory
        
        finally:
            # 恢复原始序列长度
            self.max_seq_length = original_max_seq_length
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        计算RRT损失函数
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            
        Returns:
            损失值
        """
        # 路径长度损失 (鼓励短路径)
        pred_diff = torch.diff(predictions, dim=1)
        path_length = torch.sum(torch.norm(pred_diff, dim=2), dim=1).mean()
        
        # 终点误差
        end_loss = torch.mean((predictions[:, -1] - targets[:, -1]) ** 2)
        
        # 平滑度损失
        pred_acc = torch.diff(predictions, n=2, dim=1)
        smoothness_loss = torch.mean(pred_acc ** 2)
        
        # 总损失
        total_loss = 0.1 * path_length + 2.0 * end_loss + 0.1 * smoothness_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'model_type': 'Rapidly-exploring Random Tree (RRT)',
            'max_iterations': self.max_iterations,
            'step_size': self.step_size,
            'goal_bias': self.goal_bias,
            'goal_tolerance': self.goal_tolerance,
            'workspace_bounds': self.workspace_bounds,
            'num_obstacles': len(self.obstacles)
        })
        return info