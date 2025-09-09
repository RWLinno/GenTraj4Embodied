"""
Probabilistic Roadmap (PRM) Trajectory Generation Model
概率路线图轨迹生成模型

基于概率路线图的采样规划方法，通过构建随机采样的路线图来生成轨迹。
PRM是一种多查询路径规划算法，适用于高维配置空间的轨迹生成。

Reference:
- Kavraki, L. E., et al. "Probabilistic roadmaps for path planning in high-dimensional configuration spaces." 
  IEEE transactions on Robotics and Automation 12.4 (1996): 566-580.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean

from ...base_model import BaseTrajectoryModel


class PRMTrajectoryModel(BaseTrajectoryModel):
    """
    Probabilistic Roadmap (PRM) 轨迹生成模型
    
    PRM通过以下步骤生成轨迹：
    1. 在配置空间中随机采样节点
    2. 构建路线图连接邻近节点
    3. 使用图搜索算法找到从起点到终点的路径
    4. 对路径进行平滑处理生成最终轨迹
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # PRM参数
        self.num_samples = config.get('num_samples', 1000)  # 采样节点数量
        self.k_neighbors = config.get('k_neighbors', 10)    # 邻居节点数量
        self.max_edge_length = config.get('max_edge_length', 0.5)  # 最大边长
        self.collision_check = config.get('collision_check', True)  # 是否进行碰撞检测
        self.smoothing_iterations = config.get('smoothing_iterations', 10)  # 平滑迭代次数
        
        # 工作空间边界
        self.workspace_bounds = config.get('workspace_bounds', {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0, 
            'z_min': -1.0, 'z_max': 1.0
        })
        
        # 障碍物定义（简化为球形障碍物）
        self.obstacles = config.get('obstacles', [])
        
        # 路线图
        self.roadmap = None
        self.samples = None
        self.is_built = False
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_roadmap(self):
        """构建概率路线图"""
        print(f"构建PRM路线图，采样 {self.num_samples} 个节点...")
        
        # 1. 随机采样节点
        self.samples = self._sample_configuration_space()
        
        # 2. 过滤有效节点（无碰撞）
        if self.collision_check:
            valid_samples = []
            for sample in self.samples:
                if not self._is_collision(sample):
                    valid_samples.append(sample)
            self.samples = np.array(valid_samples)
            print(f"碰撞检测后保留 {len(self.samples)} 个有效节点")
        
        # 3. 构建路线图
        self.roadmap = nx.Graph()
        
        # 添加节点
        for i, sample in enumerate(self.samples):
            self.roadmap.add_node(i, pos=sample)
        
        # 4. 连接邻近节点
        self._connect_neighbors()
        
        self.is_built = True
        print(f"路线图构建完成，包含 {len(self.roadmap.nodes)} 个节点，{len(self.roadmap.edges)} 条边")
    
    def _sample_configuration_space(self) -> np.ndarray:
        """在配置空间中随机采样"""
        samples = []
        
        for _ in range(self.num_samples):
            sample = np.array([
                np.random.uniform(self.workspace_bounds['x_min'], self.workspace_bounds['x_max']),
                np.random.uniform(self.workspace_bounds['y_min'], self.workspace_bounds['y_max']),
                np.random.uniform(self.workspace_bounds['z_min'], self.workspace_bounds['z_max'])
            ])
            samples.append(sample)
        
        return np.array(samples)
    
    def _is_collision(self, point: np.ndarray) -> bool:
        """检查点是否与障碍物碰撞"""
        for obstacle in self.obstacles:
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            if np.linalg.norm(point - center) < radius:
                return True
        return False
    
    def _connect_neighbors(self):
        """连接邻近节点"""
        if len(self.samples) == 0:
            return
        
        # 使用KNN找到邻近节点
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(self.samples)), 
                              algorithm='ball_tree').fit(self.samples)
        
        for i, sample in enumerate(self.samples):
            distances, indices = nbrs.kneighbors([sample])
            
            for j, neighbor_idx in enumerate(indices[0]):
                if neighbor_idx != i:  # 不连接自己
                    distance = distances[0][j]
                    
                    # 检查距离和碰撞
                    if distance < self.max_edge_length:
                        if not self.collision_check or self._is_path_valid(sample, self.samples[neighbor_idx]):
                            self.roadmap.add_edge(i, neighbor_idx, weight=distance)
    
    def _is_path_valid(self, start: np.ndarray, end: np.ndarray, num_checks: int = 10) -> bool:
        """检查两点之间的路径是否有效（无碰撞）"""
        for t in np.linspace(0, 1, num_checks):
            point = start + t * (end - start)
            if self._is_collision(point):
                return False
        return True
    
    def _find_path(self, start_pos: np.ndarray, end_pos: np.ndarray) -> Optional[List[int]]:
        """在路线图中找到从起点到终点的路径"""
        if not self.is_built:
            self.build_roadmap()
        
        # 找到最近的起点和终点节点
        start_idx = self._find_nearest_node(start_pos)
        end_idx = self._find_nearest_node(end_pos)
        
        if start_idx is None or end_idx is None:
            return None
        
        # 使用Dijkstra算法找路径
        try:
            path = nx.shortest_path(self.roadmap, start_idx, end_idx, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None
    
    def _find_nearest_node(self, pos: np.ndarray) -> Optional[int]:
        """找到位置最近的节点"""
        if len(self.samples) == 0:
            return None
        
        distances = [np.linalg.norm(pos - sample) for sample in self.samples]
        min_idx = np.argmin(distances)
        
        # 检查最近节点是否在合理范围内
        if distances[min_idx] < self.max_edge_length:
            return min_idx
        return None
    
    def _smooth_path(self, path_points: np.ndarray) -> np.ndarray:
        """对路径进行平滑处理"""
        if len(path_points) < 3:
            return path_points
        
        smoothed = path_points.copy()
        
        for _ in range(self.smoothing_iterations):
            for i in range(1, len(smoothed) - 1):
                # 简单的平滑：取相邻点的平均
                smoothed[i] = 0.25 * smoothed[i-1] + 0.5 * smoothed[i] + 0.25 * smoothed[i+1]
        
        return smoothed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # PRM是基于采样的方法，不需要神经网络前向传播
        # 这里返回输入作为占位符
        return x
    
    def generate_trajectory(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                          num_points: int = 50) -> torch.Tensor:
        """
        使用PRM生成轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, 3] 或 [3]
            end_pose: 结束位姿 [batch_size, 3] 或 [3]
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 生成的轨迹 [batch_size, num_points, 3]
        """
        if start_pose.dim() == 1:
            start_pose = start_pose.unsqueeze(0)
        if end_pose.dim() == 1:
            end_pose = end_pose.unsqueeze(0)
        
        batch_size = start_pose.shape[0]
        trajectories = []
        
        for b in range(batch_size):
            start_np = start_pose[b].cpu().numpy()
            end_np = end_pose[b].cpu().numpy()
            
            # 在路线图中找路径
            path_indices = self._find_path(start_np, end_np)
            
            if path_indices is None:
                # 如果找不到路径，使用直线插值
                trajectory = self._linear_interpolation(start_np, end_np, num_points)
            else:
                # 构建路径点
                path_points = np.array([self.samples[idx] for idx in path_indices])
                
                # 添加精确的起点和终点
                path_points[0] = start_np
                path_points[-1] = end_np
                
                # 平滑路径
                path_points = self._smooth_path(path_points)
                
                # 重新采样到指定点数
                trajectory = self._resample_path(path_points, num_points)
            
            trajectories.append(trajectory)
        
        # 转换为tensor
        trajectories = np.stack(trajectories)
        return torch.from_numpy(trajectories).float().to(self.device)
    
    def _linear_interpolation(self, start: np.ndarray, end: np.ndarray, num_points: int) -> np.ndarray:
        """线性插值作为备选方案"""
        t = np.linspace(0, 1, num_points)
        trajectory = np.outer(1 - t, start) + np.outer(t, end)
        return trajectory
    
    def _resample_path(self, path_points: np.ndarray, num_points: int) -> np.ndarray:
        """重新采样路径到指定点数"""
        if len(path_points) == 1:
            return np.tile(path_points[0], (num_points, 1))
        
        # 计算累积距离
        distances = [0]
        for i in range(1, len(path_points)):
            dist = np.linalg.norm(path_points[i] - path_points[i-1])
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        if total_distance == 0:
            return np.tile(path_points[0], (num_points, 1))
        
        # 均匀采样
        target_distances = np.linspace(0, total_distance, num_points)
        resampled_points = []
        
        for target_dist in target_distances:
            # 找到对应的路径段
            for i in range(len(distances) - 1):
                if distances[i] <= target_dist <= distances[i + 1]:
                    if distances[i + 1] == distances[i]:
                        point = path_points[i]
                    else:
                        t = (target_dist - distances[i]) / (distances[i + 1] - distances[i])
                        point = path_points[i] + t * (path_points[i + 1] - path_points[i])
                    resampled_points.append(point)
                    break
        
        return np.array(resampled_points)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        # PRM是无监督方法，使用路径长度和平滑度作为损失
        path_length_loss = self._compute_path_length_loss(predictions)
        smoothness_loss = self._compute_smoothness_loss(predictions)
        
        total_loss = path_length_loss + 0.1 * smoothness_loss
        return total_loss
    
    def _compute_path_length_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """计算路径长度损失（鼓励较短路径）"""
        diffs = trajectories[:, 1:] - trajectories[:, :-1]
        lengths = torch.norm(diffs, dim=-1)
        total_length = torch.sum(lengths, dim=1)
        return torch.mean(total_length)
    
    def _compute_smoothness_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """计算平滑度损失"""
        if trajectories.shape[1] < 3:
            return torch.tensor(0.0, device=trajectories.device)
        
        # 计算二阶导数
        second_derivatives = trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2]
        smoothness = torch.norm(second_derivatives, dim=-1)
        return torch.mean(smoothness)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        # PRM不需要传统的训练，这里重新构建路线图
        if not self.is_built or np.random.random() < 0.1:  # 10%概率重建
            self.build_roadmap()
        
        # 生成一些示例轨迹来计算损失
        start_poses = batch.get('start_pose', torch.randn(4, 3) * 0.5)
        end_poses = batch.get('end_pose', torch.randn(4, 3) * 0.5)
        
        trajectories = self.generate_trajectory(start_poses, end_poses)
        loss = self.compute_loss(trajectories, trajectories)  # 自监督
        
        return {
            'loss': loss.item(),
            'path_length': self._compute_path_length_loss(trajectories).item(),
            'smoothness': self._compute_smoothness_loss(trajectories).item()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'PRM',
            'num_samples': self.num_samples,
            'k_neighbors': self.k_neighbors,
            'max_edge_length': self.max_edge_length,
            'is_built': self.is_built,
            'num_nodes': len(self.roadmap.nodes) if self.roadmap else 0,
            'num_edges': len(self.roadmap.edges) if self.roadmap else 0
        }