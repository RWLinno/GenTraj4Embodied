"""
Data transforms for trajectory generation
轨迹生成数据变换
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable, List
import random
from ..utils.math_utils import add_noise_to_trajectory, normalize_quaternion


class TrajectoryTransforms:
    """轨迹数据变换类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据变换
        
        Args:
            config: 变换配置
        """
        self.config = config
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> List[Callable]:
        """构建变换列表"""
        transforms = []
        
        # 添加噪声变换
        if 'noise_level' in self.config:
            transforms.append(AddNoise(
                position_noise=self.config['noise_level'],
                rotation_noise=self.config.get('rotation_noise', 0.05)
            ))
        
        # 添加时间抖动变换
        if 'temporal_jitter' in self.config:
            transforms.append(TemporalJitter(
                jitter_factor=self.config['temporal_jitter']
            ))
        
        # 添加归一化变换
        transforms.append(NormalizeTrajectory())
        
        return transforms
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        应用变换
        
        Args:
            sample: 输入样本
            
        Returns:
            变换后的样本
        """
        for transform in self.transforms:
            sample = transform(sample)
        
        return sample


class AddNoise:
    """添加噪声变换"""
    
    def __init__(self, position_noise: float = 0.01, rotation_noise: float = 0.05):
        """
        初始化噪声变换
        
        Args:
            position_noise: 位置噪声标准差
            rotation_noise: 旋转噪声标准差
        """
        self.position_noise = position_noise
        self.rotation_noise = rotation_noise
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用噪声变换"""
        trajectory = sample['trajectory'].numpy()
        
        # 添加噪声
        noisy_trajectory = add_noise_to_trajectory(
            trajectory, self.position_noise, self.rotation_noise
        )
        
        sample['trajectory'] = torch.from_numpy(noisy_trajectory.astype(np.float32))
        
        return sample


class TemporalJitter:
    """时间抖动变换"""
    
    def __init__(self, jitter_factor: float = 0.1):
        """
        初始化时间抖动变换
        
        Args:
            jitter_factor: 抖动因子
        """
        self.jitter_factor = jitter_factor
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用时间抖动"""
        trajectory = sample['trajectory']
        original_length = len(trajectory)
        
        # 随机改变长度
        length_factor = random.uniform(1 - self.jitter_factor, 1 + self.jitter_factor)
        new_length = int(original_length * length_factor)
        new_length = max(10, min(new_length, original_length * 2))
        
        if new_length != original_length:
            # 使用线性插值调整长度
            indices = torch.linspace(0, original_length - 1, new_length)
            
            # 插值位置
            positions = trajectory[:, :3]
            new_positions = torch.zeros(new_length, 3)
            for i in range(3):
                new_positions[:, i] = torch.interp(
                    indices, torch.arange(original_length).float(), positions[:, i]
                )
            
            # 插值四元数
            quaternions = trajectory[:, 3:]
            new_quaternions = torch.zeros(new_length, 4)
            for i in range(4):
                new_quaternions[:, i] = torch.interp(
                    indices, torch.arange(original_length).float(), quaternions[:, i]
                )
            
            # 归一化四元数
            new_quaternions = new_quaternions / torch.norm(new_quaternions, dim=1, keepdim=True)
            
            new_trajectory = torch.cat([new_positions, new_quaternions], dim=1)
            sample['trajectory'] = new_trajectory
            sample['length'] = torch.tensor(new_length, dtype=torch.long)
        
        return sample


class NormalizeTrajectory:
    """轨迹归一化变换"""
    
    def __init__(self, position_mean: Optional[np.ndarray] = None,
                 position_std: Optional[np.ndarray] = None):
        """
        初始化归一化变换
        
        Args:
            position_mean: 位置均值
            position_std: 位置标准差
        """
        self.position_mean = position_mean
        self.position_std = position_std
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用归一化"""
        trajectory = sample['trajectory']
        
        # 归一化四元数
        quaternions = trajectory[:, 3:]
        normalized_quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
        
        # 如果提供了统计信息，归一化位置
        positions = trajectory[:, :3]
        if self.position_mean is not None and self.position_std is not None:
            position_mean = torch.from_numpy(self.position_mean).float()
            position_std = torch.from_numpy(self.position_std).float()
            normalized_positions = (positions - position_mean) / position_std
        else:
            normalized_positions = positions
        
        # 重新组合轨迹
        normalized_trajectory = torch.cat([normalized_positions, normalized_quaternions], dim=1)
        sample['trajectory'] = normalized_trajectory
        
        return sample


class RandomRotation:
    """随机旋转变换"""
    
    def __init__(self, max_angle: float = 0.1):
        """
        初始化随机旋转变换
        
        Args:
            max_angle: 最大旋转角度（弧度）
        """
        self.max_angle = max_angle
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用随机旋转"""
        trajectory = sample['trajectory']
        
        # 生成随机旋转
        angle = random.uniform(-self.max_angle, self.max_angle)
        axis = random.choice([0, 1, 2])  # 选择旋转轴
        
        # 创建旋转矩阵
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        if axis == 0:  # 绕X轴旋转
            rotation_matrix = torch.tensor([
                [1, 0, 0],
                [0, cos_angle, -sin_angle],
                [0, sin_angle, cos_angle]
            ], dtype=torch.float32)
        elif axis == 1:  # 绕Y轴旋转
            rotation_matrix = torch.tensor([
                [cos_angle, 0, sin_angle],
                [0, 1, 0],
                [-sin_angle, 0, cos_angle]
            ], dtype=torch.float32)
        else:  # 绕Z轴旋转
            rotation_matrix = torch.tensor([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]
            ], dtype=torch.float32)
        
        # 应用旋转到位置
        positions = trajectory[:, :3]
        rotated_positions = torch.matmul(positions, rotation_matrix.T)
        
        # 更新轨迹
        trajectory[:, :3] = rotated_positions
        sample['trajectory'] = trajectory
        
        return sample


class RandomTranslation:
    """随机平移变换"""
    
    def __init__(self, max_translation: float = 0.05):
        """
        初始化随机平移变换
        
        Args:
            max_translation: 最大平移距离
        """
        self.max_translation = max_translation
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用随机平移"""
        trajectory = sample['trajectory']
        
        # 生成随机平移向量
        translation = torch.uniform(-self.max_translation, self.max_translation, (3,))
        
        # 应用平移
        trajectory[:, :3] += translation
        
        # 更新起点和终点
        sample['start_pose'][:3] += translation
        sample['end_pose'][:3] += translation
        
        sample['trajectory'] = trajectory
        
        return sample


class TrajectorySubsampling:
    """轨迹子采样变换"""
    
    def __init__(self, subsample_ratio: float = 0.8):
        """
        初始化子采样变换
        
        Args:
            subsample_ratio: 子采样比例
        """
        self.subsample_ratio = subsample_ratio
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用子采样"""
        trajectory = sample['trajectory']
        original_length = len(trajectory)
        
        # 计算新长度
        new_length = int(original_length * self.subsample_ratio)
        new_length = max(10, new_length)  # 确保最小长度
        
        if new_length < original_length:
            # 均匀采样索引
            indices = torch.linspace(0, original_length - 1, new_length).long()
            subsampled_trajectory = trajectory[indices]
            
            sample['trajectory'] = subsampled_trajectory
            sample['length'] = torch.tensor(new_length, dtype=torch.long)
        
        return sample


class TrajectorySmoothing:
    """轨迹平滑变换"""
    
    def __init__(self, window_size: int = 5):
        """
        初始化平滑变换
        
        Args:
            window_size: 平滑窗口大小
        """
        self.window_size = window_size
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用平滑"""
        trajectory = sample['trajectory']
        
        if len(trajectory) > self.window_size:
            # 对位置进行移动平均平滑
            positions = trajectory[:, :3]
            smoothed_positions = torch.zeros_like(positions)
            
            half_window = self.window_size // 2
            
            for i in range(len(positions)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(positions), i + half_window + 1)
                smoothed_positions[i] = positions[start_idx:end_idx].mean(dim=0)
            
            # 保持四元数不变（或者可以实现四元数的平滑）
            smoothed_trajectory = torch.cat([smoothed_positions, trajectory[:, 3:]], dim=1)
            sample['trajectory'] = smoothed_trajectory
        
        return sample


class ConditionalTransform:
    """条件变换，根据任务类型应用不同的变换"""
    
    def __init__(self, task_transforms: Dict[str, List[Callable]]):
        """
        初始化条件变换
        
        Args:
            task_transforms: 任务特定的变换字典
        """
        self.task_transforms = task_transforms
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用条件变换"""
        task_id = sample['task_id'].item()
        
        # 根据任务ID选择变换
        if task_id in self.task_transforms:
            transforms = self.task_transforms[task_id]
            for transform in transforms:
                sample = transform(sample)
        
        return sample


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List[Callable]):
        """
        初始化组合变换
        
        Args:
            transforms: 变换列表
        """
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用所有变换"""
        for transform in self.transforms:
            sample = transform(sample)
        
        return sample


class RandomChoice:
    """随机选择一个变换"""
    
    def __init__(self, transforms: List[Callable], probabilities: Optional[List[float]] = None):
        """
        初始化随机选择变换
        
        Args:
            transforms: 变换列表
            probabilities: 选择概率
        """
        self.transforms = transforms
        self.probabilities = probabilities
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """随机选择并应用一个变换"""
        if self.probabilities:
            transform = np.random.choice(self.transforms, p=self.probabilities)
        else:
            transform = random.choice(self.transforms)
        
        return transform(sample)


def create_training_transforms(config: Dict[str, Any]) -> TrajectoryTransforms:
    """
    创建训练时的数据变换
    
    Args:
        config: 配置字典
        
    Returns:
        训练变换对象
    """
    return TrajectoryTransforms(config.get('augmentation', {}))


def create_evaluation_transforms() -> Compose:
    """
    创建评估时的数据变换（通常只包含归一化）
    
    Returns:
        评估变换对象
    """
    return Compose([NormalizeTrajectory()])


def create_custom_transforms(transform_configs: List[Dict[str, Any]]) -> Compose:
    """
    根据配置创建自定义变换
    
    Args:
        transform_configs: 变换配置列表
        
    Returns:
        自定义变换对象
    """
    transforms = []
    
    for config in transform_configs:
        transform_type = config['type']
        params = config.get('params', {})
        
        if transform_type == 'noise':
            transforms.append(AddNoise(**params))
        elif transform_type == 'temporal_jitter':
            transforms.append(TemporalJitter(**params))
        elif transform_type == 'normalize':
            transforms.append(NormalizeTrajectory(**params))
        elif transform_type == 'rotation':
            transforms.append(RandomRotation(**params))
        elif transform_type == 'translation':
            transforms.append(RandomTranslation(**params))
        elif transform_type == 'subsample':
            transforms.append(TrajectorySubsampling(**params))
        elif transform_type == 'smooth':
            transforms.append(TrajectorySmoothing(**params))
        else:
            raise ValueError(f"未知的变换类型: {transform_type}")
    
    return Compose(transforms)