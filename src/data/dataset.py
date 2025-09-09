"""
Dataset classes for trajectory generation
轨迹生成数据集类
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from .data_generator import TrajectoryDataGenerator
from .transforms import TrajectoryTransforms


class TrajectoryDataset(Dataset):
    """轨迹数据集类"""
    
    def __init__(self, 
                 train_path: Optional[Path] = None,
                 val_path: Optional[Path] = None,
                 test_path: Optional[Path] = None,
                 config: Dict[str, Any] = None,
                 mode: str = 'train',
                 transforms: Optional[TrajectoryTransforms] = None):
        """
        初始化轨迹数据集
        
        Args:
            train_path: 训练数据路径
            val_path: 验证数据路径
            test_path: 测试数据路径
            config: 数据配置
            mode: 数据集模式 ('train', 'val', 'test')
            transforms: 数据变换
        """
        self.config = config or {}
        self.mode = mode
        self.transforms = transforms
        self.logger = logging.getLogger(__name__)
        
        # 根据模式选择数据路径
        if mode == 'train' and train_path and train_path.exists():
            self.data_path = train_path
        elif mode == 'val' and val_path and val_path.exists():
            self.data_path = val_path
        elif mode == 'test' and test_path and test_path.exists():
            self.data_path = test_path
        else:
            raise ValueError(f"无法找到{mode}模式的数据文件")
        
        # 加载数据
        self.trajectories = self._load_data()
        
        # 构建索引
        self._build_indices()
        
        self.logger.info(f"加载{mode}数据集: {len(self.trajectories)}条轨迹")
    
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        generator = TrajectoryDataGenerator(self.config)
        return generator.load_data(self.data_path)
    
    def _build_indices(self):
        """构建数据索引"""
        self.task_to_indices = {}
        self.modality_to_indices = {}
        
        for i, traj in enumerate(self.trajectories):
            task_name = traj['task_name']
            modality = traj['modality']
            
            if task_name not in self.task_to_indices:
                self.task_to_indices[task_name] = []
            self.task_to_indices[task_name].append(i)
            
            if modality not in self.modality_to_indices:
                self.modality_to_indices[modality] = []
            self.modality_to_indices[modality].append(i)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本数据字典
        """
        traj_data = self.trajectories[idx]
        
        # 提取基本数据
        trajectory = traj_data['trajectory'].astype(np.float32)
        start_pose = traj_data['start_pose'].astype(np.float32)
        end_pose = traj_data['end_pose'].astype(np.float32)
        timestamps = traj_data['timestamps'].astype(np.float32)
        
        # 创建样本字典
        sample = {
            'trajectory': torch.from_numpy(trajectory),
            'start_pose': torch.from_numpy(start_pose),
            'end_pose': torch.from_numpy(end_pose),
            'timestamps': torch.from_numpy(timestamps),
            'task_id': torch.tensor(traj_data['task_id'], dtype=torch.long),
            'modality': torch.tensor(traj_data['modality'], dtype=torch.long),
            'length': torch.tensor(traj_data['length'], dtype=torch.long),
            'duration': torch.tensor(traj_data['duration'], dtype=torch.float32)
        }
        
        # 应用数据变换
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def get_by_task(self, task_name: str) -> List[Dict]:
        """
        根据任务名称获取轨迹
        
        Args:
            task_name: 任务名称
            
        Returns:
            轨迹列表
        """
        if task_name not in self.task_to_indices:
            return []
        
        indices = self.task_to_indices[task_name]
        return [self[i] for i in indices]
    
    def get_by_modality(self, modality: int) -> List[Dict]:
        """
        根据模态获取轨迹
        
        Args:
            modality: 模态编号
            
        Returns:
            轨迹列表
        """
        if modality not in self.modality_to_indices:
            return []
        
        indices = self.modality_to_indices[modality]
        return [self[i] for i in indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.trajectories),
            'tasks': {},
            'modalities': {},
            'trajectory_lengths': [],
            'durations': []
        }
        
        for traj in self.trajectories:
            task_name = traj['task_name']
            modality = traj['modality']
            
            if task_name not in stats['tasks']:
                stats['tasks'][task_name] = 0
            stats['tasks'][task_name] += 1
            
            if modality not in stats['modalities']:
                stats['modalities'][modality] = 0
            stats['modalities'][modality] += 1
            
            stats['trajectory_lengths'].append(traj['length'])
            stats['durations'].append(traj['duration'])
        
        return stats
    
    @property
    def train_size(self) -> int:
        """训练集大小"""
        return len(self.trajectories) if self.mode == 'train' else 0
    
    @property
    def val_size(self) -> int:
        """验证集大小"""
        return len(self.trajectories) if self.mode == 'val' else 0
    
    @property
    def test_size(self) -> int:
        """测试集大小"""
        return len(self.trajectories) if self.mode == 'test' else 0


class ConditionalTrajectoryDataset(TrajectoryDataset):
    """条件轨迹数据集，支持条件生成"""
    
    def __init__(self, *args, condition_type: str = 'start_end', **kwargs):
        """
        初始化条件轨迹数据集
        
        Args:
            condition_type: 条件类型 ('start_end', 'task', 'waypoints')
        """
        super().__init__(*args, **kwargs)
        self.condition_type = condition_type
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取条件样本"""
        sample = super().__getitem__(idx)
        
        # 根据条件类型添加条件信息
        if self.condition_type == 'start_end':
            sample['condition'] = torch.cat([sample['start_pose'], sample['end_pose']])
        elif self.condition_type == 'task':
            # 使用任务ID作为条件
            sample['condition'] = sample['task_id']
        elif self.condition_type == 'waypoints':
            # 使用轨迹中的关键点作为条件
            trajectory = sample['trajectory']
            num_waypoints = min(5, len(trajectory))
            waypoint_indices = np.linspace(0, len(trajectory)-1, num_waypoints, dtype=int)
            waypoints = trajectory[waypoint_indices].flatten()
            sample['condition'] = waypoints
        
        return sample


class SequenceTrajectoryDataset(TrajectoryDataset):
    """序列轨迹数据集，用于序列到序列的模型"""
    
    def __init__(self, *args, sequence_length: int = 50, **kwargs):
        """
        初始化序列轨迹数据集
        
        Args:
            sequence_length: 序列长度
        """
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取序列样本"""
        sample = super().__getitem__(idx)
        
        trajectory = sample['trajectory']
        
        # 如果轨迹长度不等于目标长度，进行插值或截断
        if len(trajectory) != self.sequence_length:
            trajectory = self._resize_trajectory(trajectory, self.sequence_length)
        
        # 创建输入和目标序列
        sample['input_sequence'] = trajectory[:-1]  # 前n-1个点
        sample['target_sequence'] = trajectory[1:]  # 后n-1个点
        sample['trajectory'] = trajectory
        
        return sample
    
    def _resize_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """调整轨迹长度"""
        current_length = len(trajectory)
        
        if current_length == target_length:
            return trajectory
        
        # 使用线性插值调整长度
        indices = torch.linspace(0, current_length - 1, target_length)
        
        # 对位置进行插值
        positions = trajectory[:, :3]
        new_positions = torch.zeros(target_length, 3)
        for i in range(3):
            new_positions[:, i] = torch.interp(indices, torch.arange(current_length).float(), positions[:, i])
        
        # 对四元数进行球面线性插值（简化版本）
        quaternions = trajectory[:, 3:]
        new_quaternions = torch.zeros(target_length, 4)
        for i in range(4):
            new_quaternions[:, i] = torch.interp(indices, torch.arange(current_length).float(), quaternions[:, i])
        
        # 归一化四元数
        new_quaternions = new_quaternions / torch.norm(new_quaternions, dim=1, keepdim=True)
        
        return torch.cat([new_positions, new_quaternions], dim=1)


class MultiModalTrajectoryDataset(TrajectoryDataset):
    """多模态轨迹数据集，支持同一任务的多种解决方案"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_multimodal_indices()
    
    def _build_multimodal_indices(self):
        """构建多模态索引"""
        self.task_modality_groups = {}
        
        for i, traj in enumerate(self.trajectories):
            task_id = traj['task_id']
            start_pose = tuple(traj['start_pose'][:3])  # 使用位置作为分组键
            end_pose = tuple(traj['end_pose'][:3])
            
            key = (task_id, start_pose, end_pose)
            
            if key not in self.task_modality_groups:
                self.task_modality_groups[key] = []
            self.task_modality_groups[key].append(i)
    
    def get_multimodal_samples(self, task_id: int, start_pose: np.ndarray, 
                              end_pose: np.ndarray) -> List[Dict]:
        """
        获取给定起终点的所有模态样本
        
        Args:
            task_id: 任务ID
            start_pose: 起始位姿
            end_pose: 终止位姿
            
        Returns:
            多模态样本列表
        """
        key = (task_id, tuple(start_pose[:3]), tuple(end_pose[:3]))
        
        if key not in self.task_modality_groups:
            return []
        
        indices = self.task_modality_groups[key]
        return [self[i] for i in indices]
    
    def sample_multimodal_batch(self, batch_size: int) -> List[List[Dict]]:
        """
        采样多模态批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            多模态批次列表
        """
        batch = []
        keys = list(self.task_modality_groups.keys())
        
        for _ in range(batch_size):
            # 随机选择一个任务组
            key = np.random.choice(keys)
            indices = self.task_modality_groups[key]
            
            # 获取该组的所有样本
            group_samples = [self[i] for i in indices]
            batch.append(group_samples)
        
        return batch


def create_data_loaders(dataset_config: Dict[str, Any], 
                       train_path: Path, 
                       val_path: Path, 
                       test_path: Path,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       dataset_type: str = 'standard') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        dataset_config: 数据集配置
        train_path: 训练数据路径
        val_path: 验证数据路径
        test_path: 测试数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        dataset_type: 数据集类型
        
    Returns:
        (训练加载器, 验证加载器, 测试加载器)
    """
    # 创建数据变换
    transforms = TrajectoryTransforms(dataset_config.get('augmentation', {}))
    
    # 选择数据集类
    if dataset_type == 'conditional':
        dataset_class = ConditionalTrajectoryDataset
    elif dataset_type == 'sequence':
        dataset_class = SequenceTrajectoryDataset
    elif dataset_type == 'multimodal':
        dataset_class = MultiModalTrajectoryDataset
    else:
        dataset_class = TrajectoryDataset
    
    # 创建数据集
    train_dataset = dataset_class(
        train_path=train_path,
        config=dataset_config,
        mode='train',
        transforms=transforms
    )
    
    val_dataset = dataset_class(
        val_path=val_path,
        config=dataset_config,
        mode='val',
        transforms=None  # 验证集不使用数据增强
    )
    
    test_dataset = dataset_class(
        test_path=test_path,
        config=dataset_config,
        mode='test',
        transforms=None  # 测试集不使用数据增强
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_trajectory_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_trajectory_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_trajectory_batch
    )
    
    return train_loader, val_loader, test_loader


def collate_trajectory_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    轨迹批次整理函数
    
    Args:
        batch: 批次数据列表
        
    Returns:
        整理后的批次字典
    """
    # 获取所有键
    keys = batch[0].keys()
    
    # 整理批次数据
    collated = {}
    
    for key in keys:
        if key in ['trajectory', 'timestamps']:
            # 轨迹和时间戳可能有不同长度，需要填充
            sequences = [item[key] for item in batch]
            max_length = max(len(seq) for seq in sequences)
            
            # 创建填充后的张量
            batch_size = len(sequences)
            if key == 'trajectory':
                feature_dim = sequences[0].shape[-1]
                padded_sequences = torch.zeros(batch_size, max_length, feature_dim)
            else:  # timestamps
                padded_sequences = torch.zeros(batch_size, max_length)
            
            # 填充序列
            for i, seq in enumerate(sequences):
                padded_sequences[i, :len(seq)] = seq
            
            collated[key] = padded_sequences
            
            # 添加长度信息（只需要一次）
            if key == 'trajectory':
                lengths = torch.tensor([len(seq) for seq in sequences])
                collated['trajectory_lengths'] = lengths
            
        else:
            # 其他字段直接堆叠
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


class TrajectoryDataModule:
    """轨迹数据模块，封装数据加载逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据模块
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data']
        self.training_config = config['training']
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.logger = logging.getLogger(__name__)
    
    def setup(self, data_dir: Path):
        """
        设置数据加载器
        
        Args:
            data_dir: 数据目录
        """
        train_path = data_dir / "train.h5"
        val_path = data_dir / "val.h5"
        test_path = data_dir / "test.h5"
        
        # 检查数据文件是否存在
        if not all(path.exists() for path in [train_path, val_path, test_path]):
            self.logger.warning("数据文件不存在，需要先生成数据")
            return
        
        # 创建数据加载器
        batch_size = self.training_config['dataloader'].get('batch_size', 32)
        num_workers = self.training_config['dataloader'].get('num_workers', 4)
        
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            dataset_config=self.data_config,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        self.logger.info("数据加载器设置完成")
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        return self.test_loader