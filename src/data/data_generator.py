"""
Trajectory data generator for creating synthetic training data
合成轨迹数据生成器
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
import logging
from ..utils.math_utils import (
    interpolate_poses, add_noise_to_trajectory, 
    generate_workspace_bounds, check_workspace_bounds,
    normalize_quaternion
)


class TrajectoryDataGenerator:
    """轨迹数据生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据生成器
        
        Args:
            config: 数据生成配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 提取配置参数
        self.num_trajectories = config['generation']['num_trajectories']
        self.num_modalities = config['generation']['num_modalities']
        self.trajectory_length = config['generation']['trajectory_length']
        self.time_horizon = config['generation']['time_horizon']
        self.workspace_bounds = config['generation']['workspace_bounds']
        
        # 任务配置
        self.tasks = config['tasks']
        self.task_weights = [task['weight'] for task in self.tasks]
        
        # 数据分割配置
        self.split_config = config['split']
        
        # 数据增强配置
        self.augmentation = config.get('augmentation', {})
        
        self.logger.info(f"初始化轨迹数据生成器: {self.num_trajectories}条轨迹, {self.num_modalities}种模态")
    
    def generate_all_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        生成所有数据分割
        
        Returns:
            (训练数据, 验证数据, 测试数据)
        """
        self.logger.info("开始生成所有数据分割...")
        
        # 生成所有轨迹
        all_trajectories = self._generate_all_trajectories()
        
        # 分割数据
        train_size = int(len(all_trajectories) * self.split_config['train'])
        val_size = int(len(all_trajectories) * self.split_config['val'])
        
        # 随机打乱
        np.random.shuffle(all_trajectories)
        
        train_data = all_trajectories[:train_size]
        val_data = all_trajectories[train_size:train_size + val_size]
        test_data = all_trajectories[train_size + val_size:]
        
        self.logger.info(f"数据分割完成: 训练集{len(train_data)}, 验证集{len(val_data)}, 测试集{len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _generate_all_trajectories(self) -> List[Dict]:
        """生成所有轨迹数据"""
        all_trajectories = []
        
        for i in range(self.num_trajectories):
            # 随机选择任务类型
            task_idx = np.random.choice(len(self.tasks), p=self.task_weights)
            task = self.tasks[task_idx]
            
            # 为每个任务生成多种模态的轨迹
            for modality in range(self.num_modalities):
                trajectory_data = self._generate_single_trajectory(task, modality, i)
                if trajectory_data is not None:
                    all_trajectories.append(trajectory_data)
        
        return all_trajectories
    
    def _generate_single_trajectory(self, task: Dict, modality: int, trajectory_id: int) -> Optional[Dict]:
        """
        生成单条轨迹
        
        Args:
            task: 任务配置
            modality: 模态编号
            trajectory_id: 轨迹ID
            
        Returns:
            轨迹数据字典
        """
        try:
            # 根据任务类型生成轨迹
            task_name = task['name']
            
            if task_name == 'pick_and_place':
                trajectory = self._generate_pick_and_place_trajectory(modality)
            elif task_name == 'pouring':
                trajectory = self._generate_pouring_trajectory(modality)
            elif task_name == 'assembly':
                trajectory = self._generate_assembly_trajectory(modality)
            else:
                self.logger.warning(f"未知任务类型: {task_name}")
                return None
            
            # 检查轨迹是否在工作空间内
            if not self._validate_trajectory(trajectory):
                return None
            
            # 应用数据增强
            if self.augmentation:
                trajectory = self._apply_augmentation(trajectory)
            
            # 创建轨迹数据字典
            trajectory_data = {
                'trajectory_id': trajectory_id,
                'task_name': task_name,
                'task_id': self.tasks.index(task),
                'modality': modality,
                'start_pose': trajectory[0].copy(),
                'end_pose': trajectory[-1].copy(),
                'trajectory': trajectory,
                'length': len(trajectory),
                'duration': self.time_horizon,
                'timestamps': np.linspace(0, self.time_horizon, len(trajectory))
            }
            
            return trajectory_data
            
        except Exception as e:
            self.logger.error(f"生成轨迹时出错: {e}")
            return None
    
    def _generate_pick_and_place_trajectory(self, modality: int) -> np.ndarray:
        """
        生成抓取和放置轨迹
        
        Args:
            modality: 模态编号
            
        Returns:
            轨迹数组 [N, 7] (x, y, z, qx, qy, qz, qw)
        """
        # 随机生成起点和终点
        start_pos = self._sample_random_position()
        end_pos = self._sample_random_position()
        
        # 确保起点和终点有足够距离
        while np.linalg.norm(end_pos - start_pos) < 0.2:
            end_pos = self._sample_random_position()
        
        # 生成中间抓取点（高度较高）
        pick_height = max(start_pos[2], end_pos[2]) + 0.3
        approach_pos = start_pos.copy()
        approach_pos[2] = pick_height
        retreat_pos = end_pos.copy()
        retreat_pos[2] = pick_height
        
        # 根据模态生成不同的轨迹路径
        if modality == 0:
            # 直接路径
            waypoints = [start_pos, approach_pos, retreat_pos, end_pos]
        elif modality == 1:
            # 弧形路径
            mid_pos = (start_pos + end_pos) / 2
            mid_pos[2] += 0.4  # 增加高度
            waypoints = [start_pos, approach_pos, mid_pos, retreat_pos, end_pos]
        else:
            # 绕行路径
            detour_pos = start_pos.copy()
            detour_pos[0] += 0.3 * (1 if np.random.random() > 0.5 else -1)
            waypoints = [start_pos, approach_pos, detour_pos, retreat_pos, end_pos]
        
        # 生成方向（简单的垂直向下抓取）
        orientations = []
        for _ in waypoints:
            # 垂直向下的方向
            quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
            orientations.append(quat)
        
        # 组合位置和方向
        waypoint_poses = []
        for pos, quat in zip(waypoints, orientations):
            pose = np.concatenate([pos, quat])
            waypoint_poses.append(pose)
        
        # 插值生成完整轨迹
        trajectory = self._interpolate_waypoints(waypoint_poses, self.trajectory_length)
        
        return trajectory
    
    def _generate_pouring_trajectory(self, modality: int) -> np.ndarray:
        """
        生成倾倒轨迹
        
        Args:
            modality: 模态编号
            
        Returns:
            轨迹数组
        """
        # 随机生成起点和倾倒点
        start_pos = self._sample_random_position()
        pour_pos = self._sample_random_position()
        
        # 确保倾倒点在起点附近但有一定距离
        pour_pos = start_pos + np.random.uniform(-0.3, 0.3, 3)
        pour_pos[2] = max(pour_pos[2], start_pos[2] - 0.2)  # 倾倒点不能太低
        
        # 根据模态生成不同的倾倒动作
        if modality == 0:
            # 简单倾倒
            tilt_angle = np.pi / 4  # 45度倾斜
        elif modality == 1:
            # 大角度倾倒
            tilt_angle = np.pi / 2  # 90度倾斜
        else:
            # 小角度倾倒
            tilt_angle = np.pi / 6  # 30度倾斜
        
        # 生成轨迹关键点
        waypoints = []
        orientations = []
        
        # 起始姿态（水平持握）
        waypoints.append(start_pos)
        orientations.append(R.from_euler('xyz', [0, 0, 0]).as_quat())
        
        # 移动到倾倒位置
        waypoints.append(pour_pos)
        orientations.append(R.from_euler('xyz', [0, 0, 0]).as_quat())
        
        # 倾倒姿态
        waypoints.append(pour_pos)
        orientations.append(R.from_euler('xyz', [tilt_angle, 0, 0]).as_quat())
        
        # 回到水平
        waypoints.append(pour_pos)
        orientations.append(R.from_euler('xyz', [0, 0, 0]).as_quat())
        
        # 组合位置和方向
        waypoint_poses = []
        for pos, quat in zip(waypoints, orientations):
            pose = np.concatenate([pos, quat])
            waypoint_poses.append(pose)
        
        # 插值生成完整轨迹
        trajectory = self._interpolate_waypoints(waypoint_poses, self.trajectory_length)
        
        return trajectory
    
    def _generate_assembly_trajectory(self, modality: int) -> np.ndarray:
        """
        生成装配轨迹
        
        Args:
            modality: 模态编号
            
        Returns:
            轨迹数组
        """
        # 随机生成起点和装配点
        start_pos = self._sample_random_position()
        assembly_pos = self._sample_random_position()
        
        # 装配通常需要精确对齐
        assembly_pos[2] = max(assembly_pos[2], 0.1)  # 确保不会太低
        
        # 根据模态生成不同的装配策略
        if modality == 0:
            # 直接插入
            approach_offset = np.array([0, 0, 0.1])
        elif modality == 1:
            # 螺旋插入
            approach_offset = np.array([0.05, 0.05, 0.1])
        else:
            # 侧向插入
            approach_offset = np.array([0.1, 0, 0.05])
        
        # 生成轨迹关键点
        waypoints = []
        orientations = []
        
        # 起始位置
        waypoints.append(start_pos)
        orientations.append(R.from_euler('xyz', [0, 0, 0]).as_quat())
        
        # 接近位置
        approach_pos = assembly_pos + approach_offset
        waypoints.append(approach_pos)
        orientations.append(R.from_euler('xyz', [0, 0, 0]).as_quat())
        
        # 装配位置
        waypoints.append(assembly_pos)
        orientations.append(R.from_euler('xyz', [0, 0, 0]).as_quat())
        
        # 如果是螺旋插入，添加旋转动作
        if modality == 1:
            for i in range(3):
                rotation_angle = (i + 1) * np.pi / 2
                waypoints.append(assembly_pos)
                orientations.append(R.from_euler('xyz', [0, 0, rotation_angle]).as_quat())
        
        # 组合位置和方向
        waypoint_poses = []
        for pos, quat in zip(waypoints, orientations):
            pose = np.concatenate([pos, quat])
            waypoint_poses.append(pose)
        
        # 插值生成完整轨迹
        trajectory = self._interpolate_waypoints(waypoint_poses, self.trajectory_length)
        
        return trajectory
    
    def _sample_random_position(self) -> np.ndarray:
        """在工作空间内随机采样位置"""
        x = np.random.uniform(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1])
        y = np.random.uniform(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1])
        z = np.random.uniform(self.workspace_bounds['z'][0], self.workspace_bounds['z'][1])
        return np.array([x, y, z])
    
    def _interpolate_waypoints(self, waypoints: List[np.ndarray], num_points: int) -> np.ndarray:
        """
        在关键点之间插值生成轨迹
        
        Args:
            waypoints: 关键点列表
            num_points: 目标点数
            
        Returns:
            插值后的轨迹
        """
        if len(waypoints) < 2:
            raise ValueError("至少需要2个关键点")
        
        waypoints = np.array(waypoints)
        
        # 使用球面线性插值
        trajectory = interpolate_poses(waypoints, num_points, method='slerp')
        
        return trajectory
    
    def _validate_trajectory(self, trajectory: np.ndarray) -> bool:
        """
        验证轨迹是否有效
        
        Args:
            trajectory: 轨迹数组
            
        Returns:
            是否有效
        """
        # 检查是否在工作空间内
        in_bounds = check_workspace_bounds(trajectory, self.workspace_bounds)
        if not np.all(in_bounds):
            return False
        
        # 检查轨迹长度
        if len(trajectory) < 2:
            return False
        
        # 检查四元数是否归一化
        quaternions = trajectory[:, 3:]
        norms = np.linalg.norm(quaternions, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            # 尝试归一化
            trajectory[:, 3:] = normalize_quaternion(quaternions)
        
        return True
    
    def _apply_augmentation(self, trajectory: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            trajectory: 原始轨迹
            
        Returns:
            增强后的轨迹
        """
        augmented_trajectory = trajectory.copy()
        
        # 添加噪声
        if 'noise_level' in self.augmentation:
            noise_level = self.augmentation['noise_level']
            rotation_noise = self.augmentation.get('rotation_noise', 0.05)
            augmented_trajectory = add_noise_to_trajectory(
                augmented_trajectory, noise_level, rotation_noise
            )
        
        # 时间抖动
        if 'temporal_jitter' in self.augmentation:
            jitter = self.augmentation['temporal_jitter']
            # 随机改变轨迹长度
            length_factor = np.random.uniform(1 - jitter, 1 + jitter)
            new_length = int(len(trajectory) * length_factor)
            new_length = max(10, min(new_length, self.trajectory_length * 2))  # 限制范围
            
            if new_length != len(trajectory):
                augmented_trajectory = interpolate_poses(
                    augmented_trajectory, new_length, method='slerp'
                )
        
        return augmented_trajectory
    
    def save_data(self, trajectories: List[Dict], file_path: Path):
        """
        保存轨迹数据到HDF5文件
        
        Args:
            trajectories: 轨迹数据列表
            file_path: 保存路径
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(file_path, 'w') as f:
            # 保存元数据
            f.attrs['num_trajectories'] = len(trajectories)
            f.attrs['trajectory_length'] = self.trajectory_length
            f.attrs['time_horizon'] = self.time_horizon
            f.attrs['num_modalities'] = self.num_modalities
            
            # 保存任务信息
            task_names = [task['name'] for task in self.tasks]
            f.attrs['task_names'] = [name.encode('utf-8') for name in task_names]
            
            # 为每条轨迹创建组
            for i, traj_data in enumerate(trajectories):
                group = f.create_group(f'trajectory_{i}')
                
                # 保存轨迹数据
                group.create_dataset('trajectory', data=traj_data['trajectory'])
                group.create_dataset('timestamps', data=traj_data['timestamps'])
                group.create_dataset('start_pose', data=traj_data['start_pose'])
                group.create_dataset('end_pose', data=traj_data['end_pose'])
                
                # 保存元数据
                group.attrs['trajectory_id'] = traj_data['trajectory_id']
                group.attrs['task_name'] = traj_data['task_name'].encode('utf-8')
                group.attrs['task_id'] = traj_data['task_id']
                group.attrs['modality'] = traj_data['modality']
                group.attrs['length'] = traj_data['length']
                group.attrs['duration'] = traj_data['duration']
        
        self.logger.info(f"保存{len(trajectories)}条轨迹到: {file_path}")
    
    def load_data(self, file_path: Path) -> List[Dict]:
        """
        从HDF5文件加载轨迹数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            轨迹数据列表
        """
        trajectories = []
        
        with h5py.File(file_path, 'r') as f:
            num_trajectories = f.attrs['num_trajectories']
            
            for i in range(num_trajectories):
                group = f[f'trajectory_{i}']
                
                # 处理字符串属性，兼容不同版本的h5py
                task_name = group.attrs['task_name']
                if isinstance(task_name, bytes):
                    task_name = task_name.decode('utf-8')
                
                traj_data = {
                    'trajectory_id': group.attrs['trajectory_id'],
                    'task_name': task_name,
                    'task_id': group.attrs['task_id'],
                    'modality': group.attrs['modality'],
                    'length': group.attrs['length'],
                    'duration': group.attrs['duration'],
                    'trajectory': np.array(group['trajectory']),
                    'timestamps': np.array(group['timestamps']),
                    'start_pose': np.array(group['start_pose']),
                    'end_pose': np.array(group['end_pose'])
                }
                
                trajectories.append(traj_data)
        
        self.logger.info(f"从{file_path}加载{len(trajectories)}条轨迹")
        return trajectories
    
    def generate_statistics(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """
        生成数据集统计信息
        
        Args:
            trajectories: 轨迹数据列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_trajectories': len(trajectories),
            'tasks': {},
            'modalities': {},
            'trajectory_lengths': [],
            'durations': []
        }
        
        # 统计任务分布
        for traj in trajectories:
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
        
        # 计算统计量
        lengths = np.array(stats['trajectory_lengths'])
        durations = np.array(stats['durations'])
        
        stats['length_stats'] = {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths))
        }
        
        stats['duration_stats'] = {
            'mean': float(np.mean(durations)),
            'std': float(np.std(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations))
        }
        
        return stats