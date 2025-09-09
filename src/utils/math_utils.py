"""
Mathematical utilities for trajectory generation
轨迹生成数学工具
"""

import numpy as np
import torch
from typing import Tuple, List, Union, Optional
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d, CubicSpline
import warnings


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    将四元数转换为旋转矩阵
    
    Args:
        q: 四元数 [qx, qy, qz, qw] 或 [qw, qx, qy, qz]
        
    Returns:
        3x3旋转矩阵
    """
    if len(q) != 4:
        raise ValueError("四元数必须有4个元素")
    
    # 假设输入格式为 [qx, qy, qz, qw]
    if q.shape[-1] == 4:
        q = q[..., [3, 0, 1, 2]]  # 转换为 [qw, qx, qy, qz]
    
    rotation = R.from_quat(q)
    return rotation.as_matrix()


def rotation_matrix_to_quaternion(R_matrix: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为四元数
    
    Args:
        R_matrix: 3x3旋转矩阵
        
    Returns:
        四元数 [qx, qy, qz, qw]
    """
    rotation = R.from_matrix(R_matrix)
    q = rotation.as_quat()  # [qx, qy, qz, qw]
    return q


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    将位姿转换为4x4变换矩阵
    
    Args:
        pose: 位姿 [x, y, z, qx, qy, qz, qw]
        
    Returns:
        4x4变换矩阵
    """
    if len(pose) != 7:
        raise ValueError("位姿必须有7个元素: [x, y, z, qx, qy, qz, qw]")
    
    position = pose[:3]
    quaternion = pose[3:]
    
    # 创建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    T[:3, 3] = position
    
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """
    将4x4变换矩阵转换为位姿
    
    Args:
        T: 4x4变换矩阵
        
    Returns:
        位姿 [x, y, z, qx, qy, qz, qw]
    """
    if T.shape != (4, 4):
        raise ValueError("变换矩阵必须是4x4")
    
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    
    return np.concatenate([position, quaternion])


def interpolate_poses(poses: np.ndarray, num_points: int, method: str = 'slerp') -> np.ndarray:
    """
    插值位姿序列
    
    Args:
        poses: 位姿序列 [N, 7]
        num_points: 插值后的点数
        method: 插值方法 ('linear', 'cubic', 'slerp')
        
    Returns:
        插值后的位姿序列 [num_points, 7]
    """
    if len(poses) < 2:
        raise ValueError("至少需要2个位姿点进行插值")
    
    original_indices = np.linspace(0, len(poses) - 1, len(poses))
    new_indices = np.linspace(0, len(poses) - 1, num_points)
    
    # 插值位置
    positions = poses[:, :3]
    if method == 'linear':
        interp_func = interp1d(original_indices, positions, axis=0, kind='linear')
    elif method == 'cubic':
        interp_func = interp1d(original_indices, positions, axis=0, kind='cubic')
    else:
        interp_func = interp1d(original_indices, positions, axis=0, kind='linear')
    
    new_positions = interp_func(new_indices)
    
    # 插值旋转（使用球面线性插值）
    quaternions = poses[:, 3:]
    if method == 'slerp':
        new_quaternions = slerp_interpolation(quaternions, new_indices, original_indices)
    else:
        # 对于非slerp方法，使用线性插值（不推荐用于旋转）
        interp_func = interp1d(original_indices, quaternions, axis=0, kind='linear')
        new_quaternions = interp_func(new_indices)
        # 归一化四元数
        new_quaternions = new_quaternions / np.linalg.norm(new_quaternions, axis=1, keepdims=True)
    
    return np.concatenate([new_positions, new_quaternions], axis=1)


def slerp_interpolation(quaternions: np.ndarray, new_indices: np.ndarray, 
                       original_indices: np.ndarray) -> np.ndarray:
    """
    球面线性插值四元数
    
    Args:
        quaternions: 原始四元数序列
        new_indices: 新的索引
        original_indices: 原始索引
        
    Returns:
        插值后的四元数
    """
    rotations = R.from_quat(quaternions)
    
    # 使用scipy的Slerp插值
    from scipy.spatial.transform import Slerp
    slerp = Slerp(original_indices, rotations)
    
    new_rotations = slerp(new_indices)
    return new_rotations.as_quat()


def compute_trajectory_smoothness(trajectory: np.ndarray, dt: float = 0.1) -> float:
    """
    计算轨迹平滑度（基于加速度变化）
    
    Args:
        trajectory: 轨迹 [N, 7] (位置+四元数)
        dt: 时间步长
        
    Returns:
        平滑度分数（越小越平滑）
    """
    if len(trajectory) < 3:
        return 0.0
    
    positions = trajectory[:, :3]
    
    # 计算速度
    velocities = np.diff(positions, axis=0) / dt
    
    # 计算加速度
    accelerations = np.diff(velocities, axis=0) / dt
    
    # 计算加速度的变化率（jerk）
    jerks = np.diff(accelerations, axis=0) / dt
    
    # 计算平滑度分数（jerk的RMS）
    smoothness = np.sqrt(np.mean(np.sum(jerks**2, axis=1)))
    
    return smoothness


def compute_trajectory_length(trajectory: np.ndarray) -> float:
    """
    计算轨迹长度
    
    Args:
        trajectory: 轨迹 [N, 7]
        
    Returns:
        轨迹总长度
    """
    if len(trajectory) < 2:
        return 0.0
    
    positions = trajectory[:, :3]
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.sum(distances)


def compute_pose_distance(pose1: np.ndarray, pose2: np.ndarray, 
                         position_weight: float = 1.0, 
                         rotation_weight: float = 1.0) -> float:
    """
    计算两个位姿之间的距离
    
    Args:
        pose1: 位姿1 [x, y, z, qx, qy, qz, qw]
        pose2: 位姿2 [x, y, z, qx, qy, qz, qw]
        position_weight: 位置权重
        rotation_weight: 旋转权重
        
    Returns:
        位姿距离
    """
    # 位置距离
    pos_dist = np.linalg.norm(pose1[:3] - pose2[:3])
    
    # 旋转距离（四元数角度差）
    q1 = pose1[3:]
    q2 = pose2[3:]
    
    # 确保四元数归一化
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算角度差
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, 0, 1)  # 避免数值误差
    angle_diff = 2 * np.arccos(dot_product)
    
    return position_weight * pos_dist + rotation_weight * angle_diff


def generate_workspace_bounds(center: np.ndarray, size: np.ndarray) -> dict:
    """
    生成工作空间边界
    
    Args:
        center: 工作空间中心 [x, y, z]
        size: 工作空间尺寸 [width, height, depth]
        
    Returns:
        工作空间边界字典
    """
    half_size = size / 2
    bounds = {
        'x': [center[0] - half_size[0], center[0] + half_size[0]],
        'y': [center[1] - half_size[1], center[1] + half_size[1]],
        'z': [center[2] - half_size[2], center[2] + half_size[2]]
    }
    return bounds


def check_workspace_bounds(poses: np.ndarray, bounds: dict) -> np.ndarray:
    """
    检查位姿是否在工作空间内
    
    Args:
        poses: 位姿序列 [N, 7]
        bounds: 工作空间边界
        
    Returns:
        布尔数组，表示每个位姿是否在边界内
    """
    positions = poses[:, :3]
    
    in_bounds = np.ones(len(positions), dtype=bool)
    
    for i, axis in enumerate(['x', 'y', 'z']):
        if axis in bounds:
            min_bound, max_bound = bounds[axis]
            in_bounds &= (positions[:, i] >= min_bound) & (positions[:, i] <= max_bound)
    
    return in_bounds


def add_noise_to_trajectory(trajectory: np.ndarray, 
                          position_noise: float = 0.01,
                          rotation_noise: float = 0.05) -> np.ndarray:
    """
    向轨迹添加噪声
    
    Args:
        trajectory: 原始轨迹 [N, 7]
        position_noise: 位置噪声标准差
        rotation_noise: 旋转噪声标准差（弧度）
        
    Returns:
        带噪声的轨迹
    """
    noisy_trajectory = trajectory.copy()
    
    # 添加位置噪声
    position_noise_array = np.random.normal(0, position_noise, trajectory[:, :3].shape)
    noisy_trajectory[:, :3] += position_noise_array
    
    # 添加旋转噪声
    for i in range(len(trajectory)):
        # 生成随机旋转轴和角度
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.normal(0, rotation_noise)
        
        # 创建噪声旋转
        noise_rotation = R.from_rotvec(axis * angle)
        
        # 应用噪声到原始旋转
        original_rotation = R.from_quat(trajectory[i, 3:])
        noisy_rotation = noise_rotation * original_rotation
        
        noisy_trajectory[i, 3:] = noisy_rotation.as_quat()
    
    return noisy_trajectory


def compute_trajectory_diversity(trajectories: List[np.ndarray], 
                               num_samples: int = 100) -> float:
    """
    计算轨迹集合的多样性
    
    Args:
        trajectories: 轨迹列表
        num_samples: 采样点数量
        
    Returns:
        多样性分数
    """
    if len(trajectories) < 2:
        return 0.0
    
    # 将所有轨迹插值到相同长度
    interpolated_trajectories = []
    for traj in trajectories:
        if len(traj) >= 2:
            interp_traj = interpolate_poses(traj, num_samples)
            interpolated_trajectories.append(interp_traj)
    
    if len(interpolated_trajectories) < 2:
        return 0.0
    
    # 计算轨迹之间的平均距离
    total_distance = 0.0
    count = 0
    
    for i in range(len(interpolated_trajectories)):
        for j in range(i + 1, len(interpolated_trajectories)):
            traj1 = interpolated_trajectories[i]
            traj2 = interpolated_trajectories[j]
            
            # 计算对应点之间的距离
            distances = []
            for k in range(num_samples):
                dist = compute_pose_distance(traj1[k], traj2[k])
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            total_distance += avg_distance
            count += 1
    
    diversity = total_distance / count if count > 0 else 0.0
    return diversity


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    归一化四元数
    
    Args:
        q: 四元数或四元数数组
        
    Returns:
        归一化后的四元数
    """
    if q.ndim == 1:
        return q / np.linalg.norm(q)
    else:
        return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法
    
    Args:
        q1: 四元数1 [qx, qy, qz, qw]
        q2: 四元数2 [qx, qy, qz, qw]
        
    Returns:
        乘积四元数
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    result = r1 * r2
    return result.as_quat()


def compute_angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    """
    从四元数序列计算角速度
    
    Args:
        quaternions: 四元数序列 [N, 4]
        dt: 时间步长
        
    Returns:
        角速度序列 [N-1, 3]
    """
    if len(quaternions) < 2:
        return np.array([])
    
    angular_velocities = []
    
    for i in range(len(quaternions) - 1):
        q1 = quaternions[i]
        q2 = quaternions[i + 1]
        
        # 计算相对旋转
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        relative_rotation = r2 * r1.inv()
        
        # 转换为旋转向量（轴角表示）
        rotvec = relative_rotation.as_rotvec()
        
        # 计算角速度
        angular_velocity = rotvec / dt
        angular_velocities.append(angular_velocity)
    
    return np.array(angular_velocities)