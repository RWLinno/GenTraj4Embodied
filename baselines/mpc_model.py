"""
Model Predictive Control (MPC) for Trajectory Generation
基于模型预测控制的轨迹生成方法

MPC通过在有限时域内求解优化问题来生成轨迹，
考虑系统动力学约束和控制输入限制。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.optimize import minimize
import cvxpy as cp

from ...base_model import BaseTrajectoryModel


class MPCTrajectoryModel(BaseTrajectoryModel):
    """
    Model Predictive Control轨迹生成模型
    
    基于模型预测控制理论，通过滚动优化生成最优轨迹
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # MPC参数
        self.prediction_horizon = config.get('prediction_horizon', 10)
        self.control_horizon = config.get('control_horizon', 5)
        self.dt = config.get('dt', 0.1)
        
        # 权重参数
        self.Q = torch.eye(self.output_dim) * config.get('state_weight', 1.0)
        self.R = torch.eye(self.output_dim) * config.get('control_weight', 0.1)
        self.Qf = torch.eye(self.output_dim) * config.get('terminal_weight', 10.0)
        
        # 约束参数
        self.max_velocity = config.get('max_velocity', 1.0)
        self.max_acceleration = config.get('max_acceleration', 0.5)
        
        # 系统动力学模型 (简化为双积分器模型)
        self.A = torch.eye(self.output_dim * 2)  # [position, velocity]
        self.A[:self.output_dim, self.output_dim:] = torch.eye(self.output_dim) * self.dt
        
        self.B = torch.zeros(self.output_dim * 2, self.output_dim)
        self.B[self.output_dim:, :] = torch.eye(self.output_dim) * self.dt
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        MPC前向传播
        
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
        # 初始化状态 [position, velocity]
        current_state = np.concatenate([start_pose, np.zeros(self.output_dim)])
        target_state = np.concatenate([end_pose, np.zeros(self.output_dim)])
        
        trajectory = [start_pose.copy()]
        
        # MPC滚动优化
        for step in range(self.max_seq_length - 1):
            # 求解MPC优化问题
            optimal_control = self._solve_mpc(current_state, target_state)
            
            # 应用第一个控制输入
            current_state = self._apply_dynamics(current_state, optimal_control[0])
            trajectory.append(current_state[:self.output_dim].copy())
            
            # 检查是否到达目标
            if np.linalg.norm(current_state[:self.output_dim] - end_pose) < 1e-3:
                break
        
        return np.array(trajectory)
    
    def _solve_mpc(self, current_state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        """
        求解MPC优化问题
        
        Args:
            current_state: 当前状态 [state_dim]
            target_state: 目标状态 [state_dim]
            
        Returns:
            最优控制序列 [control_horizon, output_dim]
        """
        # 使用cvxpy求解二次规划问题
        state_dim = len(current_state)
        
        # 决策变量
        x = cp.Variable((self.prediction_horizon + 1, state_dim))
        u = cp.Variable((self.control_horizon, self.output_dim))
        
        # 目标函数
        cost = 0
        
        # 状态代价
        for k in range(self.prediction_horizon):
            state_error = x[k, :self.output_dim] - target_state[:self.output_dim]
            cost += cp.quad_form(state_error, self.Q.numpy())
        
        # 控制代价
        for k in range(self.control_horizon):
            cost += cp.quad_form(u[k], self.R.numpy())
        
        # 终端代价
        terminal_error = x[self.prediction_horizon, :self.output_dim] - target_state[:self.output_dim]
        cost += cp.quad_form(terminal_error, self.Qf.numpy())
        
        # 约束
        constraints = [x[0] == current_state]  # 初始状态约束
        
        # 动力学约束
        A_np = self.A.numpy()
        B_np = self.B.numpy()
        
        for k in range(min(self.control_horizon, self.prediction_horizon)):
            constraints.append(x[k+1] == A_np @ x[k] + B_np @ u[k])
        
        # 如果控制时域小于预测时域，后续状态用零控制输入
        for k in range(self.control_horizon, self.prediction_horizon):
            constraints.append(x[k+1] == A_np @ x[k])
        
        # 速度约束
        for k in range(self.prediction_horizon + 1):
            constraints.append(cp.abs(x[k, self.output_dim:]) <= self.max_velocity)
        
        # 加速度约束 (控制输入)
        for k in range(self.control_horizon):
            constraints.append(cp.abs(u[k]) <= self.max_acceleration)
        
        # 求解优化问题
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return u.value
            else:
                # 如果优化失败，返回零控制输入
                return np.zeros((self.control_horizon, self.output_dim))
        except:
            # 求解器错误时的备选方案
            return np.zeros((self.control_horizon, self.output_dim))
    
    def _apply_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        应用系统动力学
        
        Args:
            state: 当前状态
            control: 控制输入
            
        Returns:
            下一时刻状态
        """
        A_np = self.A.numpy()
        B_np = self.B.numpy()
        
        next_state = A_np @ state + B_np @ control
        return next_state
    
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
            
            # 如果生成的轨迹点数不足，进行插值
            if len(trajectory) < num_points:
                from scipy.interpolate import interp1d
                
                t_old = np.linspace(0, 1, len(trajectory))
                t_new = np.linspace(0, 1, num_points)
                
                interpolated = []
                for dim in range(self.output_dim):
                    f = interp1d(t_old, trajectory[:, dim], kind='cubic', 
                               fill_value='extrapolate')
                    interpolated.append(f(t_new))
                
                trajectory = np.column_stack(interpolated)
            
            return trajectory[:num_points]
        
        finally:
            # 恢复原始序列长度
            self.max_seq_length = original_max_seq_length
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        计算MPC损失函数
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            
        Returns:
            损失值
        """
        # 位置误差
        position_loss = torch.mean((predictions - targets) ** 2)
        
        # 平滑度损失 (最小化加速度)
        pred_acc = torch.diff(predictions, n=2, dim=1)
        smoothness_loss = torch.mean(pred_acc ** 2)
        
        # 终点误差
        end_loss = torch.mean((predictions[:, -1] - targets[:, -1]) ** 2)
        
        # 总损失
        total_loss = position_loss + 0.1 * smoothness_loss + 2.0 * end_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'model_type': 'Model Predictive Control',
            'prediction_horizon': self.prediction_horizon,
            'control_horizon': self.control_horizon,
            'dt': self.dt,
            'max_velocity': self.max_velocity,
            'max_acceleration': self.max_acceleration
        })
        return info