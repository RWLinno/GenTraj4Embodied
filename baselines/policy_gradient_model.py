"""
Policy Gradient Model for Trajectory Generation
策略梯度轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import RLBasedModel


class PolicyGradientTrajectoryModel(RLBasedModel):
    """
    基于策略梯度的轨迹生成模型
    使用REINFORCE算法优化轨迹生成策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gamma = config.get('gamma', 0.99)  # 折扣因子
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)  # 熵正则化系数
        self.baseline_coeff = config.get('baseline_coeff', 0.5)  # 基线损失系数
        self.use_baseline = config.get('use_baseline', True)
        
        # 策略网络
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('policy_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # 基线网络（价值函数近似）
        if self.use_baseline:
            self.baseline_net = BaselineNetwork(
                state_dim=self.state_dim,
                hidden_dims=config.get('baseline_hidden_dims', [256, 256]),
                activation=config.get('activation', 'relu'),
                dropout=self.dropout
            )
        
        # 轨迹环境
        self.trajectory_env = TrajectoryEnvironment(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_seq_length=self.max_seq_length,
            reward_type=self.reward_type
        )
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate
        )
        
        if self.use_baseline:
            self.baseline_optimizer = torch.optim.Adam(
                self.baseline_net.parameters(),
                lr=self.learning_rate
            )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 策略采样生成轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 初始化轨迹
        trajectory = torch.zeros(batch_size, self.max_seq_length, self.output_dim, device=device)
        trajectory[:, 0] = start_pose  # 设置起始点
        
        # 构建初始状态
        state = self._construct_state(start_pose, end_pose, trajectory, 0)
        
        # 逐步生成轨迹
        for t in range(1, self.max_seq_length):
            # 策略网络预测动作分布
            action_dist = self.policy_net(state)
            
            # 采样动作
            if self.training:
                action = action_dist.sample()
            else:
                action = action_dist.mean  # 确定性策略用于推理
            
            # 更新轨迹
            trajectory[:, t] = action
            
            # 更新状态
            if t < self.max_seq_length - 1:
                state = self._construct_state(start_pose, end_pose, trajectory, t)
        
        # 强制终点约束
        trajectory[:, -1] = end_pose
        
        return trajectory
    
    def _construct_state(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                        trajectory: torch.Tensor, current_step: int) -> torch.Tensor:
        """
        构建当前状态表示
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            trajectory: 当前轨迹 [batch_size, seq_length, output_dim]
            current_step: 当前步骤
            
        Returns:
            状态表示 [batch_size, state_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 当前位置
        current_pos = trajectory[:, current_step]
        
        # 目标位置
        target_pos = end_pose
        
        # 相对位置
        relative_pos = target_pos - current_pos
        
        # 时间步信息
        time_info = torch.ones(batch_size, 1, device=device) * (current_step / self.max_seq_length)
        
        # 历史轨迹信息（最近几步的平均速度）
        if current_step > 0:
            recent_velocity = trajectory[:, current_step] - trajectory[:, max(0, current_step-1)]
        else:
            recent_velocity = torch.zeros_like(current_pos)
        
        # 组合状态
        state = torch.cat([
            current_pos,      # 当前位置
            target_pos,       # 目标位置
            relative_pos,     # 相对位置
            time_info,        # 时间信息
            recent_velocity   # 最近速度
        ], dim=-1)
        
        return state
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练步骤 - REINFORCE算法
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        target_trajectory = batch.get('trajectory', None)
        
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 收集轨迹和奖励
        trajectories, log_probs, rewards, baselines = self._collect_trajectories(
            start_pose, end_pose, target_trajectory
        )
        
        # 计算回报
        returns = self._compute_returns(rewards)
        
        # 计算优势
        if self.use_baseline:
            advantages = returns - baselines
        else:
            advantages = returns
        
        # 策略损失（REINFORCE）
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # 熵正则化
        entropy_loss = -self.entropy_coeff * self._compute_entropy(trajectories, start_pose, end_pose)
        
        # 总策略损失
        total_policy_loss = policy_loss + entropy_loss
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_policy_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_return': returns.mean().item()
        }
        
        # 基线损失
        if self.use_baseline:
            baseline_loss = nn.MSELoss()(baselines, returns.detach())
            
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.baseline_net.parameters(), 0.5)
            self.baseline_optimizer.step()
            
            loss_dict['baseline_loss'] = baseline_loss.item()
            loss_dict['total_loss'] += baseline_loss.item() * self.baseline_coeff
        
        return loss_dict
    
    def _collect_trajectories(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                            target_trajectory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        收集轨迹、对数概率、奖励和基线值
        
        Returns:
            trajectories: 生成的轨迹 [batch_size, seq_length, output_dim]
            log_probs: 对数概率 [batch_size, seq_length-1]
            rewards: 奖励 [batch_size, seq_length-1]
            baselines: 基线值 [batch_size, seq_length-1]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 初始化
        trajectory = torch.zeros(batch_size, self.max_seq_length, self.output_dim, device=device)
        trajectory[:, 0] = start_pose
        
        log_probs = []
        rewards = []
        baselines = []
        
        # 逐步生成
        for t in range(1, self.max_seq_length):
            # 构建状态
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            
            # 策略预测
            action_dist = self.policy_net(state)
            
            # 采样动作
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            # 更新轨迹
            trajectory[:, t] = action
            
            # 计算奖励
            reward = self.trajectory_env.compute_reward(
                trajectory, start_pose, end_pose, t, target_trajectory
            )
            
            # 基线值
            if self.use_baseline:
                baseline = self.baseline_net(state).squeeze(-1)
            else:
                baseline = torch.zeros(batch_size, device=device)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            baselines.append(baseline)
        
        # 强制终点约束
        trajectory[:, -1] = end_pose
        
        # 转换为张量
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, seq_length-1]
        rewards = torch.stack(rewards, dim=1)      # [batch_size, seq_length-1]
        baselines = torch.stack(baselines, dim=1)  # [batch_size, seq_length-1]
        
        return trajectory, log_probs, rewards, baselines
    
    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算折扣回报
        
        Args:
            rewards: 奖励序列 [batch_size, seq_length-1]
            
        Returns:
            returns: 折扣回报 [batch_size, seq_length-1]
        """
        batch_size, seq_length = rewards.shape
        returns = torch.zeros_like(rewards)
        
        # 从后往前计算
        running_return = torch.zeros(batch_size, device=rewards.device)
        for t in reversed(range(seq_length)):
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return
        
        return returns
    
    def _compute_entropy(self, trajectories: torch.Tensor, start_pose: torch.Tensor, 
                        end_pose: torch.Tensor) -> torch.Tensor:
        """
        计算策略熵
        """
        total_entropy = 0.0
        count = 0
        
        for t in range(1, self.max_seq_length):
            state = self._construct_state(start_pose, end_pose, trajectories, t-1)
            action_dist = self.policy_net(state)
            entropy = action_dist.entropy().sum(dim=-1).mean()
            total_entropy += entropy
            count += 1
        
        return total_entropy / count if count > 0 else torch.tensor(0.0)
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            
        Returns:
            生成的轨迹 [num_points, output_dim]
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self.forward(start_tensor, end_tensor)
            
        self.max_seq_length = original_seq_length
        
        return trajectory.squeeze(0).numpy()
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    start_poses: torch.Tensor, end_poses: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算损失函数
        """
        batch = {
            'start_pose': start_poses,
            'end_pose': end_poses,
            'trajectory': targets
        }
        
        loss_dict = self.training_step(batch)
        return torch.tensor(loss_dict['total_loss'])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'entropy_coeff': self.entropy_coeff,
            'baseline_coeff': self.baseline_coeff,
            'use_baseline': self.use_baseline,
            'model_category': 'RL-based Methods'
        })
        return info


class PolicyNetwork(nn.Module):
    """
    策略网络
    输出动作的概率分布
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 网络层
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 输出层（均值和标准差）
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.logstd_head = nn.Linear(input_dim, action_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.distributions.Normal:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            
        Returns:
            action_dist: 动作分布
        """
        # 特征提取
        features = self.backbone(state)
        
        # 均值和标准差
        mean = self.mean_head(features)
        logstd = self.logstd_head(features)
        
        # 限制标准差范围
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        
        # 构建正态分布
        action_dist = torch.distributions.Normal(mean, std)
        
        return action_dist


class BaselineNetwork(nn.Module):
    """
    基线网络（价值函数）
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 网络层
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            
        Returns:
            value: 价值 [batch_size, 1]
        """
        return self.network(state)


class TrajectoryEnvironment:
    """
    轨迹生成环境
    定义奖励函数和环境动态
    """
    
    def __init__(self, input_dim: int, output_dim: int, max_seq_length: int, reward_type: str = 'sparse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        self.reward_type = reward_type
    
    def compute_reward(self, trajectory: torch.Tensor, start_pose: torch.Tensor,
                      end_pose: torch.Tensor, current_step: int,
                      target_trajectory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算奖励
        
        Args:
            trajectory: 当前轨迹 [batch_size, seq_length, output_dim]
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            current_step: 当前步骤
            target_trajectory: 目标轨迹（如果有）
            
        Returns:
            reward: 奖励 [batch_size]
        """
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        if self.reward_type == 'sparse':
            # 稀疏奖励：只在最后一步给奖励
            if current_step == self.max_seq_length - 1:
                # 终点距离奖励
                final_pos = trajectory[:, -1]
                distance_to_goal = torch.norm(final_pos - end_pose, dim=-1)
                reward = -distance_to_goal
            else:
                reward = torch.zeros(batch_size, device=device)
        
        elif self.reward_type == 'dense':
            # 密集奖励：每步都给奖励
            current_pos = trajectory[:, current_step]
            
            # 目标导向奖励
            distance_to_goal = torch.norm(current_pos - end_pose, dim=-1)
            goal_reward = -distance_to_goal
            
            # 平滑性奖励
            if current_step > 1:
                velocity_curr = trajectory[:, current_step] - trajectory[:, current_step-1]
                velocity_prev = trajectory[:, current_step-1] - trajectory[:, current_step-2]
                acceleration = velocity_curr - velocity_prev
                smoothness_reward = -torch.norm(acceleration, dim=-1) * 0.1
            else:
                smoothness_reward = torch.zeros(batch_size, device=device)
            
            reward = goal_reward + smoothness_reward
        
        elif self.reward_type == 'imitation' and target_trajectory is not None:
            # 模仿学习奖励
            current_pos = trajectory[:, current_step]
            target_pos = target_trajectory[:, current_step]
            
            # 与目标轨迹的距离
            imitation_reward = -torch.norm(current_pos - target_pos, dim=-1)
            
            # 平滑性奖励
            if current_step > 1:
                velocity_curr = trajectory[:, current_step] - trajectory[:, current_step-1]
                velocity_prev = trajectory[:, current_step-1] - trajectory[:, current_step-2]
                acceleration = velocity_curr - velocity_prev
                smoothness_reward = -torch.norm(acceleration, dim=-1) * 0.1
            else:
                smoothness_reward = torch.zeros(batch_size, device=device)
            
            reward = imitation_reward + smoothness_reward
        
        else:
            reward = torch.zeros(batch_size, device=device)
        
        return reward


class AdvancedPolicyGradient(PolicyGradientTrajectoryModel):
    """
    高级策略梯度模型
    包含更多技巧和改进
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_gae = config.get('use_gae', True)  # 广义优势估计
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.use_reward_normalization = config.get('use_reward_normalization', True)
        self.use_advantage_normalization = config.get('use_advantage_normalization', True)
        
        # 奖励统计
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def _compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor,
                               next_values: torch.Tensor) -> torch.Tensor:
        """
        计算广义优势估计(GAE)
        
        Args:
            rewards: 奖励 [batch_size, seq_length]
            values: 当前状态价值 [batch_size, seq_length]
            next_values: 下一状态价值 [batch_size, seq_length]
            
        Returns:
            advantages: GAE优势 [batch_size, seq_length]
        """
        batch_size, seq_length = rewards.shape
        advantages = torch.zeros_like(rewards)
        
        gae = torch.zeros(batch_size, device=rewards.device)
        
        for t in reversed(range(seq_length)):
            if t == seq_length - 1:
                next_value = torch.zeros(batch_size, device=rewards.device)
            else:
                next_value = next_values[:, t]
            
            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[:, t] = gae
        
        return advantages
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        奖励归一化
        """
        if self.use_reward_normalization:
            # 更新统计信息
            batch_mean = rewards.mean().item()
            batch_std = rewards.std().item() + 1e-8
            
            self.reward_count += 1
            alpha = 1.0 / self.reward_count
            
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
            self.reward_std = (1 - alpha) * self.reward_std + alpha * batch_std
            
            # 归一化
            normalized_rewards = (rewards - self.reward_mean) / self.reward_std
            return normalized_rewards
        else:
            return rewards
    
    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        优势归一化
        """
        if self.use_advantage_normalization:
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            return (advantages - mean) / std
        else:
            return advantages


class HierarchicalPolicyGradient(PolicyGradientTrajectoryModel):
    """
    分层策略梯度模型
    使用高级策略和低级策略的分层结构
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_waypoints = config.get('num_waypoints', 5)
        self.waypoint_horizon = self.max_seq_length // self.num_waypoints
        
        # 高级策略（生成路径点）
        self.high_level_policy = PolicyNetwork(
            state_dim=self.input_dim * 2,  # start + end pose
            action_dim=self.output_dim * self.num_waypoints,
            hidden_dims=config.get('high_level_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # 低级策略（连接路径点）
        self.low_level_policy = PolicyNetwork(
            state_dim=self.output_dim * 2,  # current + target waypoint
            action_dim=self.output_dim,
            hidden_dims=config.get('low_level_hidden_dims', [128, 128]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # 分别的优化器
        self.high_level_optimizer = torch.optim.Adam(
            self.high_level_policy.parameters(),
            lr=self.learning_rate
        )
        
        self.low_level_optimizer = torch.optim.Adam(
            self.low_level_policy.parameters(),
            lr=self.learning_rate
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        分层前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 高级策略生成路径点
        high_level_state = torch.cat([start_pose, end_pose], dim=-1)
        waypoint_dist = self.high_level_policy(high_level_state)
        
        if self.training:
            waypoints_flat = waypoint_dist.sample()
        else:
            waypoints_flat = waypoint_dist.mean
        
        waypoints = waypoints_flat.view(batch_size, self.num_waypoints, self.output_dim)
        
        # 添加起始和终止点
        all_waypoints = torch.cat([
            start_pose.unsqueeze(1),
            waypoints,
            end_pose.unsqueeze(1)
        ], dim=1)  # [batch_size, num_waypoints+2, output_dim]
        
        # 低级策略连接路径点
        trajectory = torch.zeros(batch_size, self.max_seq_length, self.output_dim, device=device)
        
        for i in range(self.num_waypoints + 1):
            start_wp = all_waypoints[:, i]
            end_wp = all_waypoints[:, i + 1]
            
            # 计算这段的时间步
            start_idx = i * self.waypoint_horizon
            end_idx = min((i + 1) * self.waypoint_horizon, self.max_seq_length)
            
            # 生成这段轨迹
            segment_length = end_idx - start_idx
            for t in range(segment_length):
                if t == 0:
                    trajectory[:, start_idx + t] = start_wp
                elif t == segment_length - 1:
                    trajectory[:, start_idx + t] = end_wp
                else:
                    # 低级策略预测
                    progress = t / (segment_length - 1)
                    current_target = start_wp + progress * (end_wp - start_wp)
                    current_pos = trajectory[:, start_idx + t - 1]
                    
                    low_level_state = torch.cat([current_pos, current_target], dim=-1)
                    action_dist = self.low_level_policy(low_level_state)
                    
                    if self.training:
                        action = action_dist.sample()
                    else:
                        action = action_dist.mean
                    
                    trajectory[:, start_idx + t] = action
        
        return trajectory