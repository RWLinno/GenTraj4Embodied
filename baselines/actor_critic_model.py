"""
Actor-Critic Model for Trajectory Generation
Actor-Critic轨迹生成模型
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


class ActorCriticTrajectoryModel(RLBasedModel):
    """
    基于Actor-Critic的轨迹生成模型
    同时学习策略(Actor)和价值函数(Critic)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.value_coeff = config.get('value_coeff', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.use_gae = config.get('use_gae', True)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        
        # Actor网络
        self.actor = ActorNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('actor_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # Critic网络
        self.critic = CriticNetwork(
            state_dim=self.state_dim,
            hidden_dims=config.get('critic_hidden_dims', [256, 256]),
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
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate
        )
        
        # 或者分别的优化器
        self.use_separate_optimizers = config.get('use_separate_optimizers', False)
        if self.use_separate_optimizers:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.learning_rate
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.learning_rate
            )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - Actor-Critic采样生成轨迹
        
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
        trajectory[:, 0] = start_pose
        
        # 逐步生成轨迹
        for t in range(1, self.max_seq_length):
            # 构建状态
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            
            # Actor预测动作分布
            action_dist = self.actor(state)
            
            # 采样动作
            if self.training:
                action = action_dist.sample()
            else:
                action = action_dist.mean
            
            # 更新轨迹
            trajectory[:, t] = action
        
        # 强制终点约束
        trajectory[:, -1] = end_pose
        
        return trajectory
    
    def _construct_state(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                        trajectory: torch.Tensor, current_step: int) -> torch.Tensor:
        """
        构建当前状态表示
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 当前位置
        current_pos = trajectory[:, current_step]
        
        # 目标位置
        target_pos = end_pose
        
        # 相对位置和距离
        relative_pos = target_pos - current_pos
        distance_to_goal = torch.norm(relative_pos, dim=-1, keepdim=True)
        
        # 时间步信息
        time_info = torch.ones(batch_size, 1, device=device) * (current_step / self.max_seq_length)
        remaining_time = torch.ones(batch_size, 1, device=device) * ((self.max_seq_length - current_step) / self.max_seq_length)
        
        # 历史轨迹信息
        if current_step > 0:
            recent_velocity = trajectory[:, current_step] - trajectory[:, max(0, current_step-1)]
        else:
            recent_velocity = torch.zeros_like(current_pos)
        
        # 平均速度（到目前为止）
        if current_step > 0:
            total_distance = torch.norm(trajectory[:, current_step] - trajectory[:, 0], dim=-1, keepdim=True)
            avg_speed = total_distance / (current_step + 1e-8)
        else:
            avg_speed = torch.zeros(batch_size, 1, device=device)
        
        # 组合状态
        state = torch.cat([
            current_pos,        # 当前位置
            target_pos,         # 目标位置
            relative_pos,       # 相对位置
            distance_to_goal,   # 到目标的距离
            time_info,          # 当前时间步比例
            remaining_time,     # 剩余时间比例
            recent_velocity,    # 最近速度
            avg_speed          # 平均速度
        ], dim=-1)
        
        return state
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练步骤 - Actor-Critic算法
        
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
        
        # 收集轨迹数据
        trajectories, log_probs, rewards, values, next_values = self._collect_episode_data(
            start_pose, end_pose, target_trajectory
        )
        
        # 计算优势和回报
        if self.use_gae:
            advantages = self._compute_gae_advantages(rewards, values, next_values)
            returns = advantages + values
        else:
            returns = self._compute_returns(rewards)
            advantages = returns - values
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor损失（策略梯度）
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic损失（价值函数）
        critic_loss = nn.MSELoss()(values, returns.detach())
        
        # 熵损失（鼓励探索）
        entropy_loss = -self._compute_entropy_loss(trajectories, start_pose, end_pose)
        
        # 总损失
        total_loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss
        
        # 反向传播
        if self.use_separate_optimizers:
            # 分别更新Actor和Critic
            self.actor_optimizer.zero_grad()
            actor_total_loss = actor_loss + self.entropy_coeff * entropy_loss
            actor_total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        else:
            # 联合更新
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 
                self.max_grad_norm
            )
            self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_return': returns.mean().item(),
            'avg_value': values.mean().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
    
    def _collect_episode_data(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                            target_trajectory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        收集一个episode的数据
        
        Returns:
            trajectories: 生成的轨迹 [batch_size, seq_length, output_dim]
            log_probs: 对数概率 [batch_size, seq_length-1]
            rewards: 奖励 [batch_size, seq_length-1]
            values: 状态价值 [batch_size, seq_length-1]
            next_values: 下一状态价值 [batch_size, seq_length-1]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 初始化
        trajectory = torch.zeros(batch_size, self.max_seq_length, self.output_dim, device=device)
        trajectory[:, 0] = start_pose
        
        log_probs = []
        rewards = []
        values = []
        next_values = []
        
        # 逐步生成和收集数据
        for t in range(1, self.max_seq_length):
            # 当前状态
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            
            # Actor预测
            action_dist = self.actor(state)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            # Critic预测
            value = self.critic(state).squeeze(-1)
            
            # 更新轨迹
            trajectory[:, t] = action
            
            # 计算奖励
            reward = self.trajectory_env.compute_reward(
                trajectory, start_pose, end_pose, t, target_trajectory
            )
            
            # 下一状态价值
            if t < self.max_seq_length - 1:
                next_state = self._construct_state(start_pose, end_pose, trajectory, t)
                next_value = self.critic(next_state).squeeze(-1)
            else:
                next_value = torch.zeros(batch_size, device=device)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            next_values.append(next_value)
        
        # 强制终点约束
        trajectory[:, -1] = end_pose
        
        # 转换为张量
        log_probs = torch.stack(log_probs, dim=1)
        rewards = torch.stack(rewards, dim=1)
        values = torch.stack(values, dim=1)
        next_values = torch.stack(next_values, dim=1)
        
        return trajectory, log_probs, rewards, values, next_values
    
    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算折扣回报
        """
        batch_size, seq_length = rewards.shape
        returns = torch.zeros_like(rewards)
        
        running_return = torch.zeros(batch_size, device=rewards.device)
        for t in reversed(range(seq_length)):
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return
        
        return returns
    
    def _compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor,
                               next_values: torch.Tensor) -> torch.Tensor:
        """
        计算广义优势估计(GAE)
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
    
    def _compute_entropy_loss(self, trajectories: torch.Tensor, start_pose: torch.Tensor,
                             end_pose: torch.Tensor) -> torch.Tensor:
        """
        计算熵损失
        """
        total_entropy = 0.0
        count = 0
        
        for t in range(1, self.max_seq_length):
            state = self._construct_state(start_pose, end_pose, trajectories, t-1)
            action_dist = self.actor(state)
            entropy = action_dist.entropy().sum(dim=-1).mean()
            total_entropy += entropy
            count += 1
        
        return total_entropy / count if count > 0 else torch.tensor(0.0)
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
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
            'value_coeff': self.value_coeff,
            'use_gae': self.use_gae,
            'gae_lambda': self.gae_lambda,
            'use_separate_optimizers': self.use_separate_optimizers,
            'model_category': 'RL-based Methods'
        })
        return info


class ActorNetwork(nn.Module):
    """
    Actor网络 - 策略网络
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
        
        # 输出层
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
        features = self.backbone(state)
        
        mean = self.mean_head(features)
        logstd = self.logstd_head(features)
        
        # 限制标准差范围
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        
        return torch.distributions.Normal(mean, std)


class CriticNetwork(nn.Module):
    """
    Critic网络 - 价值网络
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
    """
    
    def __init__(self, input_dim: int, output_dim: int, max_seq_length: int, reward_type: str = 'dense'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        self.reward_type = reward_type
    
    def compute_reward(self, trajectory: torch.Tensor, start_pose: torch.Tensor,
                      end_pose: torch.Tensor, current_step: int,
                      target_trajectory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算奖励
        """
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        current_pos = trajectory[:, current_step]
        
        if self.reward_type == 'sparse':
            # 稀疏奖励
            if current_step == self.max_seq_length - 1:
                distance_to_goal = torch.norm(current_pos - end_pose, dim=-1)
                reward = -distance_to_goal
            else:
                reward = torch.zeros(batch_size, device=device)
        
        elif self.reward_type == 'dense':
            # 密集奖励
            # 1. 目标导向奖励
            distance_to_goal = torch.norm(current_pos - end_pose, dim=-1)
            progress_reward = -distance_to_goal
            
            # 2. 平滑性奖励
            if current_step > 1:
                velocity_curr = trajectory[:, current_step] - trajectory[:, current_step-1]
                velocity_prev = trajectory[:, current_step-1] - trajectory[:, current_step-2]
                acceleration = velocity_curr - velocity_prev
                smoothness_reward = -torch.norm(acceleration, dim=-1) * 0.1
            else:
                smoothness_reward = torch.zeros(batch_size, device=device)
            
            # 3. 进度奖励
            if current_step > 0:
                prev_distance = torch.norm(trajectory[:, current_step-1] - end_pose, dim=-1)
                curr_distance = torch.norm(current_pos - end_pose, dim=-1)
                progress_made = prev_distance - curr_distance
                progress_bonus = progress_made * 0.5
            else:
                progress_bonus = torch.zeros(batch_size, device=device)
            
            # 4. 时间惩罚（鼓励效率）
            time_penalty = -0.01
            
            reward = progress_reward + smoothness_reward + progress_bonus + time_penalty
        
        elif self.reward_type == 'shaped':
            # 形状奖励（更复杂的奖励设计）
            # 1. 距离奖励
            distance_to_goal = torch.norm(current_pos - end_pose, dim=-1)
            distance_reward = -distance_to_goal
            
            # 2. 方向奖励
            if current_step > 0:
                movement = current_pos - trajectory[:, current_step-1]
                desired_direction = end_pose - trajectory[:, current_step-1]
                
                # 归一化
                movement_norm = torch.norm(movement, dim=-1, keepdim=True)
                desired_norm = torch.norm(desired_direction, dim=-1, keepdim=True)
                
                # 避免除零
                movement_norm = torch.clamp(movement_norm, min=1e-8)
                desired_norm = torch.clamp(desired_norm, min=1e-8)
                
                movement_unit = movement / movement_norm
                desired_unit = desired_direction / desired_norm
                
                # 余弦相似度
                direction_similarity = torch.sum(movement_unit * desired_unit, dim=-1)
                direction_reward = direction_similarity * 0.5
            else:
                direction_reward = torch.zeros(batch_size, device=device)
            
            # 3. 速度奖励（鼓励适中的速度）
            if current_step > 0:
                velocity = torch.norm(trajectory[:, current_step] - trajectory[:, current_step-1], dim=-1)
                optimal_velocity = distance_to_goal / (self.max_seq_length - current_step + 1e-8)
                velocity_diff = torch.abs(velocity - optimal_velocity)
                velocity_reward = -velocity_diff * 0.1
            else:
                velocity_reward = torch.zeros(batch_size, device=device)
            
            reward = distance_reward + direction_reward + velocity_reward
        
        else:
            reward = torch.zeros(batch_size, device=device)
        
        return reward


class AdvantageActorCritic(ActorCriticTrajectoryModel):
    """
    优势Actor-Critic (A2C)
    包含更多技巧和改进
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_reward_normalization = config.get('use_reward_normalization', True)
        self.use_value_normalization = config.get('use_value_normalization', True)
        self.clip_rewards = config.get('clip_rewards', False)
        self.reward_clip_value = config.get('reward_clip_value', 10.0)
        
        # 统计信息
        self.reward_rms = RunningMeanStd()
        self.value_rms = RunningMeanStd()
        
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        奖励归一化
        """
        if self.use_reward_normalization:
            self.reward_rms.update(rewards.cpu().numpy())
            normalized_rewards = rewards / (self.reward_rms.std + 1e-8)
            
            if self.clip_rewards:
                normalized_rewards = torch.clamp(
                    normalized_rewards, 
                    -self.reward_clip_value, 
                    self.reward_clip_value
                )
            
            return normalized_rewards
        else:
            return rewards
    
    def _normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """
        价值归一化
        """
        if self.use_value_normalization:
            self.value_rms.update(values.detach().cpu().numpy())
            normalized_values = values / (self.value_rms.std + 1e-8)
            return normalized_values
        else:
            return values


class RunningMeanStd:
    """
    运行时均值和标准差统计
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


class AsynchronousActorCritic(ActorCriticTrajectoryModel):
    """
    异步Actor-Critic (A3C风格)
    支持多个并行环境
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_envs = config.get('num_envs', 4)
        self.rollout_length = config.get('rollout_length', 20)
        
        # 创建多个环境
        self.envs = [
            TrajectoryEnvironment(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                max_seq_length=self.max_seq_length,
                reward_type=self.reward_type
            ) for _ in range(self.num_envs)
        ]
        
    def collect_rollouts(self, start_poses: torch.Tensor, end_poses: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        收集多个环境的rollouts
        
        Args:
            start_poses: 起始位姿 [num_envs, input_dim]
            end_poses: 终止位姿 [num_envs, input_dim]
            
        Returns:
            rollout数据字典
        """
        device = start_poses.device
        
        # 初始化存储
        trajectories = torch.zeros(self.num_envs, self.rollout_length, self.output_dim, device=device)
        log_probs = torch.zeros(self.num_envs, self.rollout_length, device=device)
        rewards = torch.zeros(self.num_envs, self.rollout_length, device=device)
        values = torch.zeros(self.num_envs, self.rollout_length, device=device)
        dones = torch.zeros(self.num_envs, self.rollout_length, dtype=torch.bool, device=device)
        
        # 当前轨迹状态
        current_trajectories = torch.zeros(self.num_envs, self.max_seq_length, self.output_dim, device=device)
        current_trajectories[:, 0] = start_poses
        current_steps = torch.ones(self.num_envs, dtype=torch.long, device=device)
        
        # 收集rollouts
        for t in range(self.rollout_length):
            # 构建状态
            states = []
            for env_idx in range(self.num_envs):
                step = current_steps[env_idx].item()
                if step < self.max_seq_length:
                    state = self._construct_state(
                        start_poses[env_idx:env_idx+1], 
                        end_poses[env_idx:env_idx+1],
                        current_trajectories[env_idx:env_idx+1], 
                        step - 1
                    )
                    states.append(state)
                else:
                    # Episode结束，使用零状态
                    states.append(torch.zeros(1, self.state_dim, device=device))
            
            states = torch.cat(states, dim=0)
            
            # Actor和Critic预测
            action_dists = self.actor(states)
            actions = action_dists.sample()
            log_prob = action_dists.log_prob(actions).sum(dim=-1)
            value = self.critic(states).squeeze(-1)
            
            # 环境步骤
            for env_idx in range(self.num_envs):
                step = current_steps[env_idx].item()
                
                if step < self.max_seq_length:
                    # 更新轨迹
                    current_trajectories[env_idx, step] = actions[env_idx]
                    
                    # 计算奖励
                    reward = self.envs[env_idx].compute_reward(
                        current_trajectories[env_idx:env_idx+1],
                        start_poses[env_idx:env_idx+1],
                        end_poses[env_idx:env_idx+1],
                        step
                    )
                    
                    # 存储数据
                    trajectories[env_idx, t] = actions[env_idx]
                    rewards[env_idx, t] = reward
                    
                    # 更新步骤
                    current_steps[env_idx] += 1
                    
                    # 检查是否结束
                    if step == self.max_seq_length - 1:
                        dones[env_idx, t] = True
                        # 重置环境
                        current_steps[env_idx] = 1
                        current_trajectories[env_idx, 0] = start_poses[env_idx]
                else:
                    dones[env_idx, t] = True
            
            log_probs[:, t] = log_prob
            values[:, t] = value
        
        return {
            'trajectories': trajectories,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones
        }