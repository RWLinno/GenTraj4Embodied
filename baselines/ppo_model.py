"""
Proximal Policy Optimization (PPO) Model for Trajectory Generation
PPO轨迹生成模型
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


class PPOTrajectoryModel(RLBasedModel):
    """
    基于PPO的轨迹生成模型
    使用Proximal Policy Optimization算法
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.value_coeff = config.get('value_coeff', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.rollout_length = config.get('rollout_length', 2048)
        
        # Actor网络
        self.actor = PPOActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('actor_hidden_dims', [256, 256]),
            activation=config.get('activation', 'tanh'),
            dropout=self.dropout
        )
        
        # Critic网络
        self.critic = PPOCritic(
            state_dim=self.state_dim,
            hidden_dims=config.get('critic_hidden_dims', [256, 256]),
            activation=config.get('activation', 'tanh'),
            dropout=self.dropout
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # 轨迹环境
        self.trajectory_env = TrajectoryEnvironment(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_seq_length=self.max_seq_length,
            reward_type=self.reward_type
        )
        
        # 经验缓冲区
        self.rollout_buffer = PPORolloutBuffer(
            buffer_size=self.rollout_length,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - PPO策略生成轨迹
        
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
            
            # Actor预测
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
        remaining_steps = torch.ones(batch_size, 1, device=device) * ((self.max_seq_length - current_step - 1) / self.max_seq_length)
        
        # 历史信息
        if current_step > 0:
            recent_velocity = trajectory[:, current_step] - trajectory[:, max(0, current_step-1)]
            # 平均速度
            total_distance = torch.norm(trajectory[:, current_step] - trajectory[:, 0], dim=-1, keepdim=True)
            avg_speed = total_distance / (current_step + 1e-8)
        else:
            recent_velocity = torch.zeros_like(current_pos)
            avg_speed = torch.zeros(batch_size, 1, device=device)
        
        # 方向信息
        if torch.norm(relative_pos, dim=-1).max() > 1e-6:
            direction_to_goal = relative_pos / (torch.norm(relative_pos, dim=-1, keepdim=True) + 1e-8)
        else:
            direction_to_goal = torch.zeros_like(relative_pos)
        
        # 组合状态
        state = torch.cat([
            current_pos,        # 当前位置
            target_pos,         # 目标位置
            relative_pos,       # 相对位置
            distance_to_goal,   # 到目标的距离
            direction_to_goal,  # 到目标的方向
            time_info,          # 当前时间步比例
            remaining_steps,    # 剩余步数比例
            recent_velocity,    # 最近速度
            avg_speed          # 平均速度
        ], dim=-1)
        
        return state
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        PPO训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        target_trajectory = batch.get('trajectory', None)
        
        # 收集rollout数据
        rollout_data = self._collect_rollout(start_pose, end_pose, target_trajectory)
        
        # PPO更新
        total_losses = []
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(rollout_data['states']))
            
            # 分批训练
            for start_idx in range(0, len(indices), self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # 获取mini-batch数据
                mini_batch = {
                    'states': rollout_data['states'][batch_indices],
                    'actions': rollout_data['actions'][batch_indices],
                    'old_log_probs': rollout_data['log_probs'][batch_indices],
                    'advantages': rollout_data['advantages'][batch_indices],
                    'returns': rollout_data['returns'][batch_indices],
                    'old_values': rollout_data['values'][batch_indices]
                }
                
                # PPO损失计算和更新
                loss_dict = self._ppo_update_step(mini_batch)
                total_losses.append(loss_dict)
        
        # 平均损失
        avg_losses = {}
        for key in total_losses[0].keys():
            avg_losses[key] = sum(loss[key] for loss in total_losses) / len(total_losses)
        
        return avg_losses
    
    def _collect_rollout(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                        target_trajectory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        收集rollout数据
        
        Returns:
            rollout数据字典
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 存储rollout数据
        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
        
        # 初始化轨迹
        trajectory = torch.zeros(batch_size, self.max_seq_length, self.output_dim, device=device)
        trajectory[:, 0] = start_pose
        
        # 收集轨迹数据
        for t in range(1, self.max_seq_length):
            # 构建状态
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            
            # Actor和Critic预测
            action_dist = self.actor(state)
            value = self.critic(state)
            
            # 采样动作
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            # 更新轨迹
            trajectory[:, t] = action
            
            # 计算奖励
            reward = self.trajectory_env.compute_reward(
                trajectory, start_pose, end_pose, t, target_trajectory
            )
            
            # 检查是否结束
            done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if t == self.max_seq_length - 1:
                done = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            # 存储数据
            states_list.append(state)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            values_list.append(value.squeeze(-1))
            dones_list.append(done)
        
        # 强制终点约束
        trajectory[:, -1] = end_pose
        
        # 转换为张量
        states = torch.stack(states_list, dim=1).view(-1, self.state_dim)
        actions = torch.stack(actions_list, dim=1).view(-1, self.action_dim)
        log_probs = torch.stack(log_probs_list, dim=1).view(-1)
        rewards = torch.stack(rewards_list, dim=1).view(-1)
        values = torch.stack(values_list, dim=1).view(-1)
        dones = torch.stack(dones_list, dim=1).view(-1)
        
        # 计算优势和回报
        advantages, returns = self._compute_gae_advantages_and_returns(
            rewards.view(batch_size, -1),
            values.view(batch_size, -1),
            dones.view(batch_size, -1)
        )
        
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'advantages': advantages,
            'returns': returns
        }
    
    def _compute_gae_advantages_and_returns(self, rewards: torch.Tensor, values: torch.Tensor,
                                          dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势和回报
        
        Args:
            rewards: 奖励 [batch_size, seq_length]
            values: 价值 [batch_size, seq_length]
            dones: 结束标志 [batch_size, seq_length]
            
        Returns:
            advantages: GAE优势 [batch_size, seq_length]
            returns: 回报 [batch_size, seq_length]
        """
        batch_size, seq_length = rewards.shape
        device = rewards.device
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # 计算每个序列的GAE
        for b in range(batch_size):
            gae = 0
            next_value = 0  # 假设episode结束时价值为0
            
            for t in reversed(range(seq_length)):
                if t == seq_length - 1:
                    next_non_terminal = 1.0 - dones[b, t].float()
                    next_value = 0
                else:
                    next_non_terminal = 1.0 - dones[b, t].float()
                    next_value = values[b, t + 1]
                
                delta = rewards[b, t] + self.gamma * next_value * next_non_terminal - values[b, t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[b, t] = gae
                returns[b, t] = gae + values[b, t]
        
        return advantages, returns
    
    def _ppo_update_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        PPO更新步骤
        
        Args:
            batch: mini-batch数据
            
        Returns:
            损失字典
        """
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        old_values = batch['old_values']
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 当前策略预测
        action_dist = self.actor(states)
        new_log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)
        
        # 当前价值预测
        new_values = self.critic(states).squeeze(-1)
        
        # PPO策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        if self.clip_epsilon > 0:
            # 裁剪价值损失
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            value_losses = (new_values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        
        # 熵损失
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()
        
        # 计算额外统计信息
        with torch.no_grad():
            approx_kl = ((old_log_probs - new_log_probs) ** 2).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_fraction.item(),
            'entropy_mean': entropy.mean().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'returns_mean': returns.mean().item(),
            'values_mean': new_values.mean().item()
        }
    
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
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'entropy_coeff': self.entropy_coeff,
            'value_coeff': self.value_coeff,
            'ppo_epochs': self.ppo_epochs,
            'mini_batch_size': self.mini_batch_size,
            'rollout_length': self.rollout_length,
            'model_category': 'RL-based Methods'
        })
        return info


class PPOActor(nn.Module):
    """
    PPO Actor网络
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
                 activation: str = 'tanh', dropout: float = 0.0):
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
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 策略头使用较小的初始化
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.logstd_head.weight, gain=0.01)
        nn.init.constant_(self.logstd_head.bias, 0.0)
    
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


class PPOCritic(nn.Module):
    """
    PPO Critic网络
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256],
                 activation: str = 'tanh', dropout: float = 0.0):
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
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
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


class PPORolloutBuffer:
    """
    PPO Rollout缓冲区
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 缓冲区
        self.states = torch.zeros(buffer_size, state_dim)
        self.actions = torch.zeros(buffer_size, action_dim)
        self.log_probs = torch.zeros(buffer_size)
        self.rewards = torch.zeros(buffer_size)
        self.values = torch.zeros(buffer_size)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.ptr = 0
        self.full = False
    
    def add(self, state: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor,
            reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor):
        """
        添加经验到缓冲区
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        获取所有数据
        """
        size = self.buffer_size if self.full else self.ptr
        
        return {
            'states': self.states[:size],
            'actions': self.actions[:size],
            'log_probs': self.log_probs[:size],
            'rewards': self.rewards[:size],
            'values': self.values[:size],
            'dones': self.dones[:size]
        }
    
    def clear(self):
        """
        清空缓冲区
        """
        self.ptr = 0
        self.full = False


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
            progress_reward = -distance_to_goal * 0.1
            
            # 2. 进度奖励
            if current_step > 0:
                prev_distance = torch.norm(trajectory[:, current_step-1] - end_pose, dim=-1)
                curr_distance = distance_to_goal
                progress_made = prev_distance - curr_distance
                progress_bonus = progress_made * 1.0
            else:
                progress_bonus = torch.zeros(batch_size, device=device)
            
            # 3. 平滑性奖励
            if current_step > 1:
                velocity_curr = trajectory[:, current_step] - trajectory[:, current_step-1]
                velocity_prev = trajectory[:, current_step-1] - trajectory[:, current_step-2]
                acceleration = velocity_curr - velocity_prev
                smoothness_penalty = -torch.norm(acceleration, dim=-1) * 0.01
            else:
                smoothness_penalty = torch.zeros(batch_size, device=device)
            
            # 4. 完成奖励
            completion_bonus = torch.zeros(batch_size, device=device)
            if current_step == self.max_seq_length - 1:
                # 如果到达终点且距离很近，给予大奖励
                final_distance = torch.norm(current_pos - end_pose, dim=-1)
                completion_bonus = torch.where(final_distance < 0.1, 
                                             torch.tensor(10.0, device=device), 
                                             torch.tensor(0.0, device=device))
            
            # 5. 时间效率奖励
            efficiency_bonus = 0.01  # 鼓励更快完成
            
            reward = progress_reward + progress_bonus + smoothness_penalty + completion_bonus + efficiency_bonus
        
        elif self.reward_type == 'shaped':
            # 形状奖励
            distance_to_goal = torch.norm(current_pos - end_pose, dim=-1)
            
            # 距离奖励（指数衰减）
            distance_reward = torch.exp(-distance_to_goal)
            
            # 方向奖励
            if current_step > 0:
                movement = current_pos - trajectory[:, current_step-1]
                desired_direction = end_pose - trajectory[:, current_step-1]
                
                movement_norm = torch.norm(movement, dim=-1, keepdim=True)
                desired_norm = torch.norm(desired_direction, dim=-1, keepdim=True)
                
                movement_norm = torch.clamp(movement_norm, min=1e-8)
                desired_norm = torch.clamp(desired_norm, min=1e-8)
                
                movement_unit = movement / movement_norm
                desired_unit = desired_direction / desired_norm
                
                direction_similarity = torch.sum(movement_unit * desired_unit, dim=-1)
                direction_reward = torch.clamp(direction_similarity, 0, 1) * 0.5
            else:
                direction_reward = torch.zeros(batch_size, device=device)
            
            # 速度奖励
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


class PPO2(PPOTrajectoryModel):
    """
    PPO2变体
    包含一些改进和技巧
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_linear_lr_decay = config.get('use_linear_lr_decay', True)
        self.use_reward_normalization = config.get('use_reward_normalization', True)
        self.use_observation_normalization = config.get('use_observation_normalization', True)
        self.target_kl = config.get('target_kl', 0.01)
        
        # 统计信息
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(self.state_dim,))
        
        # 学习率调度器
        if self.use_linear_lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.get('total_training_steps', 1000000)
            )
    
    def _normalize_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """
        观测归一化
        """
        if self.use_observation_normalization:
            self.obs_rms.update(obs.cpu().numpy())
            normalized_obs = (obs - torch.tensor(self.obs_rms.mean, device=obs.device)) / torch.tensor(self.obs_rms.std + 1e-8, device=obs.device)
            return normalized_obs
        else:
            return obs
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        奖励归一化
        """
        if self.use_reward_normalization:
            self.reward_rms.update(rewards.cpu().numpy())
            normalized_rewards = rewards / (self.reward_rms.std + 1e-8)
            return normalized_rewards
        else:
            return rewards


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


class RecurrentPPO(PPOTrajectoryModel):
    """
    循环PPO
    使用LSTM的PPO变体
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 1)
        
        # 循环Actor
        self.actor = RecurrentPPOActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 循环Critic
        self.critic = RecurrentPPOCritic(
            state_dim=self.state_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 重新初始化优化器
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
            eps=1e-5
        )


class RecurrentPPOActor(nn.Module):
    """
    循环PPO Actor
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.logstd_head = nn.Linear(hidden_size, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        前向传播
        """
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # 添加序列维度
        
        lstm_out, new_hidden = self.lstm(state, hidden)
        features = lstm_out[:, -1]  # 取最后一个时间步
        
        mean = self.mean_head(features)
        logstd = self.logstd_head(features)
        
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        
        return torch.distributions.Normal(mean, std), new_hidden


class RecurrentPPOCritic(nn.Module):
    """
    循环PPO Critic
    """
    
    def __init__(self, state_dim: int, hidden_size: int = 256,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.value_head = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        前向传播
        """
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # 添加序列维度
        
        lstm_out, new_hidden = self.lstm(state, hidden)
        features = lstm_out[:, -1]  # 取最后一个时间步
        
        value = self.value_head(features)
        
        return value, new_hidden