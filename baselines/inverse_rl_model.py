"""
Inverse Reinforcement Learning Model for Trajectory Generation
逆强化学习轨迹生成模型
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


class InverseRLTrajectoryModel(RLBasedModel):
    """
    基于逆强化学习的轨迹生成模型
    学习隐含的奖励函数，然后优化策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.irl_method = config.get('irl_method', 'maxent')  # 'maxent', 'gail', 'airl', 'valueice'
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.reward_lr = config.get('reward_lr', 1e-3)
        self.gamma = config.get('gamma', 0.99)
        self.temperature = config.get('temperature', 1.0)  # MaxEnt IRL温度参数
        
        # 奖励网络
        self.reward_net = RewardNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('reward_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # 策略网络
        self.policy_net = IRLPolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('policy_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # 价值网络（用于某些IRL方法）
        if self.irl_method in ['airl', 'valueice']:
            self.value_net = ValueNetwork(
                state_dim=self.state_dim,
                hidden_dims=config.get('value_hidden_dims', [256, 256]),
                activation=config.get('activation', 'relu'),
                dropout=self.dropout
            )
            
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(),
                lr=self.learning_rate
            )
        
        # 优化器
        self.reward_optimizer = torch.optim.Adam(
            self.reward_net.parameters(),
            lr=self.reward_lr
        )
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        
        # 轨迹环境
        self.trajectory_env = TrajectoryEnvironment(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_seq_length=self.max_seq_length
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - IRL策略生成轨迹
        
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
            
            # 策略预测
            action_dist = self.policy_net(state)
            
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
            # 累积距离
            total_distance = torch.norm(trajectory[:, current_step] - trajectory[:, 0], dim=-1, keepdim=True)
        else:
            recent_velocity = torch.zeros_like(current_pos)
            total_distance = torch.zeros(batch_size, 1, device=device)
        
        # 组合状态
        state = torch.cat([
            current_pos,        # 当前位置
            target_pos,         # 目标位置
            relative_pos,       # 相对位置
            distance_to_goal,   # 到目标的距离
            time_info,          # 当前时间步比例
            remaining_time,     # 剩余时间比例
            recent_velocity,    # 最近速度
            total_distance      # 累积距离
        ], dim=-1)
        
        return state
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练步骤 - 根据不同的IRL方法
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        if self.irl_method == 'maxent':
            return self._maxent_irl_step(batch)
        elif self.irl_method == 'airl':
            return self._airl_step(batch)
        elif self.irl_method == 'valueice':
            return self._valueice_step(batch)
        else:
            raise ValueError(f"Unknown IRL method: {self.irl_method}")
    
    def _maxent_irl_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Maximum Entropy IRL训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch['trajectory']
        
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 收集专家数据的特征期望
        expert_features = self._compute_trajectory_features(start_pose, end_pose, expert_trajectory)
        
        # 生成策略轨迹
        policy_trajectory = self.forward(start_pose, end_pose)
        
        # 收集策略数据的特征期望
        policy_features = self._compute_trajectory_features(start_pose, end_pose, policy_trajectory)
        
        # MaxEnt IRL损失
        # 奖励网络损失：最大化专家轨迹奖励，最小化策略轨迹奖励
        expert_rewards = self._compute_trajectory_rewards(start_pose, end_pose, expert_trajectory)
        policy_rewards = self._compute_trajectory_rewards(start_pose, end_pose, policy_trajectory.detach())
        
        reward_loss = -torch.mean(expert_rewards) + torch.mean(policy_rewards)
        
        # 更新奖励网络
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 0.5)
        self.reward_optimizer.step()
        
        # 策略网络损失：最大化学习到的奖励
        policy_rewards_for_policy = self._compute_trajectory_rewards(start_pose, end_pose, policy_trajectory)
        policy_loss = -torch.mean(policy_rewards_for_policy)
        
        # 熵正则化
        entropy_loss = -self._compute_policy_entropy(start_pose, end_pose, policy_trajectory)
        total_policy_loss = policy_loss + self.temperature * entropy_loss
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        return {
            'reward_loss': reward_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'expert_reward_mean': expert_rewards.mean().item(),
            'policy_reward_mean': policy_rewards.mean().item(),
            'total_loss': total_policy_loss.item()
        }
    
    def _airl_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Adversarial IRL训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch['trajectory']
        
        # 生成策略轨迹
        policy_trajectory = self.forward(start_pose, end_pose)
        
        # 收集状态-动作对
        expert_states, expert_actions, expert_next_states = self._collect_state_action_pairs(
            start_pose, end_pose, expert_trajectory
        )
        policy_states, policy_actions, policy_next_states = self._collect_state_action_pairs(
            start_pose, end_pose, policy_trajectory
        )
        
        # AIRL判别器
        expert_rewards = self.reward_net(expert_states, expert_actions)
        policy_rewards = self.reward_net(policy_states, policy_actions.detach())
        
        # 价值函数
        expert_values = self.value_net(expert_states)
        expert_next_values = self.value_net(expert_next_states)
        policy_values = self.value_net(policy_states)
        policy_next_values = self.value_net(policy_next_states)
        
        # AIRL判别器logits
        expert_logits = expert_rewards + self.gamma * expert_next_values - expert_values
        policy_logits = policy_rewards + self.gamma * policy_next_values - policy_values
        
        # 判别器损失
        expert_labels = torch.ones_like(expert_logits)
        policy_labels = torch.zeros_like(policy_logits)
        
        discriminator_loss = (
            nn.BCEWithLogitsLoss()(expert_logits, expert_labels) +
            nn.BCEWithLogitsLoss()(policy_logits, policy_labels)
        ) / 2
        
        # 更新奖励和价值网络
        reward_value_loss = discriminator_loss
        
        self.reward_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        reward_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.reward_optimizer.step()
        self.value_optimizer.step()
        
        # 策略损失
        policy_rewards_for_generator = self.reward_net(policy_states, policy_actions)
        policy_values_for_generator = self.value_net(policy_states)
        policy_next_values_for_generator = self.value_net(policy_next_states)
        
        policy_logits_for_generator = (
            policy_rewards_for_generator + 
            self.gamma * policy_next_values_for_generator - 
            policy_values_for_generator
        )
        
        policy_loss = nn.BCEWithLogitsLoss()(
            policy_logits_for_generator,
            torch.ones_like(policy_logits_for_generator)
        )
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 计算准确率
        expert_pred = torch.sigmoid(expert_logits) > 0.5
        policy_pred = torch.sigmoid(policy_logits) <= 0.5
        discriminator_acc = (expert_pred.float().mean() + policy_pred.float().mean()) / 2
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'policy_loss': policy_loss.item(),
            'discriminator_acc': discriminator_acc.item(),
            'expert_reward_mean': expert_rewards.mean().item(),
            'policy_reward_mean': policy_rewards.mean().item(),
            'total_loss': policy_loss.item()
        }
    
    def _valueice_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        ValueDice IRL训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch['trajectory']
        
        # 生成策略轨迹
        policy_trajectory = self.forward(start_pose, end_pose)
        
        # 收集状态-动作对
        expert_states, expert_actions = self._collect_sa_pairs(start_pose, end_pose, expert_trajectory)
        policy_states, policy_actions = self._collect_sa_pairs(start_pose, end_pose, policy_trajectory)
        
        # ValueDice损失
        expert_nu = self.value_net(expert_states)
        policy_nu = self.value_net(policy_states)
        
        # Nu网络损失
        nu_loss = -torch.mean(expert_nu) + torch.mean(torch.exp(policy_nu.detach() - 1))
        
        # 更新Nu网络
        self.value_optimizer.zero_grad()
        nu_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        # 策略损失
        policy_nu_for_generator = self.value_net(policy_states)
        policy_loss = -torch.mean(policy_nu_for_generator)
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        return {
            'nu_loss': nu_loss.item(),
            'policy_loss': policy_loss.item(),
            'expert_nu_mean': expert_nu.mean().item(),
            'policy_nu_mean': policy_nu.mean().item(),
            'total_loss': policy_loss.item()
        }
    
    def _compute_trajectory_features(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                                   trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算轨迹特征
        """
        features = []
        
        for t in range(1, self.max_seq_length):
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            action = trajectory[:, t]
            
            # 可以定义各种特征
            # 1. 状态特征
            features.append(state.mean(dim=0))
            
            # 2. 动作特征
            features.append(action.mean(dim=0))
            
            # 3. 状态-动作交互特征
            sa_interaction = torch.mean(state.unsqueeze(-1) * action.unsqueeze(-2), dim=0).flatten()
            features.append(sa_interaction)
        
        return torch.cat(features)
    
    def _compute_trajectory_rewards(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                                   trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算轨迹的总奖励
        """
        total_rewards = []
        
        for b in range(trajectory.shape[0]):
            trajectory_reward = 0.0
            
            for t in range(1, self.max_seq_length):
                state = self._construct_state(
                    start_pose[b:b+1], end_pose[b:b+1], trajectory[b:b+1], t-1
                )
                action = trajectory[b:b+1, t]
                
                reward = self.reward_net(state, action)
                trajectory_reward += (self.gamma ** (t-1)) * reward
            
            total_rewards.append(trajectory_reward)
        
        return torch.cat(total_rewards)
    
    def _compute_policy_entropy(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                               trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算策略熵
        """
        total_entropy = 0.0
        count = 0
        
        for t in range(1, self.max_seq_length):
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            action_dist = self.policy_net(state)
            entropy = action_dist.entropy().sum(dim=-1).mean()
            total_entropy += entropy
            count += 1
        
        return total_entropy / count if count > 0 else torch.tensor(0.0)
    
    def _collect_state_action_pairs(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                                   trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        收集状态-动作对和下一状态
        """
        states = []
        actions = []
        next_states = []
        
        for t in range(1, self.max_seq_length - 1):
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            action = trajectory[:, t]
            next_state = self._construct_state(start_pose, end_pose, trajectory, t)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
        
        states = torch.stack(states, dim=1).view(-1, self.state_dim)
        actions = torch.stack(actions, dim=1).view(-1, self.action_dim)
        next_states = torch.stack(next_states, dim=1).view(-1, self.state_dim)
        
        return states, actions, next_states
    
    def _collect_sa_pairs(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                         trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        收集状态-动作对
        """
        states = []
        actions = []
        
        for t in range(1, self.max_seq_length):
            state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            action = trajectory[:, t]
            
            states.append(state)
            actions.append(action)
        
        states = torch.stack(states, dim=1).view(-1, self.state_dim)
        actions = torch.stack(actions, dim=1).view(-1, self.action_dim)
        
        return states, actions
    
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
            'irl_method': self.irl_method,
            'learning_rate': self.learning_rate,
            'reward_lr': self.reward_lr,
            'gamma': self.gamma,
            'temperature': self.temperature,
            'model_category': 'RL-based Methods'
        })
        return info


class RewardNetwork(nn.Module):
    """
    奖励网络
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
        input_dim = state_dim + action_dim
        
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
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            action: 动作 [batch_size, action_dim]
            
        Returns:
            reward: 奖励 [batch_size, 1]
        """
        state_action = torch.cat([state, action], dim=-1)
        return self.network(state_action)


class IRLPolicyNetwork(nn.Module):
    """
    IRL策略网络
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
        features = self.backbone(state)
        
        mean = self.mean_head(features)
        logstd = self.logstd_head(features)
        
        # 限制标准差范围
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        
        return torch.distributions.Normal(mean, std)


class ValueNetwork(nn.Module):
    """
    价值网络
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
    
    def __init__(self, input_dim: int, output_dim: int, max_seq_length: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
    
    def compute_features(self, trajectory: torch.Tensor, start_pose: torch.Tensor,
                        end_pose: torch.Tensor) -> torch.Tensor:
        """
        计算轨迹特征
        """
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        features = []
        
        # 1. 总距离特征
        total_distance = 0.0
        for t in range(1, self.max_seq_length):
            distance = torch.norm(trajectory[:, t] - trajectory[:, t-1], dim=-1)
            total_distance += distance
        features.append(total_distance.unsqueeze(-1))
        
        # 2. 平滑度特征
        if self.max_seq_length > 2:
            accelerations = []
            for t in range(2, self.max_seq_length):
                velocity_curr = trajectory[:, t] - trajectory[:, t-1]
                velocity_prev = trajectory[:, t-1] - trajectory[:, t-2]
                acceleration = velocity_curr - velocity_prev
                accelerations.append(torch.norm(acceleration, dim=-1))
            
            avg_acceleration = torch.stack(accelerations, dim=1).mean(dim=1).unsqueeze(-1)
            features.append(avg_acceleration)
        else:
            features.append(torch.zeros(batch_size, 1, device=device))
        
        # 3. 目标导向特征
        final_distance = torch.norm(trajectory[:, -1] - end_pose, dim=-1).unsqueeze(-1)
        features.append(final_distance)
        
        # 4. 效率特征
        direct_distance = torch.norm(end_pose - start_pose, dim=-1).unsqueeze(-1)
        efficiency = direct_distance / (total_distance + 1e-8)
        features.append(efficiency)
        
        return torch.cat(features, dim=-1)


class LinearFeatureIRL(InverseRLTrajectoryModel):
    """
    线性特征IRL
    使用线性奖励函数
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feature_dim = config.get('feature_dim', 10)
        
        # 线性奖励权重
        self.reward_weights = nn.Parameter(torch.randn(self.feature_dim))
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            feature_dim=self.feature_dim
        )
    
    def compute_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算线性奖励
        """
        features = self.feature_extractor(state, action)
        reward = torch.sum(features * self.reward_weights, dim=-1, keepdim=True)
        return reward


class FeatureExtractor(nn.Module):
    """
    特征提取器
    """
    
    def __init__(self, state_dim: int, action_dim: int, feature_dim: int):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        """
        state_action = torch.cat([state, action], dim=-1)
        return self.feature_net(state_action)


class DeepMaxEntIRL(InverseRLTrajectoryModel):
    """
    深度最大熵IRL
    使用深度网络的最大熵IRL
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_state_only_reward = config.get('use_state_only_reward', False)
        
        if self.use_state_only_reward:
            # 仅基于状态的奖励
            self.reward_net = StateOnlyRewardNetwork(
                state_dim=self.state_dim,
                hidden_dims=config.get('reward_hidden_dims', [256, 256]),
                activation=config.get('activation', 'relu'),
                dropout=self.dropout
            )
    
    def _compute_state_visitation_frequency(self, trajectory: torch.Tensor,
                                           start_pose: torch.Tensor, 
                                           end_pose: torch.Tensor) -> torch.Tensor:
        """
        计算状态访问频率
        """
        state_features = []
        
        for t in range(self.max_seq_length):
            if t == 0:
                state = self._construct_initial_state(start_pose, end_pose)
            else:
                state = self._construct_state(start_pose, end_pose, trajectory, t-1)
            
            state_features.append(state)
        
        return torch.stack(state_features, dim=1)  # [batch_size, seq_length, state_dim]
    
    def _construct_initial_state(self, start_pose: torch.Tensor, end_pose: torch.Tensor) -> torch.Tensor:
        """
        构建初始状态
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        current_pos = start_pose
        target_pos = end_pose
        relative_pos = target_pos - current_pos
        distance_to_goal = torch.norm(relative_pos, dim=-1, keepdim=True)
        time_info = torch.zeros(batch_size, 1, device=device)
        remaining_time = torch.ones(batch_size, 1, device=device)
        recent_velocity = torch.zeros_like(current_pos)
        total_distance = torch.zeros(batch_size, 1, device=device)
        
        state = torch.cat([
            current_pos, target_pos, relative_pos, distance_to_goal,
            time_info, remaining_time, recent_velocity, total_distance
        ], dim=-1)
        
        return state


class StateOnlyRewardNetwork(nn.Module):
    """
    仅基于状态的奖励网络
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        
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
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播（忽略动作）
        """
        return self.network(state)