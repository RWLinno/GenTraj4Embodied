"""
Imitation Learning Model for Trajectory Generation
模仿学习轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from .base_model import RLBasedModel


class ImitationLearningTrajectoryModel(RLBasedModel):
    """
    基于模仿学习的轨迹生成模型
    包含行为克隆(BC)、DAgger和GAIL等方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.imitation_method = config.get('imitation_method', 'behavioral_cloning')  # 'behavioral_cloning', 'dagger', 'gail'
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.beta_schedule = config.get('beta_schedule', 'linear')  # DAgger中的beta调度
        self.discriminator_lr = config.get('discriminator_lr', 1e-4)  # GAIL判别器学习率
        
        # 策略网络
        self.policy_net = ImitationPolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('policy_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout,
            output_type=config.get('output_type', 'deterministic')  # 'deterministic', 'stochastic'
        )
        
        # GAIL判别器网络
        if self.imitation_method == 'gail':
            self.discriminator = GAILDiscriminator(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=config.get('discriminator_hidden_dims', [256, 256]),
                activation=config.get('activation', 'relu'),
                dropout=self.dropout
            )
            
            self.discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.discriminator_lr
            )
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        
        # DAgger相关
        if self.imitation_method == 'dagger':
            self.dagger_iterations = 0
            self.max_dagger_iterations = config.get('max_dagger_iterations', 10)
            self.initial_beta = config.get('initial_beta', 1.0)
            self.final_beta = config.get('final_beta', 0.1)
        
        # 专家策略（用于DAgger）
        self.expert_policy = None  # 需要外部设置
        
    def set_expert_policy(self, expert_policy):
        """
        设置专家策略（用于DAgger）
        """
        self.expert_policy = expert_policy
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 模仿学习生成轨迹
        
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
            action = self.policy_net(state)
            
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
            # 轨迹历史（最近几步的平均位置）
            history_window = min(3, current_step + 1)
            recent_positions = trajectory[:, max(0, current_step-history_window+1):current_step+1]
            avg_recent_pos = torch.mean(recent_positions, dim=1)
        else:
            recent_velocity = torch.zeros_like(current_pos)
            avg_recent_pos = current_pos
        
        # 组合状态
        state = torch.cat([
            current_pos,        # 当前位置
            target_pos,         # 目标位置
            relative_pos,       # 相对位置
            distance_to_goal,   # 到目标的距离
            time_info,          # 当前时间步比例
            remaining_steps,    # 剩余步数比例
            recent_velocity,    # 最近速度
            avg_recent_pos      # 最近位置平均
        ], dim=-1)
        
        return state
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练步骤 - 根据不同的模仿学习方法
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        if self.imitation_method == 'behavioral_cloning':
            return self._behavioral_cloning_step(batch)
        elif self.imitation_method == 'dagger':
            return self._dagger_step(batch)
        elif self.imitation_method == 'gail':
            return self._gail_step(batch)
        else:
            raise ValueError(f"Unknown imitation method: {self.imitation_method}")
    
    def _behavioral_cloning_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        行为克隆训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch['trajectory']
        
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 收集状态-动作对
        states = []
        actions = []
        
        for t in range(1, self.max_seq_length):
            # 构建状态（基于专家轨迹）
            state = self._construct_state(start_pose, end_pose, expert_trajectory, t-1)
            action = expert_trajectory[:, t]
            
            states.append(state)
            actions.append(action)
        
        states = torch.stack(states, dim=1).view(-1, self.state_dim)  # [batch_size * (seq_length-1), state_dim]
        actions = torch.stack(actions, dim=1).view(-1, self.action_dim)  # [batch_size * (seq_length-1), action_dim]
        
        # 策略预测
        predicted_actions = self.policy_net(states)
        
        # 行为克隆损失
        if self.policy_net.output_type == 'deterministic':
            bc_loss = nn.MSELoss()(predicted_actions, actions)
        else:  # stochastic
            # 对于随机策略，使用负对数似然
            action_dist = predicted_actions
            bc_loss = -action_dist.log_prob(actions).sum(dim=-1).mean()
        
        # 反向传播
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        return {
            'bc_loss': bc_loss.item(),
            'total_loss': bc_loss.item()
        }
    
    def _dagger_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        DAgger训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch.get('trajectory', None)
        
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 计算当前beta值
        if self.beta_schedule == 'linear':
            beta = self.initial_beta - (self.initial_beta - self.final_beta) * (self.dagger_iterations / self.max_dagger_iterations)
        elif self.beta_schedule == 'exponential':
            beta = self.initial_beta * (self.final_beta / self.initial_beta) ** (self.dagger_iterations / self.max_dagger_iterations)
        else:
            beta = self.initial_beta
        
        beta = max(beta, self.final_beta)
        
        # 生成混合轨迹（策略 + 专家）
        mixed_trajectory = torch.zeros(batch_size, self.max_seq_length, self.output_dim, device=device)
        mixed_trajectory[:, 0] = start_pose
        
        states = []
        expert_actions = []
        
        for t in range(1, self.max_seq_length):
            # 构建状态
            state = self._construct_state(start_pose, end_pose, mixed_trajectory, t-1)
            
            # 策略预测
            policy_action = self.policy_net(state)
            
            # 专家动作（如果有专家策略）
            if self.expert_policy is not None:
                expert_action = self.expert_policy(state)
            elif expert_trajectory is not None:
                expert_action = expert_trajectory[:, t]
            else:
                expert_action = policy_action  # 退化为BC
            
            # 混合动作
            use_expert = torch.rand(batch_size, 1, device=device) < beta
            mixed_action = torch.where(use_expert, expert_action, policy_action)
            
            # 更新轨迹
            mixed_trajectory[:, t] = mixed_action
            
            # 收集训练数据
            states.append(state)
            expert_actions.append(expert_action)
        
        # 强制终点约束
        mixed_trajectory[:, -1] = end_pose
        
        # 训练策略网络
        states = torch.stack(states, dim=1).view(-1, self.state_dim)
        expert_actions = torch.stack(expert_actions, dim=1).view(-1, self.action_dim)
        
        predicted_actions = self.policy_net(states)
        
        if self.policy_net.output_type == 'deterministic':
            dagger_loss = nn.MSELoss()(predicted_actions, expert_actions)
        else:
            action_dist = predicted_actions
            dagger_loss = -action_dist.log_prob(expert_actions).sum(dim=-1).mean()
        
        # 反向传播
        self.policy_optimizer.zero_grad()
        dagger_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 更新DAgger迭代计数
        self.dagger_iterations += 1
        
        return {
            'dagger_loss': dagger_loss.item(),
            'beta': beta,
            'dagger_iteration': self.dagger_iterations,
            'total_loss': dagger_loss.item()
        }
    
    def _gail_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        GAIL训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch['trajectory']
        
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 生成策略轨迹
        policy_trajectory = self.forward(start_pose, end_pose)
        
        # 收集专家和策略的状态-动作对
        expert_states = []
        expert_actions = []
        policy_states = []
        policy_actions = []
        
        for t in range(1, self.max_seq_length):
            # 专家数据
            expert_state = self._construct_state(start_pose, end_pose, expert_trajectory, t-1)
            expert_action = expert_trajectory[:, t]
            expert_states.append(expert_state)
            expert_actions.append(expert_action)
            
            # 策略数据
            policy_state = self._construct_state(start_pose, end_pose, policy_trajectory, t-1)
            policy_action = policy_trajectory[:, t]
            policy_states.append(policy_state)
            policy_actions.append(policy_action)
        
        expert_states = torch.stack(expert_states, dim=1).view(-1, self.state_dim)
        expert_actions = torch.stack(expert_actions, dim=1).view(-1, self.action_dim)
        policy_states = torch.stack(policy_states, dim=1).view(-1, self.state_dim)
        policy_actions = torch.stack(policy_actions, dim=1).view(-1, self.action_dim)
        
        # 训练判别器
        expert_logits = self.discriminator(expert_states, expert_actions)
        policy_logits = self.discriminator(policy_states, policy_actions.detach())
        
        # 判别器损失（二分类）
        expert_labels = torch.ones_like(expert_logits)
        policy_labels = torch.zeros_like(policy_logits)
        
        discriminator_loss = (
            nn.BCEWithLogitsLoss()(expert_logits, expert_labels) +
            nn.BCEWithLogitsLoss()(policy_logits, policy_labels)
        ) / 2
        
        # 更新判别器
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        self.discriminator_optimizer.step()
        
        # 训练策略网络（生成器）
        policy_logits_for_generator = self.discriminator(policy_states, policy_actions)
        
        # 策略损失（欺骗判别器）
        generator_loss = nn.BCEWithLogitsLoss()(
            policy_logits_for_generator,
            torch.ones_like(policy_logits_for_generator)
        )
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 计算判别器准确率
        expert_pred = torch.sigmoid(expert_logits) > 0.5
        policy_pred = torch.sigmoid(policy_logits) <= 0.5
        discriminator_acc = (expert_pred.float().mean() + policy_pred.float().mean()) / 2
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'generator_loss': generator_loss.item(),
            'discriminator_acc': discriminator_acc.item(),
            'total_loss': generator_loss.item()
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
            'imitation_method': self.imitation_method,
            'learning_rate': self.learning_rate,
            'beta_schedule': self.beta_schedule,
            'model_category': 'RL-based Methods'
        })
        
        if self.imitation_method == 'dagger':
            info.update({
                'dagger_iterations': self.dagger_iterations,
                'max_dagger_iterations': self.max_dagger_iterations,
                'initial_beta': self.initial_beta,
                'final_beta': self.final_beta
            })
        elif self.imitation_method == 'gail':
            info.update({
                'discriminator_lr': self.discriminator_lr
            })
        
        return info


class ImitationPolicyNetwork(nn.Module):
    """
    模仿学习策略网络
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu', dropout: float = 0.0, output_type: str = 'deterministic'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_type = output_type
        
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
        if output_type == 'deterministic':
            self.output_layer = nn.Linear(input_dim, action_dim)
        else:  # stochastic
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
    
    def forward(self, state: torch.Tensor):
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            
        Returns:
            action or action_dist: 动作或动作分布
        """
        features = self.backbone(state)
        
        if self.output_type == 'deterministic':
            action = self.output_layer(features)
            return action
        else:  # stochastic
            mean = self.mean_head(features)
            logstd = self.logstd_head(features)
            
            # 限制标准差范围
            logstd = torch.clamp(logstd, -20, 2)
            std = torch.exp(logstd)
            
            return torch.distributions.Normal(mean, std)


class GAILDiscriminator(nn.Module):
    """
    GAIL判别器网络
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
        
        # 输出层（二分类）
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
            logits: 判别器输出 [batch_size, 1]
        """
        # 连接状态和动作
        state_action = torch.cat([state, action], dim=-1)
        
        # 判别器预测
        logits = self.network(state_action)
        
        return logits


class ValueDice(ImitationLearningTrajectoryModel):
    """
    ValueDice模仿学习方法
    基于价值函数的分布匹配
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.alpha = config.get('alpha', 0.5)  # 正则化参数
        self.nu_lr = config.get('nu_lr', 1e-3)  # nu网络学习率
        
        # Nu网络（密度比估计）
        self.nu_network = NuNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('nu_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        self.nu_optimizer = torch.optim.Adam(
            self.nu_network.parameters(),
            lr=self.nu_lr
        )
    
    def _valuedice_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        ValueDice训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch['trajectory']
        
        # 生成策略轨迹
        policy_trajectory = self.forward(start_pose, end_pose)
        
        # 收集状态-动作对
        expert_states, expert_actions = self._collect_state_action_pairs(
            start_pose, end_pose, expert_trajectory
        )
        policy_states, policy_actions = self._collect_state_action_pairs(
            start_pose, end_pose, policy_trajectory
        )
        
        # Nu网络预测
        expert_nu = self.nu_network(expert_states, expert_actions)
        policy_nu = self.nu_network(policy_states, policy_actions.detach())
        
        # ValueDice损失
        nu_loss = -torch.mean(expert_nu) + torch.mean(torch.exp(policy_nu - 1))
        
        # 更新Nu网络
        self.nu_optimizer.zero_grad()
        nu_loss.backward()
        self.nu_optimizer.step()
        
        # 策略损失
        policy_nu_for_generator = self.nu_network(policy_states, policy_actions)
        policy_loss = -torch.mean(policy_nu_for_generator)
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'nu_loss': nu_loss.item(),
            'policy_loss': policy_loss.item(),
            'total_loss': policy_loss.item()
        }
    
    def _collect_state_action_pairs(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
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


class NuNetwork(nn.Module):
    """
    Nu网络（密度比估计）
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
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
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        state_action = torch.cat([state, action], dim=-1)
        return self.network(state_action)


class SQIL(ImitationLearningTrajectoryModel):
    """
    Soft Q Imitation Learning
    基于软Q学习的模仿学习
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)  # 软更新参数
        
        # Q网络
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('q_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # 目标Q网络
        self.target_q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.get('q_hidden_dims', [256, 256]),
            activation=config.get('activation', 'relu'),
            dropout=self.dropout
        )
        
        # 初始化目标网络
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )
    
    def _sqil_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        SQIL训练步骤
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        expert_trajectory = batch['trajectory']
        
        # 生成策略轨迹
        policy_trajectory = self.forward(start_pose, end_pose)
        
        # 构建SQIL数据
        expert_states, expert_actions, expert_next_states = self._build_sqil_data(
            start_pose, end_pose, expert_trajectory, reward=1.0
        )
        policy_states, policy_actions, policy_next_states = self._build_sqil_data(
            start_pose, end_pose, policy_trajectory, reward=0.0
        )
        
        # 合并数据
        all_states = torch.cat([expert_states, policy_states], dim=0)
        all_actions = torch.cat([expert_actions, policy_actions], dim=0)
        all_next_states = torch.cat([expert_next_states, policy_next_states], dim=0)
        all_rewards = torch.cat([
            torch.ones(expert_states.shape[0], 1, device=expert_states.device),
            torch.zeros(policy_states.shape[0], 1, device=policy_states.device)
        ], dim=0)
        
        # Q学习更新
        current_q = self.q_network(all_states, all_actions)
        
        with torch.no_grad():
            # 使用策略网络选择下一个动作
            next_actions = self.policy_net(all_next_states)
            if hasattr(next_actions, 'sample'):  # 随机策略
                next_actions = next_actions.sample()
            
            target_q = self.target_q_network(all_next_states, next_actions)
            target_q_value = all_rewards + self.gamma * target_q
        
        q_loss = nn.MSELoss()(current_q, target_q_value)
        
        # 更新Q网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 策略损失（最大化Q值）
        policy_actions_for_q = self.policy_net(expert_states)
        if hasattr(policy_actions_for_q, 'sample'):
            policy_actions_for_q = policy_actions_for_q.rsample()  # 重参数化
        
        policy_q = self.q_network(expert_states, policy_actions_for_q)
        policy_loss = -policy_q.mean()
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target_network()
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'total_loss': policy_loss.item()
        }
    
    def _build_sqil_data(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                        trajectory: torch.Tensor, reward: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建SQIL训练数据
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
    
    def _soft_update_target_network(self):
        """
        软更新目标网络
        """
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class QNetwork(nn.Module):
    """
    Q网络
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
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
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        state_action = torch.cat([state, action], dim=-1)
        return self.network(state_action)