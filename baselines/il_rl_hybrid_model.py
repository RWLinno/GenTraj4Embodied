"""
Imitation Learning + Reinforcement Learning (IL+RL) Hybrid Model
模仿学习+强化学习混合模型

结合模仿学习和强化学习的混合策略，先通过模仿学习获得初始策略，
然后通过强化学习进一步优化策略性能。

Reference:
- Ho, J., & Ermon, S. "Generative adversarial imitation learning." 
  Advances in neural information processing systems 29 (2016).
- Schulman, J., et al. "Proximal policy optimization algorithms." 
  arXiv preprint arXiv:1707.06347 (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from collections import deque
import random

from ...base_model import BaseTrajectoryModel


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Discriminator(nn.Module):
    """判别器网络（用于GAIL）"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ILRLTrajectoryModel(BaseTrajectoryModel):
    """
    IL+RL混合轨迹生成模型
    
    该模型结合了模仿学习和强化学习的优势：
    1. 阶段1：使用GAIL进行模仿学习，从专家轨迹中学习初始策略
    2. 阶段2：使用PPO进行强化学习，进一步优化策略性能
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型参数
        self.input_dim = config.get('input_dim', 6)  # start + end pose
        self.output_dim = config.get('output_dim', 3)  # trajectory point
        self.hidden_dim = config.get('hidden_dim', 256)
        self.trajectory_length = config.get('trajectory_length', 50)
        
        # 训练参数
        self.il_epochs = config.get('il_epochs', 100)  # 模仿学习阶段轮数
        self.rl_epochs = config.get('rl_epochs', 200)  # 强化学习阶段轮数
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 3e-4)
        
        # PPO参数
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        
        # 网络初始化
        # 策略网络输入：start_pose + end_pose = input_dim * 2
        self.policy = PolicyNetwork(self.input_dim * 2, self.output_dim, self.hidden_dim)
        self.value_net = ValueNetwork(self.input_dim * 2, self.hidden_dim)
        self.discriminator = Discriminator(self.input_dim * 2 + self.output_dim, self.hidden_dim)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        # 经验回放缓冲区
        self.experience_buffer = deque(maxlen=10000)
        self.expert_trajectories = []
        
        # 训练阶段标识
        self.current_phase = 'IL'  # 'IL' or 'RL'
        self.il_completed = False
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def set_expert_trajectories(self, expert_trajectories: List[torch.Tensor]):
        """设置专家轨迹数据"""
        self.expert_trajectories = expert_trajectories
        
    def generate_expert_data(self, num_trajectories: int = 1000):
        """生成合成专家轨迹数据"""
        expert_data = []
        
        for _ in range(num_trajectories):
            # 随机生成起点和终点
            start_pose = torch.randn(3) * 0.5
            end_pose = torch.randn(3) * 0.5
            
            # 生成高质量轨迹（使用三次样条插值 + 噪声）
            t = torch.linspace(0, 1, self.trajectory_length)
            
            # 三次样条插值
            trajectory = []
            for i in range(3):  # x, y, z
                # 添加中间控制点
                mid_point = (start_pose[i] + end_pose[i]) / 2 + torch.randn(1) * 0.1
                
                # 三次插值
                a = 2 * start_pose[i] - 4 * mid_point + 2 * end_pose[i]
                b = -3 * start_pose[i] + 4 * mid_point - end_pose[i]
                c = 0
                d = start_pose[i]
                
                coord = a * t**3 + b * t**2 + c * t + d
                trajectory.append(coord)
            
            trajectory = torch.stack(trajectory, dim=1)  # [seq_len, 3]
            
            # 添加小量噪声使轨迹更自然
            trajectory += torch.randn_like(trajectory) * 0.02
            
            expert_data.append({
                'start_pose': start_pose,
                'end_pose': end_pose,
                'trajectory': trajectory
            })
        
        self.expert_trajectories = expert_data
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.policy(x)
    
    def generate_trajectory(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                          num_points: int = 50) -> torch.Tensor:
        """
        生成轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, 3] 或 [3]
            end_pose: 结束位姿 [batch_size, 3] 或 [3]
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 生成的轨迹 [batch_size, num_points, 3]
        """
        if start_pose.dim() == 1:
            start_pose = start_pose.unsqueeze(0)
        if end_pose.dim() == 1:
            end_pose = end_pose.unsqueeze(0)
        
        batch_size = start_pose.shape[0]
        trajectories = []
        
        self.policy.eval()
        with torch.no_grad():
            for b in range(batch_size):
                trajectory = []
                current_pose = start_pose[b]
                target_pose = end_pose[b]
                
                for t in range(num_points):
                    # 计算进度
                    progress = t / (num_points - 1)
                    
                    # 输入：当前位置、目标位置
                    policy_input = torch.cat([current_pose, target_pose]).unsqueeze(0)
                    
                    # 策略网络输出动作
                    action = self.policy(policy_input).squeeze(0)
                    
                    # 更新当前位置
                    if t == 0:
                        current_pose = start_pose[b]
                    elif t == num_points - 1:
                        current_pose = end_pose[b]
                    else:
                        # 线性插值 + 策略调整
                        linear_pose = start_pose[b] + progress * (end_pose[b] - start_pose[b])
                        current_pose = linear_pose + action * 0.1  # 策略微调
                    
                    trajectory.append(current_pose)
                
                trajectories.append(torch.stack(trajectory))
        
        return torch.stack(trajectories)
    
    def imitation_learning_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """模仿学习步骤（GAIL）"""
        if not self.expert_trajectories:
            self.generate_expert_data()
        
        # 采样专家数据
        expert_batch = random.sample(self.expert_trajectories, min(self.batch_size, len(self.expert_trajectories)))
        
        # 生成策略轨迹
        policy_trajectories = []
        for expert_data in expert_batch:
            start_pose = expert_data['start_pose']
            end_pose = expert_data['end_pose']
            traj = self.generate_trajectory(start_pose, end_pose, self.trajectory_length)
            policy_trajectories.append(traj.squeeze(0))
        
        # 准备判别器训练数据
        expert_states = []
        expert_actions = []
        policy_states = []
        policy_actions = []
        
        for i, expert_data in enumerate(expert_batch):
            expert_traj = expert_data['trajectory']
            policy_traj = policy_trajectories[i]
            
            for t in range(len(expert_traj) - 1):
                # 专家数据
                state = torch.cat([expert_data['start_pose'], expert_data['end_pose']])
                action = expert_traj[t + 1] - expert_traj[t]
                expert_states.append(state)
                expert_actions.append(action)
                
                # 策略数据
                policy_states.append(state)
                policy_action = policy_traj[t + 1] - policy_traj[t]
                policy_actions.append(policy_action)
        
        expert_states = torch.stack(expert_states)
        expert_actions = torch.stack(expert_actions)
        policy_states = torch.stack(policy_states)
        policy_actions = torch.stack(policy_actions)
        
        # 训练判别器
        expert_inputs = torch.cat([expert_states, expert_actions], dim=1)
        policy_inputs = torch.cat([policy_states, policy_actions], dim=1)
        
        expert_scores = self.discriminator(expert_inputs)
        policy_scores = self.discriminator(policy_inputs)
        
        # 判别器损失
        discriminator_loss = -torch.mean(torch.log(expert_scores + 1e-8) + torch.log(1 - policy_scores + 1e-8))
        
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # 训练策略网络（生成器）
        policy_inputs_detached = policy_inputs.detach()
        policy_scores_new = self.discriminator(policy_inputs_detached)
        
        # 策略损失（愚弄判别器）
        policy_loss = -torch.mean(torch.log(policy_scores_new + 1e-8))
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'policy_loss': policy_loss.item(),
            'expert_score': torch.mean(expert_scores).item(),
            'policy_score': torch.mean(policy_scores).item()
        }
    
    def reinforcement_learning_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """强化学习步骤（PPO）"""
        # 生成轨迹并计算奖励
        start_poses = batch.get('start_pose', torch.randn(self.batch_size, 3) * 0.5)
        end_poses = batch.get('end_pose', torch.randn(self.batch_size, 3) * 0.5)
        
        trajectories = self.generate_trajectory(start_poses, end_poses)
        
        # 计算奖励
        rewards = self.compute_trajectory_rewards(trajectories, start_poses, end_poses)
        
        # 计算价值和优势
        states = torch.cat([start_poses, end_poses], dim=1)
        values = self.value_net(states).squeeze(-1)
        
        # 简化的优势计算
        advantages = rewards - values.detach()
        returns = rewards
        
        # PPO更新
        old_log_probs = self.compute_log_probs(states, trajectories)
        
        for _ in range(self.ppo_epochs):
            # 计算新的log概率
            new_log_probs = self.compute_log_probs(states, trajectories)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # PPO裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            # 价值损失
            new_values = self.value_net(states).squeeze(-1)
            value_loss = F.mse_loss(new_values, returns)
            
            # 熵损失
            entropy_loss = -torch.mean(new_log_probs)  # 简化的熵计算
            
            # 总损失
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mean_reward': torch.mean(rewards).item(),
            'mean_advantage': torch.mean(advantages).item()
        }
    
    def compute_trajectory_rewards(self, trajectories: torch.Tensor, start_poses: torch.Tensor, 
                                 end_poses: torch.Tensor) -> torch.Tensor:
        """计算轨迹奖励"""
        batch_size = trajectories.shape[0]
        rewards = []
        
        for b in range(batch_size):
            traj = trajectories[b]
            start = start_poses[b]
            end = end_poses[b]
            
            # 1. 终点到达奖励
            final_pos = traj[-1]
            endpoint_reward = -torch.norm(final_pos - end)
            
            # 2. 路径平滑度奖励
            if len(traj) > 2:
                second_derivatives = traj[2:] - 2 * traj[1:-1] + traj[:-2]
                smoothness_reward = -torch.mean(torch.norm(second_derivatives, dim=1))
            else:
                smoothness_reward = torch.tensor(0.0)
            
            # 3. 路径长度惩罚
            path_lengths = torch.norm(traj[1:] - traj[:-1], dim=1)
            length_penalty = -torch.sum(path_lengths) * 0.1
            
            # 4. 起点约束
            start_penalty = -torch.norm(traj[0] - start) * 10
            
            total_reward = endpoint_reward + smoothness_reward + length_penalty + start_penalty
            rewards.append(total_reward)
        
        return torch.stack(rewards)
    
    def compute_log_probs(self, states: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率（简化版本）"""
        # 这里使用简化的概率计算
        # 在实际实现中，应该基于策略网络的输出分布计算
        actions = trajectories[:, 1:] - trajectories[:, :-1]  # 计算动作
        policy_outputs = self.policy(states)
        
        # 简化：假设高斯分布，计算负均方误差作为log概率的近似
        log_probs = -torch.mean(torch.norm(actions - policy_outputs.unsqueeze(1), dim=-1), dim=1)
        return log_probs
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        if self.current_phase == 'IL' and not self.il_completed:
            # 模仿学习阶段
            metrics = self.imitation_learning_step(batch)
            
            # 检查是否完成模仿学习
            if hasattr(self, 'il_step_count'):
                self.il_step_count += 1
            else:
                self.il_step_count = 1
            
            if self.il_step_count >= self.il_epochs:
                self.il_completed = True
                self.current_phase = 'RL'
                print("模仿学习阶段完成，切换到强化学习阶段")
            
            metrics['phase'] = 'IL'
            return metrics
        
        else:
            # 强化学习阶段
            metrics = self.reinforcement_learning_step(batch)
            metrics['phase'] = 'RL'
            return metrics
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        if self.current_phase == 'IL':
            # 模仿学习阶段使用MSE损失
            return F.mse_loss(predictions, targets)
        else:
            # 强化学习阶段使用奖励信号
            return -torch.mean(predictions)  # 最大化奖励
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'IL+RL Hybrid',
            'current_phase': self.current_phase,
            'il_completed': self.il_completed,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'trajectory_length': self.trajectory_length,
            'num_expert_trajectories': len(self.expert_trajectories)
        }