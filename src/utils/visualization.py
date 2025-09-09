"""
Visualization utilities for trajectory generation
轨迹生成可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from .math_utils import compute_trajectory_length, compute_trajectory_smoothness

# 设置中文字体支持
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_3d_trajectory(self, trajectory: np.ndarray, title: str = "3D Trajectory", 
                          save_path: Optional[Path] = None, show_orientation: bool = True,
                          color: str = None) -> plt.Figure:
        """
        绘制3D轨迹
        
        Args:
            trajectory: 轨迹数据 [N, 7] (x, y, z, qx, qy, qz, qw)
            title: 图表标题
            save_path: 保存路径
            show_orientation: 是否显示方向
            color: 轨迹颜色
            
        Returns:
            matplotlib图形对象
        """
        fig = plt.figure(figsize=self.config.get('trajectory_3d', {}).get('figsize', [12, 8]))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = trajectory[:, :3]
        
        # 绘制轨迹线
        color = color or self.colors[0]
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=color, linewidth=2, alpha=0.8, label='轨迹')
        
        # 标记起点和终点
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  color='green', s=100, marker='o', label='起点')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color='red', s=100, marker='s', label='终点')
        
        # 显示方向箭头
        if show_orientation and len(trajectory) > 1:
            step = max(1, len(trajectory) // 10)  # 每10个点显示一个方向
            for i in range(0, len(trajectory), step):
                if i + 1 < len(trajectory):
                    direction = positions[i + 1] - positions[i]
                    direction = direction / np.linalg.norm(direction) * 0.05
                    ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                             direction[0], direction[1], direction[2],
                             color='black', alpha=0.6, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # 设置相等的轴比例
        max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                             positions[:, 1].max() - positions[:, 1].min(),
                             positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_trajectories(self, trajectories: Dict[str, np.ndarray], 
                                 title: str = "Multiple Trajectories",
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        绘制多条轨迹对比
        
        Args:
            trajectories: 轨迹字典 {name: trajectory}
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, (name, trajectory) in enumerate(trajectories.items()):
            positions = trajectory[:, :3]
            color = self.colors[i % len(self.colors)]
            
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=color, linewidth=2, alpha=0.8, label=name)
            
            # 标记起点和终点
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                      color=color, s=80, marker='o', alpha=0.8)
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                      color=color, s=80, marker='s', alpha=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict], output_dir: Path):
        """
        绘制模型比较图表
        
        Args:
            results: 模型结果字典
            output_dir: 输出目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取指标数据
        metrics_data = {}
        model_names = list(results.keys())
        
        # 收集所有指标
        all_metrics = set()
        for model_results in results.values():
            if 'metrics' in model_results:
                all_metrics.update(model_results['metrics'].keys())
        
        # 为每个指标创建数据
        for metric in all_metrics:
            metrics_data[metric] = []
            for model_name in model_names:
                if 'metrics' in results[model_name] and metric in results[model_name]['metrics']:
                    metrics_data[metric].append(results[model_name]['metrics'][metric])
                else:
                    metrics_data[metric].append(0)
        
        # 绘制雷达图
        self._plot_radar_chart(metrics_data, model_names, output_dir / "model_comparison_radar.png")
        
        # 绘制柱状图
        self._plot_bar_chart(metrics_data, model_names, output_dir / "model_comparison_bar.png")
        
        # 绘制箱线图（如果有多次运行的结果）
        if self._has_multiple_runs(results):
            self._plot_box_chart(results, output_dir / "model_comparison_box.png")
    
    def _plot_radar_chart(self, metrics_data: Dict, model_names: List[str], save_path: Path):
        """绘制雷达图"""
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        metrics = list(metrics_data.keys())
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for i, model_name in enumerate(model_names):
            values = [metrics_data[metric][i] for metric in metrics]
            values += values[:1]  # 闭合图形
            
            color = self.colors[i % len(self.colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', size=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bar_chart(self, metrics_data: Dict, model_names: List[str], save_path: Path):
        """绘制柱状图"""
        metrics = list(metrics_data.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model_name in enumerate(model_names):
            values = [metrics_data[metric][i] for metric in metrics]
            color = self.colors[i % len(self.colors)]
            ax.bar(x + i * width, values, width, label=model_name, color=color, alpha=0.8)
        
        ax.set_xlabel('评估指标')
        ax.set_ylabel('分数')
        ax.set_title('模型性能对比')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_box_chart(self, results: Dict, save_path: Path):
        """绘制箱线图"""
        # 这里假设results中包含多次运行的数据
        # 实际实现需要根据具体的数据结构调整
        pass
    
    def _has_multiple_runs(self, results: Dict) -> bool:
        """检查是否有多次运行的结果"""
        for model_results in results.values():
            if 'runs' in model_results:
                return True
        return False
    
    def plot_trajectory_samples(self, results: Dict, output_dir: Path, num_samples: int = 5):
        """
        绘制轨迹样本
        
        Args:
            results: 模型结果字典
            output_dir: 输出目录
            num_samples: 每个模型显示的样本数量
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_results in results.items():
            if 'sample_trajectories' in model_results:
                trajectories = model_results['sample_trajectories'][:num_samples]
                
                # 为每个模型创建单独的图
                fig = plt.figure(figsize=(15, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                for i, trajectory in enumerate(trajectories):
                    positions = trajectory[:, :3]
                    color = self.colors[i % len(self.colors)]
                    
                    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                           color=color, linewidth=2, alpha=0.8, label=f'样本 {i+1}')
                    
                    # 标记起点和终点
                    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                              color=color, s=80, marker='o', alpha=0.8)
                    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                              color=color, s=80, marker='s', alpha=0.8)
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'{model_name} - 轨迹样本')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{model_name}_samples.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_training_curves(self, training_logs: Dict, save_path: Path):
        """
        绘制训练曲线
        
        Args:
            training_logs: 训练日志字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['loss', 'val_loss', 'learning_rate', 'gradient_norm']
        
        for i, metric in enumerate(metrics):
            if i < len(axes) and metric in training_logs:
                ax = axes[i]
                data = training_logs[metric]
                
                if isinstance(data, dict):
                    # 多个模型的数据
                    for model_name, values in data.items():
                        ax.plot(values, label=model_name)
                    ax.legend()
                else:
                    # 单个模型的数据
                    ax.plot(data)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} 变化曲线')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_3d_plot(self, trajectories: Dict[str, np.ndarray], 
                                  save_path: Optional[Path] = None) -> go.Figure:
        """
        创建交互式3D轨迹图
        
        Args:
            trajectories: 轨迹字典
            save_path: 保存路径
            
        Returns:
            plotly图形对象
        """
        fig = go.Figure()
        
        for i, (name, trajectory) in enumerate(trajectories.items()):
            positions = trajectory[:, :3]
            color = self.colors[i % len(self.colors)]
            
            # 添加轨迹线
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1], 
                z=positions[:, 2],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color)
            ))
            
            # 添加起点和终点标记
            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]],
                y=[positions[0, 1]],
                z=[positions[0, 2]],
                mode='markers',
                name=f'{name} 起点',
                marker=dict(size=10, color='green', symbol='circle')
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[positions[-1, 0]],
                y=[positions[-1, 1]],
                z=[positions[-1, 2]],
                mode='markers',
                name=f'{name} 终点',
                marker=dict(size=10, color='red', symbol='square')
            ))
        
        fig.update_layout(
            title='交互式3D轨迹可视化',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube'
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_trajectory_statistics(self, trajectories: Dict[str, List[np.ndarray]], 
                                 save_path: Path):
        """
        绘制轨迹统计信息
        
        Args:
            trajectories: 轨迹字典 {model_name: [trajectory1, trajectory2, ...]}
            save_path: 保存路径
        """
        # 计算统计信息
        stats_data = []
        
        for model_name, trajs in trajectories.items():
            for traj in trajs:
                length = compute_trajectory_length(traj)
                smoothness = compute_trajectory_smoothness(traj)
                
                stats_data.append({
                    'Model': model_name,
                    'Length': length,
                    'Smoothness': smoothness,
                    'Duration': len(traj) * 0.1  # 假设时间步长为0.1s
                })
        
        df = pd.DataFrame(stats_data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 轨迹长度分布
        sns.boxplot(data=df, x='Model', y='Length', ax=axes[0, 0])
        axes[0, 0].set_title('轨迹长度分布')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 平滑度分布
        sns.boxplot(data=df, x='Model', y='Smoothness', ax=axes[0, 1])
        axes[0, 1].set_title('轨迹平滑度分布')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 持续时间分布
        sns.boxplot(data=df, x='Model', y='Duration', ax=axes[1, 0])
        axes[1, 0].set_title('轨迹持续时间分布')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 长度vs平滑度散点图
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            axes[1, 1].scatter(model_data['Length'], model_data['Smoothness'], 
                             label=model, alpha=0.7)
        axes[1, 1].set_xlabel('轨迹长度')
        axes[1, 1].set_ylabel('平滑度')
        axes[1, 1].set_title('轨迹长度 vs 平滑度')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_animation(self, trajectory: np.ndarray, save_path: Path, 
                      fps: int = 30, duration: float = 5.0):
        """
        保存轨迹动画
        
        Args:
            trajectory: 轨迹数据
            save_path: 保存路径
            fps: 帧率
            duration: 持续时间
        """
        try:
            from matplotlib.animation import FuncAnimation
            
            positions = trajectory[:, :3]
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 设置轴范围
            ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
            ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
            ax.set_zlim(positions[:, 2].min(), positions[:, 2].max())
            
            line, = ax.plot([], [], [], 'b-', linewidth=2)
            point, = ax.plot([], [], [], 'ro', markersize=8)
            
            def animate(frame):
                end_idx = min(frame + 1, len(positions))
                line.set_data(positions[:end_idx, 0], positions[:end_idx, 1])
                line.set_3d_properties(positions[:end_idx, 2])
                
                if end_idx > 0:
                    point.set_data([positions[end_idx-1, 0]], [positions[end_idx-1, 1]])
                    point.set_3d_properties([positions[end_idx-1, 2]])
                
                return line, point
            
            frames = int(fps * duration)
            frame_interval = max(1, len(positions) // frames)
            
            anim = FuncAnimation(fig, animate, frames=frames, 
                               interval=1000/fps, blit=False, repeat=True)
            
            # 保存为GIF
            if save_path.suffix.lower() == '.gif':
                anim.save(save_path, writer='pillow', fps=fps)
            else:
                anim.save(save_path, writer='ffmpeg', fps=fps)
            
            plt.close()
            
        except ImportError:
            print("警告: 无法创建动画，缺少必要的依赖库")
        except Exception as e:
            print(f"创建动画时出错: {e}")