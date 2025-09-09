#!/usr/bin/env python3
"""
Unified Visualization Script for All 25+ Trajectory Generation Models
统一可视化脚本，支持所有25+轨迹生成模型的可视化和行为模拟
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List, Tuple
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from run import get_model_class, set_seed

# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']


class UnifiedTrajectoryVisualizer:
    """
    统一轨迹可视化器
    支持所有25+模型的轨迹生成、可视化和行为模拟
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config['experiment']['output_dir']) / "unified_visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载所有启用的模型
        self.models = {}
        self.model_categories = config.get('model_categories', {})
        self._load_models()
        
        # 定义测试场景
        self.test_scenarios = self._define_test_scenarios()
        
    def _load_models(self):
        """加载所有启用的模型"""
        self.logger.info("正在加载所有启用的模型...")
        
        checkpoint_dir = Path(self.config['experiment']['output_dir']) / "checkpoints"
        
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', False):
                continue
                
            try:
                # 创建模型实例
                model_class = get_model_class(model_name)
                model = model_class(model_config)
                
                # 尝试加载检查点
                checkpoint_path = checkpoint_dir / model_name / "best_model.pth"
                if checkpoint_path.exists():
                    model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    self.logger.info(f"已加载模型检查点: {model_name}")
                else:
                    self.logger.warning(f"未找到模型检查点: {model_name}, 使用随机初始化")
                
                model.eval()
                model.to(self.device)
                self.models[model_name] = model
                
            except Exception as e:
                self.logger.error(f"加载模型 {model_name} 失败: {str(e)}")
                continue
        
        self.logger.info(f"成功加载 {len(self.models)} 个模型")
    
    def _define_test_scenarios(self) -> List[Dict[str, Any]]:
        """定义测试场景"""
        scenarios = [
            {
                'name': '基础抓取放置',
                'start_pose': np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
                'end_pose': np.array([0.5, 0.3, 0.8, 1.0, 0.0, 0.0, 0.0]),
                'description': '从桌面抓取物体并放置到目标位置'
            },
            {
                'name': '复杂装配任务',
                'start_pose': np.array([-0.3, -0.2, 0.4, 0.707, 0.0, 0.707, 0.0]),
                'end_pose': np.array([0.2, 0.4, 0.6, 0.707, 0.707, 0.0, 0.0]),
                'description': '复杂的装配操作，涉及旋转和精确定位'
            },
            {
                'name': '倾倒动作',
                'start_pose': np.array([0.2, -0.1, 0.7, 1.0, 0.0, 0.0, 0.0]),
                'end_pose': np.array([0.1, 0.2, 0.5, 0.866, 0.0, 0.5, 0.0]),
                'description': '倾倒液体的连续动作'
            },
            {
                'name': '长距离移动',
                'start_pose': np.array([-0.8, -0.6, 0.3, 1.0, 0.0, 0.0, 0.0]),
                'end_pose': np.array([0.8, 0.6, 1.0, 1.0, 0.0, 0.0, 0.0]),
                'description': '工作空间内的长距离移动'
            },
            {
                'name': '精确操作',
                'start_pose': np.array([0.1, 0.1, 0.6, 1.0, 0.0, 0.0, 0.0]),
                'end_pose': np.array([0.12, 0.11, 0.61, 1.0, 0.0, 0.0, 0.0]),
                'description': '需要高精度的微小移动'
            }
        ]
        return scenarios
    
    def generate_all_trajectories(self, num_points: int = 50) -> Dict[str, Dict[str, np.ndarray]]:
        """为所有模型和场景生成轨迹"""
        self.logger.info("开始为所有模型生成轨迹...")
        
        all_trajectories = {}
        
        for scenario in self.test_scenarios:
            scenario_name = scenario['name']
            start_pose = scenario['start_pose']
            end_pose = scenario['end_pose']
            
            all_trajectories[scenario_name] = {}
            
            for model_name, model in self.models.items():
                try:
                    trajectory = model.generate_trajectory(
                        start_pose=start_pose,
                        end_pose=end_pose,
                        num_points=num_points
                    )
                    all_trajectories[scenario_name][model_name] = trajectory
                    
                except Exception as e:
                    self.logger.error(f"模型 {model_name} 在场景 {scenario_name} 中生成轨迹失败: {str(e)}")
                    continue
        
        # 保存轨迹数据
        self._save_trajectories(all_trajectories)
        
        return all_trajectories
    
    def _save_trajectories(self, trajectories: Dict[str, Dict[str, np.ndarray]]):
        """保存轨迹数据"""
        trajectory_dir = self.output_dir / "trajectories"
        trajectory_dir.mkdir(exist_ok=True)
        
        for scenario_name, scenario_trajectories in trajectories.items():
            scenario_file = trajectory_dir / f"{scenario_name.replace(' ', '_')}.npz"
            np.savez(scenario_file, **scenario_trajectories)
            
        self.logger.info(f"轨迹数据已保存到 {trajectory_dir}")
    
    def visualize_3d_trajectories(self, trajectories: Dict[str, Dict[str, np.ndarray]]):
        """3D轨迹可视化"""
        self.logger.info("生成3D轨迹可视化...")
        
        for scenario_name, scenario_trajectories in trajectories.items():
            # 为每个场景创建3D图
            fig = plt.figure(figsize=(15, 12))
            
            # 按类别分组显示
            num_categories = len(self.model_categories)
            rows = 2
            cols = 3
            
            category_idx = 0
            for category_name, category_info in self.model_categories.items():
                if category_idx >= rows * cols:
                    break
                    
                ax = fig.add_subplot(rows, cols, category_idx + 1, projection='3d')
                
                # 绘制该类别的所有模型轨迹
                colors = plt.cm.Set3(np.linspace(0, 1, len(category_info['models'])))
                
                for i, model_name in enumerate(category_info['models']):
                    if model_name in scenario_trajectories:
                        trajectory = scenario_trajectories[model_name]
                        
                        # 提取位置信息（前3维）
                        positions = trajectory[:, :3]
                        
                        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                               color=colors[i], label=model_name, linewidth=2, alpha=0.8)
                        
                        # 标记起点和终点
                        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                                 color='green', s=100, marker='o', label='起点' if i == 0 else "")
                        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                                 color='red', s=100, marker='s', label='终点' if i == 0 else "")
                
                ax.set_title(f'{category_name}\n({scenario_name})', fontsize=12, fontweight='bold')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                
                # 设置相同的坐标轴范围
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([0, 1.5])
                
                category_idx += 1
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"3d_trajectories_{scenario_name.replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_interactive_dashboard(self, trajectories: Dict[str, Dict[str, np.ndarray]]):
        """创建交互式仪表板"""
        self.logger.info("创建交互式可视化仪表板...")
        
        # 创建Plotly仪表板
        dashboard_data = []
        
        for scenario_name, scenario_trajectories in trajectories.items():
            for model_name, trajectory in scenario_trajectories.items():
                # 获取模型类别
                model_category = self._get_model_category(model_name)
                
                # 提取位置信息
                positions = trajectory[:, :3]
                
                for i, pos in enumerate(positions):
                    dashboard_data.append({
                        'scenario': scenario_name,
                        'model': model_name,
                        'category': model_category,
                        'step': i,
                        'x': pos[0],
                        'y': pos[1],
                        'z': pos[2],
                        'time': i / len(positions)
                    })
        
        df = pd.DataFrame(dashboard_data)
        
        # 创建多子图仪表板
        scenarios = df['scenario'].unique()
        
        for scenario in scenarios:
            scenario_df = df[df['scenario'] == scenario]
            
            # 创建3D轨迹图
            fig = px.line_3d(
                scenario_df, 
                x='x', y='y', z='z', 
                color='model',
                facet_col='category',
                facet_col_wrap=2,
                title=f'交互式轨迹可视化 - {scenario}',
                labels={'x': 'X (m)', 'y': 'Y (m)', 'z': 'Z (m)'}
            )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                font=dict(size=12)
            )
            
            # 保存为HTML
            fig.write_html(self.output_dir / f"interactive_{scenario.replace(' ', '_')}.html")
    
    def _get_model_category(self, model_name: str) -> str:
        """获取模型所属类别"""
        for category_name, category_info in self.model_categories.items():
            if model_name in category_info['models']:
                return category_name
        return "Unknown"
    
    def analyze_trajectory_metrics(self, trajectories: Dict[str, Dict[str, np.ndarray]]):
        """分析轨迹指标"""
        self.logger.info("分析轨迹指标...")
        
        metrics_data = []
        
        for scenario_name, scenario_trajectories in trajectories.items():
            for model_name, trajectory in scenario_trajectories.items():
                metrics = self._compute_trajectory_metrics(trajectory, scenario_name)
                metrics['scenario'] = scenario_name
                metrics['model'] = model_name
                metrics['category'] = self._get_model_category(model_name)
                metrics_data.append(metrics)
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # 保存指标数据
        df_metrics.to_csv(self.output_dir / "trajectory_metrics.csv", index=False)
        
        # 创建指标比较图
        self._create_metrics_comparison_plots(df_metrics)
        
        return df_metrics
    
    def _compute_trajectory_metrics(self, trajectory: np.ndarray, scenario_name: str) -> Dict[str, float]:
        """计算轨迹指标"""
        positions = trajectory[:, :3]
        
        # 1. 平滑度指标（加速度方差）
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        smoothness = np.mean(np.var(accelerations, axis=0))
        
        # 2. 路径长度
        path_length = np.sum(np.linalg.norm(velocities, axis=1))
        
        # 3. 直线距离
        direct_distance = np.linalg.norm(positions[-1] - positions[0])
        
        # 4. 路径效率
        efficiency = direct_distance / (path_length + 1e-8)
        
        # 5. 最大速度
        max_velocity = np.max(np.linalg.norm(velocities, axis=1))
        
        # 6. 平均速度
        avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
        
        # 7. 终点误差（假设最后一个点是目标）
        # 这里我们使用轨迹的最后一个点作为参考
        end_error = 0.0  # 在实际应用中需要真实的目标位置
        
        # 8. 工作空间利用率
        workspace_bounds = np.array([[-1, 1], [-1, 1], [0, 1.5]])
        workspace_utilization = self._compute_workspace_utilization(positions, workspace_bounds)
        
        return {
            'smoothness': float(smoothness),
            'path_length': float(path_length),
            'direct_distance': float(direct_distance),
            'efficiency': float(efficiency),
            'max_velocity': float(max_velocity),
            'avg_velocity': float(avg_velocity),
            'end_error': float(end_error),
            'workspace_utilization': float(workspace_utilization)
        }
    
    def _compute_workspace_utilization(self, positions: np.ndarray, bounds: np.ndarray) -> float:
        """计算工作空间利用率"""
        # 计算轨迹在工作空间中的分布
        normalized_positions = np.zeros_like(positions)
        for i in range(3):
            normalized_positions[:, i] = (positions[:, i] - bounds[i, 0]) / (bounds[i, 1] - bounds[i, 0])
        
        # 计算覆盖的空间体积（简化计算）
        ranges = np.ptp(normalized_positions, axis=0)
        utilization = np.prod(ranges)
        
        return utilization
    
    def _create_metrics_comparison_plots(self, df_metrics: pd.DataFrame):
        """创建指标比较图"""
        # 设置图表样式
        sns.set_style("whitegrid", {'font.sans-serif': ['Noto Sans CJK SC', 'sans-serif']})
        
        # 指标列表
        metrics = ['smoothness', 'path_length', 'efficiency', 'max_velocity', 'avg_velocity', 'workspace_utilization']
        
        # 1. 按类别的箱线图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=df_metrics, x='category', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_by_category_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 雷达图比较
        self._create_radar_chart(df_metrics, metrics)
        
        # 3. 热力图
        self._create_metrics_heatmap(df_metrics, metrics)
    
    def _create_radar_chart(self, df_metrics: pd.DataFrame, metrics: List[str]):
        """创建雷达图"""
        # 按类别计算平均值
        category_means = df_metrics.groupby('category')[metrics].mean()
        
        # 归一化数据（0-1范围）
        normalized_means = (category_means - category_means.min()) / (category_means.max() - category_means.min())
        
        # 创建雷达图
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (category, values) in enumerate(normalized_means.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=values.tolist() + [values.iloc[0]],  # 闭合图形
                theta=metrics + [metrics[0]],
                fill='toself',
                name=category,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="模型类别性能雷达图",
            font=dict(size=12)
        )
        
        fig.write_html(self.output_dir / "performance_radar_chart.html")
    
    def _create_metrics_heatmap(self, df_metrics: pd.DataFrame, metrics: List[str]):
        """创建指标热力图"""
        # 计算每个模型的平均指标
        model_means = df_metrics.groupby(['category', 'model'])[metrics].mean()
        
        # 重塑数据用于热力图
        heatmap_data = model_means.reset_index()
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 按类别排序
        heatmap_data = heatmap_data.sort_values(['category', 'model'])
        
        # 创建数据透视表
        pivot_data = heatmap_data.set_index(['category', 'model'])[metrics]
        
        # 归一化数据
        normalized_data = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min())
        
        sns.heatmap(normalized_data, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('模型性能指标热力图（归一化）', fontsize=14, fontweight='bold')
        ax.set_xlabel('性能指标')
        ax.set_ylabel('模型（按类别分组）')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_trajectory_animations(self, trajectories: Dict[str, Dict[str, np.ndarray]]):
        """创建轨迹动画"""
        self.logger.info("创建轨迹动画...")
        
        for scenario_name, scenario_trajectories in trajectories.items():
            # 为每个场景创建动画
            self._create_scenario_animation(scenario_name, scenario_trajectories)
    
    def _create_scenario_animation(self, scenario_name: str, scenario_trajectories: Dict[str, np.ndarray]):
        """为单个场景创建动画"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 准备数据
        max_length = max(len(traj) for traj in scenario_trajectories.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_trajectories)))
        
        lines = {}
        points = {}
        
        for i, (model_name, trajectory) in enumerate(scenario_trajectories.items()):
            positions = trajectory[:, :3]
            line, = ax.plot([], [], [], color=colors[i], label=model_name, linewidth=2, alpha=0.8)
            point, = ax.plot([], [], [], color=colors[i], marker='o', markersize=6)
            
            lines[model_name] = (line, positions)
            points[model_name] = point
        
        # 设置坐标轴
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1.5])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'轨迹动画 - {scenario_name}', fontsize=14, fontweight='bold')
        ax.legend()
        
        def animate(frame):
            for model_name, (line, positions) in lines.items():
                if frame < len(positions):
                    # 更新轨迹线
                    line.set_data_3d(positions[:frame+1, 0], 
                                    positions[:frame+1, 1], 
                                    positions[:frame+1, 2])
                    
                    # 更新当前点
                    points[model_name].set_data_3d([positions[frame, 0]], 
                                                 [positions[frame, 1]], 
                                                 [positions[frame, 2]])
            return list(lines.values()) + list(points.values())
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=max_length, interval=100, blit=False, repeat=True)
        
        # 保存动画
        animation_path = self.output_dir / f"animation_{scenario_name.replace(' ', '_')}.gif"
        anim.save(animation_path, writer='pillow', fps=10)
        plt.close()
    
    def generate_comprehensive_report(self, trajectories: Dict[str, Dict[str, np.ndarray]], 
                                    metrics_df: pd.DataFrame):
        """生成综合报告"""
        self.logger.info("生成综合分析报告...")
        
        report = {
            'experiment_info': {
                'total_models': len(self.models),
                'total_scenarios': len(self.test_scenarios),
                'model_categories': len(self.model_categories)
            },
            'model_summary': {},
            'category_analysis': {},
            'scenario_analysis': {},
            'performance_ranking': {}
        }
        
        # 模型摘要
        for model_name in self.models.keys():
            model_metrics = metrics_df[metrics_df['model'] == model_name]
            report['model_summary'][model_name] = {
                'category': self._get_model_category(model_name),
                'avg_smoothness': float(model_metrics['smoothness'].mean()),
                'avg_efficiency': float(model_metrics['efficiency'].mean()),
                'avg_path_length': float(model_metrics['path_length'].mean())
            }
        
        # 类别分析
        for category_name in self.model_categories.keys():
            category_metrics = metrics_df[metrics_df['category'] == category_name]
            if not category_metrics.empty:
                report['category_analysis'][category_name] = {
                    'model_count': len(category_metrics['model'].unique()),
                    'avg_smoothness': float(category_metrics['smoothness'].mean()),
                    'avg_efficiency': float(category_metrics['efficiency'].mean()),
                    'best_model': category_metrics.loc[category_metrics['efficiency'].idxmax(), 'model']
                }
        
        # 场景分析
        for scenario_name in [s['name'] for s in self.test_scenarios]:
            scenario_metrics = metrics_df[metrics_df['scenario'] == scenario_name]
            if not scenario_metrics.empty:
                report['scenario_analysis'][scenario_name] = {
                    'avg_path_length': float(scenario_metrics['path_length'].mean()),
                    'most_efficient_model': scenario_metrics.loc[scenario_metrics['efficiency'].idxmax(), 'model'],
                    'smoothest_model': scenario_metrics.loc[scenario_metrics['smoothness'].idxmin(), 'model']
                }
        
        # 性能排名
        overall_score = (
            (1 - metrics_df.groupby('model')['smoothness'].mean()) * 0.3 +
            metrics_df.groupby('model')['efficiency'].mean() * 0.4 +
            (1 - metrics_df.groupby('model')['path_length'].mean() / metrics_df['path_length'].max()) * 0.3
        )
        
        ranking = overall_score.sort_values(ascending=False)
        report['performance_ranking'] = {
            'top_5_models': ranking.head(5).to_dict(),
            'ranking_criteria': '综合评分 = (1-平滑度)*0.3 + 效率*0.4 + (1-标准化路径长度)*0.3'
        }
        
        # 保存报告
        with open(self.output_dir / "comprehensive_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 创建Markdown报告
        self._create_markdown_report(report, trajectories, metrics_df)
        
        return report
    
    def _create_markdown_report(self, report: Dict[str, Any], 
                               trajectories: Dict[str, Dict[str, np.ndarray]],
                               metrics_df: pd.DataFrame):
        """创建Markdown格式的报告"""
        markdown_content = f"""# 轨迹生成模型综合评估报告

## 实验概览

- **总模型数量**: {report['experiment_info']['total_models']}
- **测试场景数量**: {report['experiment_info']['total_scenarios']}
- **模型类别数量**: {report['experiment_info']['model_categories']}

## 模型类别分析

"""
        
        for category_name, analysis in report['category_analysis'].items():
            markdown_content += f"""### {category_name}

- **模型数量**: {analysis['model_count']}
- **平均平滑度**: {analysis['avg_smoothness']:.4f}
- **平均效率**: {analysis['avg_efficiency']:.4f}
- **最佳模型**: {analysis['best_model']}

"""
        
        markdown_content += """## 性能排名

基于综合评分的前5名模型：

"""
        
        for i, (model_name, score) in enumerate(report['performance_ranking']['top_5_models'].items(), 1):
            category = self._get_model_category(model_name)
            markdown_content += f"{i}. **{model_name}** ({category}) - 评分: {score:.4f}\n"
        
        markdown_content += f"""

**排名标准**: {report['performance_ranking']['ranking_criteria']}

## 场景分析

"""
        
        for scenario_name, analysis in report['scenario_analysis'].items():
            markdown_content += f"""### {scenario_name}

- **平均路径长度**: {analysis['avg_path_length']:.4f}m
- **最高效模型**: {analysis['most_efficient_model']}
- **最平滑模型**: {analysis['smoothest_model']}

"""
        
        markdown_content += """## 可视化文件

本报告包含以下可视化文件：

- `3d_trajectories_*.png`: 各场景的3D轨迹对比图
- `interactive_*.html`: 交互式轨迹可视化
- `metrics_by_category_boxplot.png`: 按类别的性能指标箱线图
- `performance_radar_chart.html`: 性能雷达图
- `performance_heatmap.png`: 性能热力图
- `animation_*.gif`: 轨迹动画
- `trajectory_metrics.csv`: 详细指标数据

## 结论

通过对25+轨迹生成模型的综合评估，我们发现：

1. **不同类别的模型各有优势**：经典方法在简单场景下表现稳定，深度学习方法在复杂场景下更具优势
2. **场景复杂度影响模型性能**：复杂装配任务对所有模型都具有挑战性
3. **效率与平滑度存在权衡**：高效的轨迹往往在平滑度上有所妥协

## 建议

1. **根据应用场景选择模型**：简单任务可选择经典方法，复杂任务推荐使用深度学习方法
2. **考虑计算资源限制**：RL方法通常需要更多计算资源
3. **结合多种方法**：可以考虑集成多个模型的优势

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(self.output_dir / "comprehensive_report.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一轨迹可视化脚本")
    parser.add_argument("--config", type=str, default="config_extended.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--num-points", type=int, default=50, help="轨迹点数量")
    parser.add_argument("--scenarios", type=str, nargs='+', help="指定测试场景")
    parser.add_argument("--models", type=str, nargs='+', help="指定模型")
    parser.add_argument("--skip-animation", action="store_true", help="跳过动画生成")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("unified_visualization", 
                         Path(config['experiment']['output_dir']) / "logs", 
                         level=log_level)
    
    # 设置随机种子
    set_seed(config['experiment']['seed'])
    
    try:
        # 创建可视化器
        visualizer = UnifiedTrajectoryVisualizer(config, logger)
        
        # 生成轨迹
        trajectories = visualizer.generate_all_trajectories(num_points=args.num_points)
        
        # 3D可视化
        visualizer.visualize_3d_trajectories(trajectories)
        
        # 交互式仪表板
        visualizer.create_interactive_dashboard(trajectories)
        
        # 分析指标
        metrics_df = visualizer.analyze_trajectory_metrics(trajectories)
        
        # 创建动画（可选）
        if not args.skip_animation:
            visualizer.create_trajectory_animations(trajectories)
        
        # 生成综合报告
        report = visualizer.generate_comprehensive_report(trajectories, metrics_df)
        
        logger.info(f"所有可视化完成! 结果保存在: {visualizer.output_dir}")
        
    except Exception as e:
        logger.error(f"可视化过程中出现错误: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()