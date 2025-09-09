"""
Model evaluator for trajectory generation
轨迹生成模型评估器
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from .metrics import TrajectoryMetrics
from ..utils.logger import EvaluationLogger


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 dataset: Any,
                 config: Dict[str, Any],
                 logger: logging.Logger):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            dataset: 测试数据集
            config: 评估配置
            logger: 日志记录器
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger
        self.eval_logger = EvaluationLogger(logger)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 评估参数
        self.num_samples = config.get('generation', {}).get('num_samples', 1000)
        self.temperature = config.get('generation', {}).get('temperature', 1.0)
        
        # 创建评估指标
        self.metrics = TrajectoryMetrics(config.get('metrics', []))
        
        self.logger.info(f"评估器初始化完成，设备: {self.device}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        执行完整评估
        
        Returns:
            评估结果字典
        """
        self.logger.info("开始模型评估...")
        
        self.model.eval()
        
        # 生成轨迹样本
        generated_trajectories, ground_truth_trajectories, conditions = self._generate_samples()
        
        # 计算评估指标
        results = self._compute_metrics(generated_trajectories, ground_truth_trajectories, conditions)
        
        # 记录结果
        self.eval_logger.log_metrics(results)
        
        self.logger.info("模型评估完成")
        
        return results
    
    def _generate_samples(self) -> tuple:
        """
        生成评估样本
        
        Returns:
            (生成的轨迹, 真实轨迹, 条件)
        """
        generated_trajectories = []
        ground_truth_trajectories = []
        conditions = []
        
        # 从测试集采样
        test_loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=32, 
            shuffle=False
        )
        
        num_generated = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if num_generated >= self.num_samples:
                    break
                
                # 移动到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 提取条件
                start_poses = batch['start_pose']
                end_poses = batch['end_pose']
                batch_conditions = torch.cat([start_poses, end_poses], dim=-1)
                
                # 生成轨迹
                if hasattr(self.model, 'sample'):
                    generated = self.model.sample(start_poses, end_poses, num_samples=1)
                elif hasattr(self.model, 'generate'):
                    generated = self.model.generate(batch_conditions, num_samples=1)
                else:
                    # 回退到前向传播
                    generated = self.model(batch_conditions)
                
                # 收集结果
                generated_trajectories.append(generated.cpu().numpy())
                ground_truth_trajectories.append(batch['trajectory'].cpu().numpy())
                conditions.append(batch_conditions.cpu().numpy())
                
                num_generated += generated.shape[0]
        
        # 合并结果
        generated_trajectories = np.concatenate(generated_trajectories, axis=0)[:self.num_samples]
        ground_truth_trajectories = np.concatenate(ground_truth_trajectories, axis=0)[:self.num_samples]
        conditions = np.concatenate(conditions, axis=0)[:self.num_samples]
        
        self.logger.info(f"生成了{len(generated_trajectories)}个轨迹样本")
        
        return generated_trajectories, ground_truth_trajectories, conditions
    
    def _compute_metrics(self, generated_trajectories: np.ndarray, 
                        ground_truth_trajectories: np.ndarray,
                        conditions: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            generated_trajectories: 生成的轨迹 [N, seq_len, 7]
            ground_truth_trajectories: 真实轨迹 [N, seq_len, 7]
            conditions: 条件 [N, 14]
            
        Returns:
            指标结果字典
        """
        results = {}
        
        # 计算各项指标
        for metric_config in self.config.get('metrics', []):
            metric_name = metric_config['name']
            weight = metric_config.get('weight', 1.0)
            
            if metric_name == 'smoothness':
                score = self.metrics.compute_smoothness(generated_trajectories)
            elif metric_name == 'task_completion':
                score = self.metrics.compute_task_completion(generated_trajectories, conditions)
            elif metric_name == 'diversity':
                score = self.metrics.compute_diversity(generated_trajectories)
            elif metric_name == 'feasibility':
                score = self.metrics.compute_feasibility(generated_trajectories)
            elif metric_name == 'accuracy':
                score = self.metrics.compute_accuracy(generated_trajectories, ground_truth_trajectories)
            else:
                self.logger.warning(f"未知指标: {metric_name}")
                continue
            
            results[metric_name] = score
            results[f'{metric_name}_weighted'] = score * weight
        
        # 计算总分
        total_score = sum(results[k] for k in results.keys() if k.endswith('_weighted'))
        results['total_score'] = total_score
        
        return results
    
    def evaluate_single_trajectory(self, trajectory: np.ndarray, 
                                 condition: np.ndarray) -> Dict[str, float]:
        """
        评估单个轨迹
        
        Args:
            trajectory: 轨迹 [seq_len, 7]
            condition: 条件 [14]
            
        Returns:
            评估结果
        """
        # 扩展维度
        trajectories = trajectory[np.newaxis, ...]
        conditions = condition[np.newaxis, ...]
        
        # 计算指标
        results = {}
        results['smoothness'] = self.metrics.compute_smoothness(trajectories)[0]
        results['task_completion'] = self.metrics.compute_task_completion(trajectories, conditions)[0]
        results['feasibility'] = self.metrics.compute_feasibility(trajectories)[0]
        
        return results
    
    def compare_models(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        """
        比较多个模型
        
        Args:
            models: 模型字典 {name: model}
            
        Returns:
            比较结果 {model_name: metrics}
        """
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"评估模型: {model_name}")
            
            # 临时替换模型
            original_model = self.model
            self.model = model.to(self.device)
            
            # 评估
            model_results = self.evaluate()
            results[model_name] = model_results
            
            # 恢复原模型
            self.model = original_model
        
        # 记录比较结果
        self.eval_logger.log_comparison(results)
        
        # 找出最佳模型
        best_model = max(results.keys(), key=lambda k: results[k]['total_score'])
        best_score = results[best_model]['total_score']
        self.eval_logger.log_best_model(best_model, best_score, 'total_score')
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: Path):
        """
        生成评估报告
        
        Args:
            results: 评估结果
            output_path: 输出路径
        """
        report_content = []
        report_content.append("# 轨迹生成模型评估报告\n")
        
        # 基本信息
        report_content.append("## 基本信息")
        report_content.append(f"- 模型类型: {self.model.__class__.__name__}")
        report_content.append(f"- 评估样本数: {self.num_samples}")
        report_content.append(f"- 设备: {self.device}")
        report_content.append("")
        
        # 评估结果
        report_content.append("## 评估结果")
        for metric_name, score in results.items():
            if not metric_name.endswith('_weighted'):
                report_content.append(f"- {metric_name}: {score:.4f}")
        report_content.append("")
        
        # 总分
        report_content.append(f"**总分: {results.get('total_score', 0):.4f}**")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        self.logger.info(f"评估报告已保存到: {output_path}")
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            output_path: 输出路径
        """
        import json
        
        # 确保所有值都可序列化
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                serializable_results[k] = v
            elif isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif torch.is_tensor(v):
                serializable_results[k] = v.cpu().numpy().tolist()
            else:
                serializable_results[k] = str(v)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {output_path}")