#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for All 25+ Trajectory Generation Models
全面基准测试脚本，支持所有25+轨迹生成模型的训练、评估和比较
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from run import get_model_class, set_seed, train_model, evaluate_model


class ComprehensiveBenchmark:
    """
    综合基准测试类
    支持所有25+模型的并行训练、评估和比较分析
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config['experiment']['output_dir'])
        self.benchmark_dir = self.output_dir / "comprehensive_benchmark"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取启用的模型
        self.enabled_models = [
            name for name, config in self.config['models'].items() 
            if config.get('enabled', False)
        ]
        
        self.logger.info(f"启用的模型数量: {len(self.enabled_models)}")
        self.logger.info(f"启用的模型: {', '.join(self.enabled_models)}")
        
        # 模型分类
        self.model_categories = self.config.get('model_categories', {})
        
        # 基准测试配置
        self.benchmark_config = {
            'max_parallel_models': config.get('benchmark', {}).get('max_parallel_models', 3),
            'timeout_per_model': config.get('benchmark', {}).get('timeout_per_model', 3600),  # 1小时
            'retry_failed': config.get('benchmark', {}).get('retry_failed', True),
            'save_intermediate': config.get('benchmark', {}).get('save_intermediate', True)
        }
    
    def run_comprehensive_benchmark(self):
        """运行综合基准测试"""
        self.logger.info("开始综合基准测试...")
        
        # 1. 数据生成
        self._generate_benchmark_data()
        
        # 2. 并行训练所有模型
        training_results = self._parallel_train_models()
        
        # 3. 并行评估所有模型
        evaluation_results = self._parallel_evaluate_models()
        
        # 4. 性能分析
        performance_analysis = self._analyze_model_performance(evaluation_results)
        
        # 5. 生成综合报告
        comprehensive_report = self._generate_comprehensive_report(
            training_results, evaluation_results, performance_analysis
        )
        
        # 6. 创建可视化
        self._create_benchmark_visualizations(comprehensive_report)
        
        self.logger.info("综合基准测试完成!")
        return comprehensive_report
    
    def _generate_benchmark_data(self):
        """生成基准测试数据"""
        self.logger.info("生成基准测试数据...")
        
        from run import generate_data
        
        # 使用更大的数据集进行基准测试
        original_num_trajectories = self.config['data']['generation']['num_trajectories']
        self.config['data']['generation']['num_trajectories'] = 15000  # 增加数据量
        
        try:
            train_data, val_data, test_data = generate_data(self.config, self.logger)
            
            # 保存数据集信息
            dataset_info = {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'trajectory_length': self.config['data']['generation']['trajectory_length'],
                'num_modalities': self.config['data']['generation']['num_modalities'],
                'tasks': [task['name'] for task in self.config['data']['tasks']]
            }
            
            with open(self.benchmark_dir / "dataset_info.json", 'w') as f:
                json.dump(dataset_info, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"数据生成失败: {str(e)}")
            raise
        finally:
            # 恢复原始配置
            self.config['data']['generation']['num_trajectories'] = original_num_trajectories
    
    def _parallel_train_models(self) -> Dict[str, Dict[str, Any]]:
        """并行训练所有模型"""
        self.logger.info("开始并行训练所有模型...")
        
        training_results = {}
        failed_models = []
        
        # 按类别分组训练（避免资源冲突）
        for category_name, category_info in self.model_categories.items():
            category_models = [
                model for model in category_info['models'] 
                if model in self.enabled_models
            ]
            
            if not category_models:
                continue
                
            self.logger.info(f"训练 {category_name} 类别的 {len(category_models)} 个模型...")
            
            # 并行训练该类别的模型
            with ThreadPoolExecutor(max_workers=self.benchmark_config['max_parallel_models']) as executor:
                future_to_model = {
                    executor.submit(self._train_single_model, model_name): model_name
                    for model_name in category_models
                }
                
                for future in as_completed(future_to_model, timeout=self.benchmark_config['timeout_per_model']):
                    model_name = future_to_model[future]
                    
                    try:
                        result = future.result()
                        training_results[model_name] = result
                        self.logger.info(f"模型 {model_name} 训练完成")
                        
                    except Exception as e:
                        self.logger.error(f"模型 {model_name} 训练失败: {str(e)}")
                        failed_models.append(model_name)
                        training_results[model_name] = {'status': 'failed', 'error': str(e)}
        
        # 保存训练结果
        with open(self.benchmark_dir / "training_results.json", 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        self.logger.info(f"训练完成! 成功: {len(training_results) - len(failed_models)}, 失败: {len(failed_models)}")
        
        return training_results
    
    def _train_single_model(self, model_name: str) -> Dict[str, Any]:
        """训练单个模型"""
        start_time = time.time()
        
        try:
            # 使用原有的训练函数
            model = train_model(self.config, model_name, self.logger)
            
            training_time = time.time() - start_time
            
            # 获取模型信息
            if model is not None:
                model_info = model.get_model_info()
            else:
                model_info = {}
            
            return {
                'status': 'success',
                'training_time': training_time,
                'model_info': model_info
            }
            
        except Exception as e:
            training_time = time.time() - start_time
            return {
                'status': 'failed',
                'training_time': training_time,
                'error': str(e)
            }
    
    def _parallel_evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """并行评估所有模型"""
        self.logger.info("开始并行评估所有模型...")
        
        evaluation_results = {}
        failed_evaluations = []
        
        # 只评估训练成功的模型
        successful_models = [
            model for model in self.enabled_models
            if Path(self.config['experiment']['output_dir']) / "checkpoints" / model / "best_model.pth"
        ]
        
        # 并行评估
        with ThreadPoolExecutor(max_workers=self.benchmark_config['max_parallel_models']) as executor:
            future_to_model = {
                executor.submit(self._evaluate_single_model, model_name): model_name
                for model_name in successful_models
            }
            
            for future in as_completed(future_to_model, timeout=self.benchmark_config['timeout_per_model']):
                model_name = future_to_model[future]
                
                try:
                    result = future.result()
                    evaluation_results[model_name] = result
                    self.logger.info(f"模型 {model_name} 评估完成")
                    
                except Exception as e:
                    self.logger.error(f"模型 {model_name} 评估失败: {str(e)}")
                    failed_evaluations.append(model_name)
                    evaluation_results[model_name] = {'status': 'failed', 'error': str(e)}
        
        # 保存评估结果
        with open(self.benchmark_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        self.logger.info(f"评估完成! 成功: {len(evaluation_results) - len(failed_evaluations)}, 失败: {len(failed_evaluations)}")
        
        return evaluation_results
    
    def _evaluate_single_model(self, model_name: str) -> Dict[str, Any]:
        """评估单个模型"""
        start_time = time.time()
        
        try:
            # 使用原有的评估函数
            results = evaluate_model(self.config, model_name, self.logger)
            
            evaluation_time = time.time() - start_time
            
            if results is not None:
                return {
                    'status': 'success',
                    'evaluation_time': evaluation_time,
                    'metrics': results
                }
            else:
                return {
                    'status': 'failed',
                    'evaluation_time': evaluation_time,
                    'error': 'No results returned'
                }
                
        except Exception as e:
            evaluation_time = time.time() - start_time
            return {
                'status': 'failed',
                'evaluation_time': evaluation_time,
                'error': str(e)
            }
    
    def _analyze_model_performance(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """分析模型性能"""
        self.logger.info("分析模型性能...")
        
        # 提取成功评估的结果
        successful_results = {
            name: result for name, result in evaluation_results.items()
            if result.get('status') == 'success' and 'metrics' in result
        }
        
        if not successful_results:
            self.logger.warning("没有成功的评估结果可供分析")
            return {}
        
        # 创建性能数据框
        performance_data = []
        
        for model_name, result in successful_results.items():
            metrics = result['metrics']
            category = self._get_model_category(model_name)
            
            performance_data.append({
                'model': model_name,
                'category': category,
                'smoothness': metrics.get('smoothness', 0),
                'task_completion': metrics.get('task_completion', 0),
                'diversity': metrics.get('diversity', 0),
                'feasibility': metrics.get('feasibility', 0),
                'efficiency': metrics.get('efficiency', 0),
                'training_time': evaluation_results[model_name].get('evaluation_time', 0)
            })
        
        df_performance = pd.DataFrame(performance_data)
        
        # 计算统计信息
        analysis = {
            'total_models_evaluated': len(successful_results),
            'category_statistics': {},
            'overall_rankings': {},
            'performance_correlations': {},
            'best_performers': {}
        }
        
        # 按类别统计
        for category in df_performance['category'].unique():
            category_df = df_performance[df_performance['category'] == category]
            analysis['category_statistics'][category] = {
                'model_count': len(category_df),
                'avg_smoothness': float(category_df['smoothness'].mean()),
                'avg_task_completion': float(category_df['task_completion'].mean()),
                'avg_diversity': float(category_df['diversity'].mean()),
                'avg_feasibility': float(category_df['feasibility'].mean()),
                'avg_efficiency': float(category_df['efficiency'].mean()),
                'avg_training_time': float(category_df['training_time'].mean())
            }
        
        # 整体排名
        metrics_to_rank = ['smoothness', 'task_completion', 'diversity', 'feasibility', 'efficiency']
        
        for metric in metrics_to_rank:
            if metric in df_performance.columns:
                ranking = df_performance.nlargest(5, metric)[['model', 'category', metric]]
                analysis['overall_rankings'][metric] = ranking.to_dict('records')
        
        # 计算综合评分
        weights = {
            'smoothness': 0.2,
            'task_completion': 0.25,
            'diversity': 0.2,
            'feasibility': 0.2,
            'efficiency': 0.15
        }
        
        df_performance['composite_score'] = sum(
            df_performance[metric] * weight 
            for metric, weight in weights.items()
            if metric in df_performance.columns
        )
        
        # 最佳表现者
        top_models = df_performance.nlargest(10, 'composite_score')
        analysis['best_performers'] = {
            'top_10_models': top_models[['model', 'category', 'composite_score']].to_dict('records'),
            'best_by_category': {}
        }
        
        for category in df_performance['category'].unique():
            category_best = df_performance[df_performance['category'] == category].nlargest(1, 'composite_score')
            if not category_best.empty:
                analysis['best_performers']['best_by_category'][category] = {
                    'model': category_best.iloc[0]['model'],
                    'score': float(category_best.iloc[0]['composite_score'])
                }
        
        # 保存性能分析
        df_performance.to_csv(self.benchmark_dir / "performance_analysis.csv", index=False)
        
        with open(self.benchmark_dir / "performance_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _get_model_category(self, model_name: str) -> str:
        """获取模型所属类别"""
        for category_name, category_info in self.model_categories.items():
            if model_name in category_info['models']:
                return category_name
        return "Unknown"
    
    def create_benchmark_summary(self, comprehensive_report: Dict[str, Any]):
        """创建基准测试摘要"""
        self.logger.info("创建基准测试摘要...")
        
        summary = {
            'benchmark_overview': {
                'total_models': len(self.enabled_models),
                'successful_training': len([
                    r for r in comprehensive_report.get('training_results', {}).values()
                    if r.get('status') == 'success'
                ]),
                'successful_evaluation': len([
                    r for r in comprehensive_report.get('evaluation_results', {}).values()
                    if r.get('status') == 'success'
                ]),
                'categories': list(self.model_categories.keys())
            },
            'key_findings': self._extract_key_findings(comprehensive_report),
            'recommendations': self._generate_recommendations(comprehensive_report)
        }
        
        # 保存摘要
        with open(self.benchmark_dir / "benchmark_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 创建Markdown摘要
        self._create_markdown_summary(summary, comprehensive_report)
        
        return summary
    
    def _extract_key_findings(self, comprehensive_report: Dict[str, Any]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        performance_analysis = comprehensive_report.get('performance_analysis', {})
        
        if 'best_performers' in performance_analysis:
            best_performers = performance_analysis['best_performers']
            
            # 总体最佳模型
            if 'top_10_models' in best_performers and best_performers['top_10_models']:
                best_model = best_performers['top_10_models'][0]
                findings.append(f"总体最佳模型: {best_model['model']} ({best_model['category']}) - 综合评分: {best_model['composite_score']:.4f}")
            
            # 各类别最佳模型
            if 'best_by_category' in best_performers:
                for category, info in best_performers['best_by_category'].items():
                    findings.append(f"{category}类别最佳: {info['model']} - 评分: {info['score']:.4f}")
        
        # 类别统计
        if 'category_statistics' in performance_analysis:
            category_stats = performance_analysis['category_statistics']
            
            # 找出表现最好的类别
            category_scores = {
                cat: (stats['avg_task_completion'] + stats['avg_efficiency']) / 2
                for cat, stats in category_stats.items()
            }
            
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                findings.append(f"表现最佳的模型类别: {best_category}")
        
        return findings
    
    def _generate_recommendations(self, comprehensive_report: Dict[str, Any]) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        performance_analysis = comprehensive_report.get('performance_analysis', {})
        
        if 'category_statistics' in performance_analysis:
            category_stats = performance_analysis['category_statistics']
            
            # 基于性能特点的推荐
            for category, stats in category_stats.items():
                if stats['avg_efficiency'] > 0.8:
                    recommendations.append(f"{category}适合对效率要求高的应用场景")
                
                if stats['avg_smoothness'] < 0.1:
                    recommendations.append(f"{category}适合对轨迹平滑度要求高的应用")
                
                if stats['avg_training_time'] < 300:  # 5分钟
                    recommendations.append(f"{category}训练速度快，适合快速原型开发")
        
        # 通用推荐
        recommendations.extend([
            "对于实时应用，推荐使用Classical Methods或Linear Architecture",
            "对于复杂任务，推荐使用Transformer或Diffusion-based Methods",
            "对于需要与环境交互的任务，推荐使用RL-based Methods",
            "建议根据具体应用场景选择合适的模型类别"
        ])
        
        return recommendations
    
    def _create_markdown_summary(self, summary: Dict[str, Any], comprehensive_report: Dict[str, Any]):
        """创建Markdown格式的摘要"""
        markdown_content = f"""# 轨迹生成模型综合基准测试报告

## 基准测试概览

- **测试模型总数**: {summary['benchmark_overview']['total_models']}
- **训练成功模型**: {summary['benchmark_overview']['successful_training']}
- **评估成功模型**: {summary['benchmark_overview']['successful_evaluation']}
- **模型类别**: {', '.join(summary['benchmark_overview']['categories'])}

## 关键发现

"""
        
        for finding in summary['key_findings']:
            markdown_content += f"- {finding}\n"
        
        markdown_content += """

## 模型类别性能分析

"""
        
        performance_analysis = comprehensive_report.get('performance_analysis', {})
        if 'category_statistics' in performance_analysis:
            for category, stats in performance_analysis['category_statistics'].items():
                markdown_content += f"""### {category}

- **模型数量**: {stats['model_count']}
- **平均任务完成度**: {stats['avg_task_completion']:.4f}
- **平均效率**: {stats['avg_efficiency']:.4f}
- **平均平滑度**: {stats['avg_smoothness']:.4f}
- **平均训练时间**: {stats['avg_training_time']:.2f}秒

"""
        
        markdown_content += """## 推荐建议

"""
        
        for recommendation in summary['recommendations']:
            markdown_content += f"- {recommendation}\n"
        
        markdown_content += f"""

## 详细结果文件

- `performance_analysis.csv`: 详细性能数据
- `training_results.json`: 训练结果详情
- `evaluation_results.json`: 评估结果详情
- `benchmark_visualizations/`: 可视化图表目录

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(self.benchmark_dir / "benchmark_summary.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _generate_comprehensive_report(self, training_results: Dict[str, Dict[str, Any]],
                                     evaluation_results: Dict[str, Dict[str, Any]],
                                     performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合报告"""
        comprehensive_report = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'performance_analysis': performance_analysis,
            'benchmark_config': self.benchmark_config,
            'experiment_config': {
                'seed': self.config['experiment']['seed'],
                'device': self.config['training']['device'],
                'data_size': self.config['data']['generation']['num_trajectories']
            }
        }
        
        # 保存综合报告
        with open(self.benchmark_dir / "comprehensive_report.json", 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        return comprehensive_report
    
    def _create_benchmark_visualizations(self, comprehensive_report: Dict[str, Any]):
        """创建基准测试可视化"""
        self.logger.info("创建基准测试可视化...")
        
        viz_dir = self.benchmark_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        performance_analysis = comprehensive_report.get('performance_analysis', {})
        
        if 'category_statistics' in performance_analysis:
            self._create_category_comparison_charts(performance_analysis['category_statistics'], viz_dir)
        
        if 'best_performers' in performance_analysis:
            self._create_ranking_charts(performance_analysis['best_performers'], viz_dir)
    
    def _create_category_comparison_charts(self, category_stats: Dict[str, Any], viz_dir: Path):
        """创建类别比较图表"""
        # 准备数据
        categories = list(category_stats.keys())
        metrics = ['avg_task_completion', 'avg_efficiency', 'avg_smoothness', 'avg_diversity', 'avg_feasibility']
        
        data = []
        for category in categories:
            stats = category_stats[category]
            for metric in metrics:
                if metric in stats:
                    data.append({
                        'category': category,
                        'metric': metric.replace('avg_', ''),
                        'value': stats[metric]
                    })
        
        df = pd.DataFrame(data)
        
        # 创建分组柱状图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 使用seaborn创建分组柱状图
        sns.barplot(data=df, x='metric', y='value', hue='category', ax=ax)
        
        ax.set_title('模型类别性能比较', fontsize=14, fontweight='bold')
        ax.set_xlabel('性能指标')
        ax.set_ylabel('平均值')
        ax.legend(title='模型类别', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "category_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ranking_charts(self, best_performers: Dict[str, Any], viz_dir: Path):
        """创建排名图表"""
        if 'top_10_models' in best_performers:
            top_models = best_performers['top_10_models']
            
            # 创建排名柱状图
            df_top = pd.DataFrame(top_models)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bars = ax.barh(range(len(df_top)), df_top['composite_score'])
            ax.set_yticks(range(len(df_top)))
            ax.set_yticklabels([f"{row['model']}\n({row['category']})" for _, row in df_top.iterrows()])
            ax.set_xlabel('综合评分')
            ax.set_title('模型综合性能排名 (Top 10)', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "top_models_ranking.png", dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合基准测试脚本")
    parser.add_argument("--config", type=str, default="config_extended.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--max-parallel", type=int, default=3, help="最大并行模型数")
    parser.add_argument("--timeout", type=int, default=3600, help="每个模型的超时时间（秒）")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练阶段")
    parser.add_argument("--skip-evaluation", action="store_true", help="跳过评估阶段")
    parser.add_argument("--models", type=str, nargs='+', help="指定要测试的模型")
    parser.add_argument("--categories", type=str, nargs='+', help="指定要测试的模型类别")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    
    if args.max_parallel:
        config.setdefault('benchmark', {})['max_parallel_models'] = args.max_parallel
    
    if args.timeout:
        config.setdefault('benchmark', {})['timeout_per_model'] = args.timeout
    
    # 如果指定了特定模型或类别，更新配置
    if args.models:
        for model_name in config['models']:
            config['models'][model_name]['enabled'] = model_name in args.models
    
    if args.categories:
        enabled_models = []
        for category in args.categories:
            if category in config.get('model_categories', {}):
                enabled_models.extend(config['model_categories'][category]['models'])
        
        for model_name in config['models']:
            config['models'][model_name]['enabled'] = model_name in enabled_models
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("comprehensive_benchmark", 
                         Path(config['experiment']['output_dir']) / "logs", 
                         level=log_level)
    
    # 设置随机种子
    set_seed(config['experiment']['seed'])
    
    try:
        # 创建基准测试器
        benchmark = ComprehensiveBenchmark(config, logger)
        
        # 运行基准测试
        comprehensive_report = benchmark.run_comprehensive_benchmark()
        
        # 创建摘要
        summary = benchmark.create_benchmark_summary(comprehensive_report)
        
        logger.info(f"基准测试完成! 结果保存在: {benchmark.benchmark_dir}")
        logger.info(f"成功训练模型: {summary['benchmark_overview']['successful_training']}")
        logger.info(f"成功评估模型: {summary['benchmark_overview']['successful_evaluation']}")
        
    except Exception as e:
        logger.error(f"基准测试过程中出现错误: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()