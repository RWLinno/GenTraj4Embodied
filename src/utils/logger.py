"""
Logging utilities for trajectory generation
轨迹生成日志工具
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import os


def setup_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    file_prefix: str = "trajectory_generation"
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        level: 日志级别
        format_string: 日志格式字符串
        file_prefix: 日志文件前缀
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 默认格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{file_prefix}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 同时创建一个最新日志的软链接
        latest_log = log_dir / f"{file_prefix}_latest.log"
        if latest_log.exists():
            latest_log.unlink()
        
        try:
            # 在Unix系统上创建软链接
            if os.name != 'nt':
                latest_log.symlink_to(log_file.name)
        except (OSError, NotImplementedError):
            # 在Windows或不支持软链接的系统上，直接复制
            pass
    
    return logger


class TrainingLogger:
    """训练过程专用日志记录器"""
    
    def __init__(self, logger: logging.Logger, log_interval: int = 100):
        """
        初始化训练日志记录器
        
        Args:
            logger: 基础日志记录器
            log_interval: 日志记录间隔
        """
        self.logger = logger
        self.log_interval = log_interval
        self.step_count = 0
        self.epoch_count = 0
        
    def log_step(self, loss: float, metrics: dict = None, force_log: bool = False):
        """
        记录训练步骤
        
        Args:
            loss: 损失值
            metrics: 其他指标
            force_log: 强制记录日志
        """
        self.step_count += 1
        
        if force_log or self.step_count % self.log_interval == 0:
            message = f"Step {self.step_count}: Loss = {loss:.6f}"
            
            if metrics:
                metric_strs = [f"{k} = {v:.6f}" for k, v in metrics.items()]
                message += f", {', '.join(metric_strs)}"
            
            self.logger.info(message)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None, 
                  metrics: dict = None):
        """
        记录训练轮次
        
        Args:
            epoch: 轮次编号
            train_loss: 训练损失
            val_loss: 验证损失
            metrics: 其他指标
        """
        self.epoch_count = epoch
        
        message = f"Epoch {epoch}: Train Loss = {train_loss:.6f}"
        
        if val_loss is not None:
            message += f", Val Loss = {val_loss:.6f}"
        
        if metrics:
            metric_strs = [f"{k} = {v:.6f}" for k, v in metrics.items()]
            message += f", {', '.join(metric_strs)}"
        
        self.logger.info(message)
    
    def log_model_info(self, model, total_params: int = None):
        """
        记录模型信息
        
        Args:
            model: 模型对象
            total_params: 总参数数量
        """
        model_name = model.__class__.__name__
        self.logger.info(f"Model: {model_name}")
        
        if total_params is not None:
            self.logger.info(f"Total parameters: {total_params:,}")
        else:
            # 尝试计算参数数量
            try:
                import torch
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    self.logger.info(f"Total parameters: {total_params:,}")
                    self.logger.info(f"Trainable parameters: {trainable_params:,}")
            except:
                pass
    
    def log_data_info(self, dataset):
        """
        记录数据集信息
        
        Args:
            dataset: 数据集对象
        """
        if hasattr(dataset, '__len__'):
            self.logger.info(f"Dataset size: {len(dataset)}")
        
        if hasattr(dataset, 'train_size'):
            self.logger.info(f"Train size: {dataset.train_size}")
        if hasattr(dataset, 'val_size'):
            self.logger.info(f"Validation size: {dataset.val_size}")
        if hasattr(dataset, 'test_size'):
            self.logger.info(f"Test size: {dataset.test_size}")


class EvaluationLogger:
    """评估过程专用日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        """
        初始化评估日志记录器
        
        Args:
            logger: 基础日志记录器
        """
        self.logger = logger
    
    def log_metrics(self, metrics: dict, prefix: str = ""):
        """
        记录评估指标
        
        Args:
            metrics: 指标字典
            prefix: 前缀字符串
        """
        if prefix:
            prefix = f"{prefix} - "
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{prefix}{metric_name}: {value:.6f}")
            else:
                self.logger.info(f"{prefix}{metric_name}: {value}")
    
    def log_comparison(self, model_results: dict):
        """
        记录模型比较结果
        
        Args:
            model_results: 模型结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("Model Comparison Results")
        self.logger.info("=" * 60)
        
        for model_name, results in model_results.items():
            self.logger.info(f"\n{model_name}:")
            self.log_metrics(results, prefix="  ")
    
    def log_best_model(self, best_model: str, best_score: float, metric_name: str):
        """
        记录最佳模型
        
        Args:
            best_model: 最佳模型名称
            best_score: 最佳分数
            metric_name: 指标名称
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Best Model: {best_model}")
        self.logger.info(f"Best {metric_name}: {best_score:.6f}")
        self.logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """
    获取已存在的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: int):
    """
    设置日志级别
    
    Args:
        logger: 日志记录器
        level: 日志级别
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_file_handler(logger: logging.Logger, file_path: Union[str, Path], 
                    level: int = logging.INFO):
    """
    为现有日志记录器添加文件处理器
    
    Args:
        logger: 日志记录器
        file_path: 日志文件路径
        level: 日志级别
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)