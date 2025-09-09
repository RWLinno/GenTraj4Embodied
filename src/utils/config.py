"""
Configuration management utilities
配置管理工具
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理环境变量替换
    config = _replace_env_vars(config)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
        
    Raises:
        ValueError: 配置无效时抛出异常
    """
    required_sections = ['project', 'data', 'models', 'training', 'evaluation', 'experiment']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的部分: {section}")
    
    # 验证数据配置
    _validate_data_config(config['data'])
    
    # 验证模型配置
    _validate_models_config(config['models'])
    
    # 验证训练配置
    _validate_training_config(config['training'])
    
    # 验证评估配置
    _validate_evaluation_config(config['evaluation'])
    
    return True


def _validate_data_config(data_config: Dict[str, Any]):
    """验证数据配置"""
    required_fields = ['generation', 'tasks', 'split']
    
    for field in required_fields:
        if field not in data_config:
            raise ValueError(f"数据配置缺少必需字段: {field}")
    
    # 验证数据分割比例
    split_config = data_config['split']
    total_split = split_config['train'] + split_config['val'] + split_config['test']
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"数据分割比例总和必须为1.0，当前为: {total_split}")
    
    # 验证任务权重
    tasks = data_config['tasks']
    total_weight = sum(task['weight'] for task in tasks)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"任务权重总和必须为1.0，当前为: {total_weight}")


def _validate_models_config(models_config: Dict[str, Any]):
    """验证模型配置"""
    supported_models = ['diffusion_policy', 'transformer', 'vae', 'mlp', 'gflownets']
    
    for model_name in models_config:
        if model_name not in supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_config = models_config[model_name]
        required_fields = ['name', 'enabled', 'architecture', 'training']
        
        for field in required_fields:
            if field not in model_config:
                raise ValueError(f"模型 {model_name} 配置缺少必需字段: {field}")


def _validate_training_config(training_config: Dict[str, Any]):
    """验证训练配置"""
    required_fields = ['device', 'dataloader']
    
    for field in required_fields:
        if field not in training_config:
            raise ValueError(f"训练配置缺少必需字段: {field}")
    
    # 验证设备配置
    device = training_config['device']
    if device not in ['cpu', 'cuda', 'auto']:
        raise ValueError(f"不支持的设备类型: {device}")


def _validate_evaluation_config(evaluation_config: Dict[str, Any]):
    """验证评估配置"""
    required_fields = ['metrics', 'generation']
    
    for field in required_fields:
        if field not in evaluation_config:
            raise ValueError(f"评估配置缺少必需字段: {field}")
    
    # 验证指标权重
    metrics = evaluation_config['metrics']
    total_weight = sum(metric['weight'] for metric in metrics)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"评估指标权重总和必须为1.0，当前为: {total_weight}")


def _replace_env_vars(config: Any) -> Any:
    """
    递归替换配置中的环境变量
    
    Args:
        config: 配置对象
        
    Returns:
        替换后的配置
    """
    if isinstance(config, dict):
        return {key: _replace_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        # 替换 ${VAR_NAME} 格式的环境变量
        if config.startswith('${') and config.endswith('}'):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config
    else:
        return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    使用点分隔的路径获取配置值
    
    Args:
        config: 配置字典
        key_path: 点分隔的键路径，如 'models.diffusion_policy.enabled'
        default: 默认值
        
    Returns:
        配置值
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def update_config_value(config: Dict[str, Any], key_path: str, value: Any):
    """
    使用点分隔的路径更新配置值
    
    Args:
        config: 配置字典
        key_path: 点分隔的键路径
        value: 新值
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value