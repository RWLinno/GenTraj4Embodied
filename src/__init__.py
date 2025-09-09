"""
3D End-Effector Trajectory Generation Package
机械臂3D末端位姿轨迹生成包
"""

__version__ = "1.0.0"
__author__ = "Trajectory Generation Team"
__email__ = "trajectory@example.com"

from .utils.config import load_config
from .utils.logger import setup_logger
from .data.dataset import TrajectoryDataset
from .data.data_generator import TrajectoryDataGenerator

__all__ = [
    "load_config",
    "setup_logger", 
    "TrajectoryDataset",
    "TrajectoryDataGenerator"
]