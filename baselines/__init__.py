"""
Trajectory Generation Models - Restructured Baselines
所有轨迹生成模型的统一入口点

5大类分类：
1. Classical Methods - 经典方法
2. Fundamental Architectures - 基础架构  
3. Probabilistic Generative Models - 概率生成模型
4. Sequential Modeling - 序列建模
5. Hybrid_Hierarchical - 混合分层
"""

from .base_model import BaseTrajectoryModel

# Classical Methods - 这些模型通常依赖较少
try:
    from .LinearInterpolation import LinearInterpolationModel
except ImportError as e:
    print(f"Warning: Failed to import LinearInterpolationModel: {e}")
    LinearInterpolationModel = None

try:
    from .SplineInterpolation import SplineInterpolationModel
except ImportError as e:
    print(f"Warning: Failed to import SplineInterpolationModel: {e}")
    SplineInterpolationModel = None

try:
    from .DMP import DMPTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import DMPTrajectoryModel: {e}")
    DMPTrajectoryModel = None

try:
    from .ProMP import ProMPTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import ProMPTrajectoryModel: {e}")
    ProMPTrajectoryModel = None

try:
    from .GMM import GMMTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import GMMTrajectoryModel: {e}")
    GMMTrajectoryModel = None

# Fundamental Architectures - 基础模型
try:
    from .mlp_model import MLPTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import MLPTrajectoryModel: {e}")
    MLPTrajectoryModel = None

try:
    from .cnn_model import CNNTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import CNNTrajectoryModel: {e}")
    CNNTrajectoryModel = None

try:
    from .vae_model import VAETrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import VAETrajectoryModel: {e}")
    VAETrajectoryModel = None

# Sequential Modeling - 序列模型
try:
    from .lstm_model import LSTMTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import LSTMTrajectoryModel: {e}")
    LSTMTrajectoryModel = None

try:
    from .gru_model import GRUTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import GRUTrajectoryModel: {e}")
    GRUTrajectoryModel = None

try:
    from .transformer_model import TransformerTrajectoryModel
except ImportError as e:
    print(f"Warning: Failed to import TransformerTrajectoryModel: {e}")
    TransformerTrajectoryModel = None

# 模型注册表 - 只包含成功导入的模型
MODEL_REGISTRY = {}

# 添加成功导入的模型
if LinearInterpolationModel:
    MODEL_REGISTRY["linear_interpolation"] = LinearInterpolationModel
if SplineInterpolationModel:
    MODEL_REGISTRY["spline_interpolation"] = SplineInterpolationModel
if DMPTrajectoryModel:
    MODEL_REGISTRY["dmp"] = DMPTrajectoryModel
if ProMPTrajectoryModel:
    MODEL_REGISTRY["promp"] = ProMPTrajectoryModel
if GMMTrajectoryModel:
    MODEL_REGISTRY["gmm"] = GMMTrajectoryModel
if MLPTrajectoryModel:
    MODEL_REGISTRY["mlp"] = MLPTrajectoryModel
if CNNTrajectoryModel:
    MODEL_REGISTRY["cnn"] = CNNTrajectoryModel
if VAETrajectoryModel:
    MODEL_REGISTRY["vae"] = VAETrajectoryModel
if LSTMTrajectoryModel:
    MODEL_REGISTRY["lstm"] = LSTMTrajectoryModel
if GRUTrajectoryModel:
    MODEL_REGISTRY["gru"] = GRUTrajectoryModel
if TransformerTrajectoryModel:
    MODEL_REGISTRY["transformer"] = TransformerTrajectoryModel

def get_model_class(model_name):
    """根据模型名称获取模型类"""
    return MODEL_REGISTRY.get(model_name)

def list_available_models():
    """列出所有可用的模型"""
    return list(MODEL_REGISTRY.keys())

__all__ = [
    "BaseTrajectoryModel",
    "MODEL_REGISTRY", 
    "get_model_class",
    "list_available_models"
] + list(MODEL_REGISTRY.keys())