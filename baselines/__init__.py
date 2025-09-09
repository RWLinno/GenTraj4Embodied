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

# Classical Methods
try:
    from .linear_interpolation_model import LinearInterpolationModel
except ImportError:
    LinearInterpolationModel = None

try:
    from .spline_interpolation_model import SplineInterpolationModel  
except ImportError:
    SplineInterpolationModel = None

try:
    from .dmp_model import DMPModel
except ImportError:
    DMPModel = None

try:
    from .promp_model import ProMPModel
except ImportError:
    ProMPModel = None

try:
    from .gmm_model import GMMModel
except ImportError:
    GMMModel = None

# Fundamental Architectures
try:
    from .mlp_model import MLPModel
except ImportError:
    MLPModel = None

try:
    from .cnn_model import CNNModel
except ImportError:
    CNNModel = None

try:
    from .gnn_model import GNNModel
except ImportError:
    GNNModel = None

try:
    from .vae_model import VAEModel
except ImportError:
    VAEModel = None

# Probabilistic Generative Models
try:
    from .diffusion_policy_model import DiffusionPolicyModel
except ImportError:
    DiffusionPolicyModel = None

try:
    from .ddpm_model import DDPMModel
except ImportError:
    DDPMModel = None

try:
    from .ddim_model import DDIMModel
except ImportError:
    DDIMModel = None

try:
    from .difftraj_model import DiffTrajModel
except ImportError:
    DiffTrajModel = None

try:
    from .score_based_model import ScoreBasedModel
except ImportError:
    ScoreBasedModel = None

try:
    from .conditional_diffusion_model import ConditionalDiffusionModel
except ImportError:
    ConditionalDiffusionModel = None

try:
    from .latent_diffusion_model import LatentDiffusionModel
except ImportError:
    LatentDiffusionModel = None

try:
    from .normalizing_flows_model import NormalizingFlowsModel
except ImportError:
    NormalizingFlowsModel = None

try:
    from .gflownets_model import GFlowNetsModel
except ImportError:
    GFlowNetsModel = None

# Sequential Modeling
try:
    from .lstm_model import LSTMModel
except ImportError:
    LSTMModel = None

try:
    from .gru_model import GRUModel
except ImportError:
    GRUModel = None

try:
    from .rnn_model import RNNModel
except ImportError:
    RNNModel = None

try:
    from .transformer_model import TransformerModel
except ImportError:
    TransformerModel = None

try:
    from .gpt_model import GPTModel
except ImportError:
    GPTModel = None

try:
    from .bert_model import BERTModel
except ImportError:
    BERTModel = None

try:
    from .decision_transformer_model import DecisionTransformerModel
except ImportError:
    DecisionTransformerModel = None

try:
    from .mdn_model import MDNModel
except ImportError:
    MDNModel = None

try:
    from .seq2seq_model import Seq2SeqModel
except ImportError:
    Seq2SeqModel = None

# Hybrid_Hierarchical
try:
    from .policy_gradient_model import PolicyGradientModel
except ImportError:
    PolicyGradientModel = None

try:
    from .actor_critic_model import ActorCriticModel
except ImportError:
    ActorCriticModel = None

try:
    from .ppo_model import PPOModel
except ImportError:
    PPOModel = None

try:
    from .imitation_learning_model import ImitationLearningModel
except ImportError:
    ImitationLearningModel = None

try:
    from .inverse_rl_model import InverseRLModel
except ImportError:
    InverseRLModel = None

try:
    from .il_rl_hybrid_model import ILRLHybridModel
except ImportError:
    ILRLHybridModel = None

try:
    from .mpc_learning_model import MPCLearningModel
except ImportError:
    MPCLearningModel = None

# 模型注册表
MODEL_REGISTRY = {
    # Classical Methods
    "linear_interpolation": LinearInterpolationModel,
    "spline_interpolation": SplineInterpolationModel,
    "dmp": DMPModel,
    "promp": ProMPModel,
    "gmm": GMMModel,
    
    # Fundamental Architectures
    "mlp": MLPModel,
    "cnn": CNNModel,
    "gnn": GNNModel,
    "vae": VAEModel,
    
    # Probabilistic Generative Models
    "diffusion_policy": DiffusionPolicyModel,
    "ddpm": DDPMModel,
    "ddim": DDIMModel,
    "difftraj": DiffTrajModel,
    "score_based": ScoreBasedModel,
    "conditional_diffusion": ConditionalDiffusionModel,
    "latent_diffusion": LatentDiffusionModel,
    "normalizing_flows": NormalizingFlowsModel,
    "gflownets": GFlowNetsModel,
    
    # Sequential Modeling
    "lstm": LSTMModel,
    "gru": GRUModel,
    "rnn": RNNModel,
    "transformer": TransformerModel,
    "gpt": GPTModel,
    "bert": BERTModel,
    "decision_transformer": DecisionTransformerModel,
    "mdn": MDNModel,
    "seq2seq": Seq2SeqModel,
    
    # Hybrid_Hierarchical
    "policy_gradient": PolicyGradientModel,
    "actor_critic": ActorCriticModel,
    "ppo": PPOModel,
    "imitation_learning": ImitationLearningModel,
    "inverse_rl": InverseRLModel,
    "il_rl_hybrid": ILRLHybridModel,
    "mpc_learning": MPCLearningModel,
}

def get_model_class(model_name):
    """根据模型名称获取模型类"""
    return MODEL_REGISTRY.get(model_name)

def list_available_models():
    """列出所有可用的模型"""
    available = []
    for name, model_class in MODEL_REGISTRY.items():
        if model_class is not None:
            available.append(name)
    return available

__all__ = [
    "BaseTrajectoryModel",
    "MODEL_REGISTRY", 
    "get_model_class",
    "list_available_models"
] + [name for name in MODEL_REGISTRY.keys() if MODEL_REGISTRY[name] is not None]
