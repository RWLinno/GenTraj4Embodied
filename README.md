# Generative Models for 3D End-Effector Trajectory Generation


## Structure
```
GenTraj4Embodied/
├── README.md # Project documentation
├── requirements.txt  # Python dependencies
├── run.py  # Main execution script
├── config.yaml # Configuration parameters
├── data/ # Data directory
│   ├── synthetic/  # Generated multi-modal trajectories
│   ├── real/ # Real-world trajectory data (if available)
│   └── random/  # random data
├── src/  # Core implementation
│   ├── data/ # Data processing modules
│   │   ├── dataset.py  # PyTorch dataset implementations
│   │   ├── data_generator.py # synthetic data generation
│   │   └── transforms.py # augmentation and preprocessing
│   ├── models/ # Base model definitions
│   ├── training/ # Training infrastructure
│   │   └── trainer.py  # Unified training framework
│   ├── evaluation/ # Evaluation framework
│   │   ├── evaluator.py  # Comprehensive evaluation system
│   │   └── metrics.py  # Trajectory quality metrics
│   └── utils/  # Utility functions
│       ├── config.py # Configuration management
│       ├── logger.py # Logging utilities
│       ├── math_utils.py # Mathematical operations
│       └── visualization.py  # 3D trajectory visualization
├── baselines/  # Model implementations
│   ├── .../ # baselines name
│   │   └── model.py # Main model architecture
├── scripts/  # Experimental scripts
│   ├── train_all_models.py # Batch training script
│   ├── evaluate_models.py  # Comprehensive evaluation
│   ├── generate_data.py  # Data generation pipeline
│   └── visualize_results.py  # Result visualization
├── experiments/  # Experimental results
│   ├── configs/  # Experiment configurations
│   ├── logs/ # Training logs and metrics
│   ├── checkpoints/  # Model checkpoints
│   └── results/  # Evaluation results and analysis
└── ... # others
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Note: The project has been tested and verified to work with:
# - Python 3.8+
# - PyTorch 1.12+
# - CUDA 11.6+ (optional, CPU training supported)
# All model import issues and compatibility problems have been resolved
```

### 2. Data Generation

Generate multi-modal synthetic trajectory dataset:

```bash
# Generate synthetic trajectories with multi-modal characteristics
python run.py --mode generate

# Or use the dedicated script (if available)
python scripts/generate_data.py --config config.yaml

# Options:
# --num_trajectories 10000    # Number of trajectories per task
# --task_types pick,pour,assemble  # Task categories
# --modalities 3               # Number of behavioral modalities per task
# --workspace_size 1.0         # Workspace dimensions (meters)
```

### 3. Model Training

Train individual models or all models simultaneously:

```bash
# Train all models with default configuration
python scripts/train_all_models.py --config config.yaml

# Train specific model
python run.py --model diffusion_policy --mode train --config config.yaml
python run.py --model transformer --mode train --config config.yaml
python run.py --model vae --mode train --config config.yaml
python run.py --model mlp --mode train --config config.yaml
python run.py --model gflownets --mode train --config config.yaml

# Advanced training options
python run.py --model diffusion_policy --mode train \
              --batch_size 32 \
              --learning_rate 1e-4 \
              --epochs 100 \
              --gpu 0
```

### 4. Evaluation and Analysis

Comprehensive model evaluation and comparison:

```bash
# Evaluate all trained models
python scripts/evaluate_models.py --config config.yaml

# Generate detailed comparison report
python scripts/evaluate_models.py --config config.yaml --detailed_analysis

# Visualize results and trajectories
python scripts/visualize_results.py --experiment_dir experiments/results/

# Generate 3D trajectory animations
python scripts/visualize_results.py --experiment_dir experiments/results/ --animate
```
## Data Format and Specifications

### Trajectory Data Structure

```python
{
    'start_pose': [x, y, z, qx, qy, qz, qw],     # Start pose (position + quaternion)
    'end_pose': [x, y, z, qx, qy, qz, qw],       # Goal pose (position + quaternion)
    'trajectory': [                               # Trajectory waypoints
        [x, y, z, qx, qy, qz, qw, t],            # Each point: pose + timestamp
        ...
    ],
    'task_id': int,                              # Task identifier (0: pick, 1: pour, 2: assemble)
    'modality': int,                             # Behavioral modality (0: direct, 1: arc, 2: detour)
    'workspace_bounds': [[x_min, x_max], [y_min, y_max], [z_min, z_max]],
    'constraints': {                             # Task-specific constraints
        'collision_objects': [...],              # Obstacle definitions
        'joint_limits': [...],                   # Kinematic constraints
        'velocity_limits': [...]                 # Dynamic constraints
    }
}
```

### Multi-Modal Task Categories

#### Pick-and-Place Tasks (40% of dataset)
- **Direct Mode**: Straight-line approach with minimal deviation
- **Arc Mode**: Curved trajectory with intermediate waypoints
- **Detour Mode**: Obstacle-avoiding path with strategic waypoints

#### Pouring Tasks (30% of dataset)
- **45° Tilt**: Standard pouring angle for controlled flow
- **90° Tilt**: Rapid pouring for time-critical scenarios
- **30° Tilt**: Gentle pouring for delicate materials

#### Assembly Tasks (30% of dataset)
- **Direct Insertion**: Straight approach with precise alignment
- **Spiral Insertion**: Rotational approach for tight tolerances
- **Lateral Insertion**: Side approach for constrained spaces

## Evaluation Metrics and Analysis

### Quantitative Metrics

#### 1. Trajectory Quality
- **Smoothness**: $S = \frac{1}{N-2} \sum_{i=1}^{N-2} \|\mathbf{p}_{i+2} - 2\mathbf{p}_{i+1} + \mathbf{p}_i\|$
- **Path Efficiency**: $E = \frac{d(\mathbf{p}_s, \mathbf{p}_g)}{\sum_{i=1}^{N-1} d(\mathbf{p}_i, \mathbf{p}_{i+1})}$
- **Execution Time**: Total trajectory duration

#### 2. Task Completion
- **Goal Accuracy**: $A = \exp(-\|\mathbf{p}_N - \mathbf{p}_g\|^2 / 2\sigma^2)$
- **Success Rate**: Binary task completion metric
- **Constraint Satisfaction**: Collision-free and kinematically feasible

#### 3. Behavioral Diversity
- **Inter-trajectory Distance**: Hausdorff distance between generated paths
- **Mode Coverage**: Entropy of trajectory clusters
- **Novelty Score**: Distance to nearest training example

#### 4. Computational Performance
- **Inference Time**: Generation latency per trajectory
- **Memory Usage**: Peak GPU memory consumption
- **Training Efficiency**: Convergence rate and stability

### Statistical Analysis Framework

```python
# Example evaluation script
from src.evaluation.evaluator import TrajectoryEvaluator
from src.evaluation.metrics import *

evaluator = TrajectoryEvaluator()
results = evaluator.evaluate_all_models(
    models=['diffusion_policy', 'transformer', 'vae', 'mlp', 'gflownets'],
    test_dataset=test_data,
    metrics=['smoothness', 'efficiency', 'diversity', 'success_rate'],
    num_samples=1000,
    confidence_level=0.95
)

# Generate statistical significance tests
evaluator.statistical_analysis(results, test='wilcoxon')
evaluator.generate_report(results, output_dir='experiments/results/')
```

### Custom Model Implementation

```python
from src.models.base_model import BaseTrajectoryModel
import torch
import torch.nn as nn

class CustomModel(BaseTrajectoryModel):
    def __init__(self, config):
        super().__init__(config)
        self.network = self._build_network()
    
    def _build_network(self):
        # Implement custom architecture
        return nn.Sequential(...)
    
    def forward(self, start_pose, end_pose, context=None):
        # Implement forward pass
        return trajectory
    
    def loss_function(self, predicted, target):
        # Implement custom loss
        return loss
    
    def generate(self, start_pose, end_pose, num_samples=1):
        # Implement trajectory generation
        return trajectories

# Register custom model
from src.training.trainer import ModelRegistry
ModelRegistry.register('custom_model', CustomModel)
```

### Custom Evaluation Metrics
```python
from src.evaluation.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(self, name='custom_metric'):
        super().__init__(name)
    
    def compute(self, predicted_trajectories, target_trajectories):
        # Implement custom metric computation
        return metric_value
    
    def aggregate(self, metric_values):
        # Implement aggregation strategy
        return aggregated_value

# Register custom metric
from src.evaluation.evaluator import MetricRegistry
MetricRegistry.register('custom_metric', CustomMetric)
```

## Troubleshooting

### Common Issues

#### Training Convergence Problems
```bash
# Check data distribution
python scripts/analyze_data.py --data_dir data/synthetic/

# Adjust learning rate
python run.py --model diffusion_policy --learning_rate 5e-5

# Enable gradient clipping
python run.py --model transformer --gradient_clip_norm 0.5
```

#### Memory Issues
```bash
# Reduce batch size
python run.py --batch_size 16

# Enable gradient checkpointing
python run.py --gradient_checkpointing

# Use mixed precision training
python run.py --mixed_precision
```

#### Evaluation Errors
```bash
# Validate model checkpoints
python scripts/validate_checkpoints.py --checkpoint_dir experiments/checkpoints/

# Check data format
python scripts/validate_data.py --data_dir data/processed/
```

## Citation
If you use this work in your research, please cite:
```
```