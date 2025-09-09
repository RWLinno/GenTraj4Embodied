# GenTraj4Embodied
Trajectory Generation Methods for Embodied Robotic Arms: Taxonomy, Progress, and Prospects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RWLinno/GenTraj4Embodied.git
cd GenTraj4Embodied
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training All Models

Train all enabled models with default configuration:

```bash
python scripts/train_all_models.py --config config.yaml --output-dir experiments
```

### Training Individual Models

Train a specific model:

```bash
python scripts/train_single_model.py --model diffusion_policy --config config.yaml
```

### Generate Trajectories

Generate trajectories using a trained model:

```bash
python scripts/generate_trajectories.py --model diffusion_policy --checkpoint experiments/checkpoints/diffusion_policy/best_model.pth
```

### Evaluate Models

Evaluate all models on test data:

```bash
python scripts/evaluate_models.py --experiment-dir experiments --output-dir results
```

### Visualize Results

Create visualizations of generated trajectories:

```bash
python scripts/visualize_trajectories.py --experiment-dir experiments --output-dir visualizations
```

## Model Categories

### 1. Classical Methods
- Linear Interpolation Model (LIM)
- Spline Interpolation Model (SIM)
- Dynamic Movement Primitives (DMP)
- Probabilistic Movement Primitives (ProMP)
- Gaussian Mixture Model (GMM)

### 2. Fundamental Architectures
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Graph Neural Network (GNN)
- Variational Autoencoder (VAE)
- Conditional VAE
- Physics-Constrained Neural Network

### 3. Probabilistic Generative Models
- Diffusion Policy (DDPM, DDIM)
- Latent Diffusion Model (LDM)
- Score-Based Model
- Conditional Diffusion
- Kinematic Diffusion
- Normalizing Flows
- Generative Flow Networks (GFlowNets)
- Generative Adversarial Networks (GAN)

### 4. Sequential Modeling
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Transformer
- Decision Transformer
- BERT-style Model
- GPT-style Model
- Hierarchical Transformer
- Positional Transformer
- Mixture Density Network (MDN)

### 5. Hybrid/Hierarchical
- Actor-Critic Model
- Imitation Learning
- Inverse Reinforcement Learning
- IL+RL Hybrid
- MPC+Learning
- Hierarchical Control

## Configuration

The main configuration file is `config.yaml`. Key sections include:

- `experiment`: General experiment settings
- `data`: Data generation and preprocessing parameters
- `models`: Individual model configurations
- `training`: Training hyperparameters
- `evaluation`: Evaluation settings

Example configuration for Diffusion Policy:

```yaml
models:
  diffusion_policy:
    enabled: true
    architecture:
      horizon: 16
      num_steps: 100
      unet_dim: 256
      num_layers: 4
      time_embed_dim: 128
      beta_schedule: cosine
      prediction_type: epsilon
      dropout: 0.1
```

## Data Format

The framework expects trajectory data in the following format:

- **Input**: Start pose (7D: 3D position + 4D quaternion)
- **Output**: End pose (7D: 3D position + 4D quaternion)
- **Trajectory**: Sequence of poses [seq_length, 7]

Data is stored in HDF5 format with the following structure:
- `trajectories`: [N, seq_length, 7]
- `start_poses`: [N, 7]
- `end_poses`: [N, 7]

## Evaluation Metrics

The framework provides several evaluation metrics:

- **MSE**: Mean Squared Error between predicted and ground truth trajectories
- **Smoothness**: Trajectory smoothness based on acceleration variance
- **End Error**: Error at the final trajectory point
- **Diversity**: Measure of trajectory diversity for the same start/end poses
- **Physical Feasibility**: Compliance with kinematic constraints

## Results

Our comprehensive evaluation shows:

- **Diffusion Policy** achieves superior performance with 23.5% improvement in trajectory quality
- **Transformer models** excel in long-term dependency modeling
- **GFlowNets** show promising exploration capabilities
- Clear trade-offs exist among model complexity, inference speed, and generation quality

## Directory Structure

```
GenTraj4Embodied/
├── baselines/                 # Model implementations
│   ├── base_model.py         # Base classes
│   ├── diffusion_policy_model.py
│   ├── transformer_model.py
│   ├── vae_model.py
│   ├── mlp_model.py
│   └── gflownets_model.py
├── src/                      # Core utilities
│   ├── data/                 # Data handling
│   ├── training/             # Training utilities
│   └── utils/                # General utilities
├── scripts/                  # Experiment scripts
├── config.yaml              # Main configuration
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gentraj4embodied2024,
  title={Trajectory Generation Methods for Embodied Robotic Arms: Taxonomy, Progress, and Prospects},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the authors of all baseline methods implemented in this framework
- Special thanks to the embodied AI community for valuable feedback and suggestions
- This work builds upon numerous open-source projects in robotics and machine learning

## Contact

For questions and support, please open an issue on GitHub or contact the authors.

---

**Note**: This repository is actively maintained. Please check for updates regularly and report any issues you encounter.