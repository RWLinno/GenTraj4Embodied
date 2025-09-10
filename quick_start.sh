# generate data
python run.py --mode generate

# train baselines
python run.py --model mlp --mode train --config config.yaml

# evaluate
python run.py --model mlp --mode evaluate --config config.yaml

# analysis and visualization
python visualize_trajectories.py --model mlp --num-samples 5 --output-dir visualizations